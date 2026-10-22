"""Preserved upstream response contracts with CPU-only engine doubles.

Lifecycle defects captured during Stage A are now regression expectations.
These doubles do not establish compatibility with real vLLM.
"""

import asyncio
import importlib.util
import itertools
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@pytest.fixture
def legacy(monkeypatch):
    @dataclass
    class Args:
        model: str = ""
        dtype: str = "auto"
        load_format: str = ""
        enforce_eager: bool = False
        enable_prefix_caching: bool = True
        runner: str = "auto"
        convert: str = "auto"
        distributed_executor_backend: str = "mp"

    @dataclass
    class Output:
        request_id: str
        prompt_token_ids: list
        outputs: list
        metrics: object = None

    class Sampling:
        def __init__(self, **kwargs):
            if "invalid_sampling" in kwargs:
                raise ValueError("unsupported sampling")
            self.values = kwargs

    class Engine:
        def __init__(self):
            self.calls = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.fail_after_output = False
            self.pause_after_output = False
            self.finish = asyncio.Event()
            self.close_calls = 0

        async def generate(self, inputs, sampling, request_id):
            self.calls.append((inputs, sampling.values, request_id))
            self.started.set()
            await self.release.wait()
            yield Output(
                request_id,
                [10, 11],
                [
                    types.SimpleNamespace(
                        token_ids=[20],
                        text="answer",
                        logprobs=None,
                        finish_reason="stop",
                    )
                ],
            )
            if self.pause_after_output:
                await self.finish.wait()
            if self.fail_after_output:
                raise RuntimeError("engine failure")

        async def abort(self, request_id):
            self.release.set()
            self.finish.set()

        def shutdown(self):
            self.close_calls += 1

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.31.0"
    for name, value in {
        "AsyncEngineArgs": Args,
        "AsyncLLMEngine": object,
        "EmbeddingRequestOutput": Output,
        "PoolingParams": object,
        "PromptType": str,
        "RequestOutput": Output,
        "SamplingParams": Sampling,
    }.items():
        setattr(vllm, name, value)
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda **kwargs: kwargs
    utils = types.ModuleType("vllm.utils")
    utils.Counter = itertools.count
    for name, module in {
        "torch": torch,
        "vllm": vllm,
        "vllm.inputs": inputs,
        "vllm.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "sllm/backends/vllm_backend.py"
    spec = importlib.util.spec_from_file_location("legacy_vllm_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def create(config=None):
        backend = module.VllmBackend(
            "example",
            {
                "load_format": "safetensors",
                "pretrained_model_name_or_path": "example",
                **(config or {}),
            },
        )
        backend.status = module.BackendStatus.RUNNING
        backend.engine = Engine()
        return backend

    return create


@pytest.mark.asyncio
async def test_input_token_precedence_usage_and_request_identity(legacy):
    backend = legacy()
    backend.engine.release.set()
    result = await backend.generate(
        {
            "request_id": "one",
            "model": "logical",
            "input_tokens": [10, 11],
            "prompt": "ignored",
            "messages": [],
            "max_tokens": 1,
        }
    )
    assert backend.engine.calls[0][0] == {"prompt_token_ids": [10, 11]}
    assert result["id"] == "one"
    assert result["model"] == "logical"
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
    assert await backend.request_trace.request_count() == 0


@pytest.mark.asyncio
async def test_legacy_messages_are_joined_without_chat_template(legacy):
    backend = legacy()
    backend.engine.release.set()
    await backend.generate(
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }
    )
    assert backend.engine.calls[0][0] == "user: hello\nassistant: world"


@pytest.mark.asyncio
async def test_sampling_error_keeps_original_error_protocol(legacy):
    backend = legacy()
    result = await backend.generate({"invalid_sampling": True})
    assert result["error"].startswith("Invalid sampling parameters:")
    assert not backend.engine.calls


@pytest.mark.asyncio
async def test_admission_precedes_first_output(legacy):
    backend = legacy()
    request = asyncio.create_task(backend.generate({"prompt": "hello"}))
    await backend.engine.started.wait()
    try:
        # Admission must be visible even when the engine has not yielded.
        assert await backend.request_trace.request_count() == 1
    finally:
        backend.engine.release.set()
        await request


@pytest.mark.asyncio
async def test_engine_exception_unregisters_trace(legacy):
    backend = legacy()
    backend.engine.release.set()
    backend.engine.fail_after_output = True
    result = await backend.generate({"request_id": "failed", "prompt": "hello"})
    assert "engine failure" in result["error"]
    assert await backend.request_trace.request_count() == 0


@pytest.mark.asyncio
async def test_debug_history_is_separate_from_active(legacy):
    backend = legacy({"trace_debug": True})
    backend.engine.release.set()
    await backend.generate({"prompt": "hello"})
    assert await backend.request_trace.request_count() == 0
    assert backend.request_history


@pytest.mark.asyncio
async def test_replay_uses_one_token_without_claiming_kv_hit(legacy):
    backend = legacy()
    backend.engine.release.set()
    await backend.resume_kv_cache([[10, 11], [30, 31]])
    assert [call[0]["prompt_token_ids"] for call in backend.engine.calls] == [
        [10, 11],
        [30, 31],
    ]
    assert all(call[1] == {"max_tokens": 1} for call in backend.engine.calls)


def test_original_missing_continuation_method_is_explicit(legacy):
    assert not hasattr(legacy(), "resume_generate")


@pytest.mark.asyncio
async def test_stop_rejects_new_work_and_drains_observed_work(legacy):
    backend = legacy()
    engine = backend.engine
    engine.release.set()
    engine.pause_after_output = True
    request = asyncio.create_task(backend.generate({"prompt": "hello"}))
    while await backend.request_trace.request_count() == 0:
        await asyncio.sleep(0)
    stop = asyncio.create_task(backend.stop())
    await asyncio.sleep(0)
    try:
        assert not stop.done()
        assert await backend.generate({"prompt": "new"}) == {
            "error": "Engine is not running",
        }
    finally:
        engine.finish.set()
        await request
        await asyncio.wait_for(stop, timeout=3)
    # Closing the frontend must explicitly close its engine tree.
    assert engine.close_calls == 1


@pytest.mark.asyncio
async def test_parallel_requests_keep_separate_ids_and_sampling(legacy):
    backend = legacy()
    backend.engine.release.set()
    results = await asyncio.gather(
        *[
            backend.generate(
                {
                    "request_id": f"r{i}",
                    "prompt": str(i),
                    "max_tokens": i + 1,
                }
            )
            for i in range(3)
        ]
    )
    assert [result["id"] for result in results] == ["r0", "r1", "r2"]
    assert [call[1]["max_tokens"] for call in backend.engine.calls] == [1, 2, 3]


@pytest.fixture
def downloader(legacy, monkeypatch):
    legacy()  # Install the same strict CPU-only import doubles.
    torch = sys.modules["torch"]
    monkeypatch.setattr(
        torch, "nn", types.SimpleNamespace(Module=object), raising=False
    )
    ray = types.ModuleType("ray")
    ray.remote = lambda **kwargs: lambda target: target
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: pytest.fail(
        "unexpected download"
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    path = Path(__file__).parents[2] / "sllm/model_downloader.py"
    spec = importlib.util.spec_from_file_location(
        "legacy_downloader_probe", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VllmModelDownloader()


def test_downloader_delegates_even_when_output_exists(
    downloader, monkeypatch, tmp_path
):
    calls = []
    manifest = {"artifact": "verified"}
    preparation = types.ModuleType("sllm.vllm_preparation")

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return manifest

    preparation.prepare_vllm_model = prepare
    monkeypatch.setitem(sys.modules, "sllm.vllm_preparation", preparation)
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    (tmp_path / "vllm" / "logical").mkdir(parents=True)
    snapshot = {"source": "frozen"}
    result = downloader.download_vllm_model(
        "logical",
        "actual/repository",
        "bfloat16",
        2,
        revision="commit",
        source_snapshot=snapshot,
    )
    assert result is manifest
    assert calls == [
        (
            ("logical", "actual/repository", "bfloat16", 2),
            {
                "storage_path": str(tmp_path),
                "revision": "commit",
                "source_snapshot": snapshot,
            },
        )
    ]


def test_downloader_preserves_preparation_error(downloader, monkeypatch):
    preparation = types.ModuleType("sllm.vllm_preparation")
    failure = ValueError("verification failed")

    def prepare(*args, **kwargs):
        assert args[1] == "logical"
        raise failure

    preparation.prepare_vllm_model = prepare
    monkeypatch.setitem(sys.modules, "sllm.vllm_preparation", preparation)
    with pytest.raises(ValueError) as caught:
        downloader.download_vllm_model("logical", None, "float16")
    assert caught.value is failure


@pytest.mark.parametrize("kwargs", [{"pattern": "weight"}, {"max_size": 0}])
def test_downloader_rejects_partial_export(downloader, kwargs):
    with pytest.raises(ValueError, match="pattern|max_size"):
        downloader.download_vllm_model("logical", "actual", "float16", **kwargs)
