"""Characterize the unchanged upstream backend with CPU-only engine doubles.

These tests record inherited defects, rather than treating those defects as
desired behavior. They do not establish compatibility with real vLLM.
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
        task: str = "auto"

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
        backend = module.VllmBackend("example", config or {})
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
async def test_original_first_output_registration_gap(legacy):
    backend = legacy()
    request = asyncio.create_task(backend.generate({"prompt": "hello"}))
    await backend.engine.started.wait()
    try:
        # Known defect: accepted work is invisible to stop/shutdown until output.
        assert await backend.request_trace.request_count() == 0
    finally:
        backend.engine.release.set()
        await request


@pytest.mark.asyncio
async def test_original_engine_exception_leaks_trace(legacy):
    backend = legacy()
    backend.engine.release.set()
    backend.engine.fail_after_output = True
    with pytest.raises(RuntimeError, match="engine failure"):
        await backend.generate({"request_id": "failed", "prompt": "hello"})
    # Known defect, to replace with zero after lifecycle integration.
    assert await backend.request_trace.request_count() == 1


@pytest.mark.asyncio
async def test_original_debug_history_counts_as_active(legacy):
    backend = legacy({"trace_debug": True})
    backend.engine.release.set()
    await backend.generate({"prompt": "hello"})
    assert await backend.request_trace.request_count() == 1


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
    # Original shutdown deletes the reference but never closes EngineCore.
    assert engine.close_calls == 0


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


def test_existing_incomplete_directory_is_mistaken_for_export(
    downloader, monkeypatch, tmp_path
):
    constructed = []
    sys.modules["vllm"].LLM = lambda **kwargs: constructed.append(kwargs)
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    incomplete = tmp_path / "vllm" / "incomplete"
    incomplete.mkdir(parents=True)
    result = downloader.download_vllm_model(
        "incomplete",
        str(tmp_path / "source"),
        "bfloat16",
    )
    assert result is None
    assert not constructed
    assert not list(incomplete.iterdir())


def test_export_failure_propagates_and_removes_partial_output(
    downloader, monkeypatch, tmp_path
):
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "vllm" / "failed"

    def fail_export(**kwargs):
        output.mkdir(parents=True)
        (output / "incomplete").write_bytes(b"partial")
        raise RuntimeError("writer failed")

    core = types.SimpleNamespace(save_serverless_llm_state=fail_export)
    sys.modules["vllm"].LLM = lambda **kwargs: types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(engine_core=core),
    )
    with pytest.raises(RuntimeError, match="writer failed"):
        downloader.download_vllm_model("failed", str(source), "bfloat16")
    assert not output.exists()
