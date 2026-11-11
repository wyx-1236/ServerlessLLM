# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
"""Original backend contracts through the v0.31 adapter, with CPU doubles.

No real vLLM, Torch, model download or Store native extension is imported.
The actual backend and vllm_compat construction/abort/close paths execute.
"""

import asyncio
import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import make_dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture
def backend_module(monkeypatch):
    fields = json.loads(
        (
            ROOT / "tests/baseline/fixtures/vllm-0.31.0-engine-fields.json"
        ).read_text()
    )
    args = make_dataclass("Args", [(name, object, None) for name in fields])
    engines = []

    class Engine:
        @classmethod
        def from_engine_args(cls, engine_args):
            return cls(engine_args)

        def __init__(self, engine_args):
            self.args = engine_args
            self.calls = []
            self.encode_calls = []
            self.started = asyncio.Event()
            self.first_output = asyncio.Event()
            self.release = asyncio.Event()
            self.finish = asyncio.Event()
            self.release.set()
            self.finish.set()
            self.abort = AsyncMock(side_effect=self.release_request)
            self.shutdown = Mock()
            engines.append(self)

        async def release_request(self, request_id):
            self.release.set()
            self.finish.set()

        async def generate(self, inputs, sampling_params, request_id):
            self.calls.append((inputs, vars(sampling_params), request_id))
            self.started.set()
            await self.release.wait()
            tokens = (
                inputs["prompt_token_ids"]
                if isinstance(inputs, dict)
                else [1, 2, 3]
            )
            completion = [4, 5, 6][: getattr(sampling_params, "max_tokens", 3)]
            for final in (False, True):
                yield SimpleNamespace(
                    request_id=request_id,
                    prompt_token_ids=list(tokens),
                    metrics=None,
                    finished=final,
                    outputs=[
                        SimpleNamespace(
                            text="test output result" if final else "test",
                            token_ids=completion if final else completion[:1],
                            logprobs=None,
                            finish_reason="stop" if final else None,
                        )
                    ],
                )
                if not final:
                    self.first_output.set()
                    await self.finish.wait()

        async def encode(self, inputs, pooling_params, request_id):
            self.encode_calls.append((inputs, pooling_params, request_id))
            embedding = [float(len(self.encode_calls)), 0.2, 0.3]
            tokens = (
                inputs["prompt_token_ids"]
                if isinstance(inputs, dict)
                else [101, 102, 103]
            )
            # v0.31 exposes pooling data, converted by EmbeddingRequestOutput.
            yield SimpleNamespace(
                request_id=request_id,
                prompt_token_ids=list(tokens),
                outputs=SimpleNamespace(
                    data=SimpleNamespace(tolist=lambda: list(embedding))
                ),
                finished=True,
            )

    class EmbeddingRequestOutput:
        @staticmethod
        def from_base(output):
            return SimpleNamespace(
                request_id=output.request_id,
                prompt_token_ids=output.prompt_token_ids,
                outputs=SimpleNamespace(embedding=output.outputs.data.tolist()),
                finished=output.finished,
            )

    vllm = ModuleType("vllm")
    for name, value in {
        "__version__": "0.31.0",
        "AsyncEngineArgs": args,
        "AsyncLLMEngine": Engine,
        "RequestOutput": SimpleNamespace,
        "EmbeddingRequestOutput": EmbeddingRequestOutput,
        "SamplingParams": SimpleNamespace,
        "PoolingParams": SimpleNamespace,
    }.items():
        setattr(vllm, name, value)
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    # A private module avoids leaking fake vLLM classes into other test suites.
    spec = importlib.util.spec_from_file_location(
        "backend_contract_probe", ROOT / "sllm/backends/vllm_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_engines = engines
    return module


@pytest.fixture
def model_name():
    return "test-model"


@pytest.fixture
def backend_config():
    return {
        "pretrained_model_name_or_path": "test-model",
        "trace_debug": False,
        "load_format": "sharded_state",
        "torch_dtype": None,
    }


@pytest.fixture
def vllm_backend(model_name, backend_config, backend_module):
    return backend_module.VllmBackend(model_name, backend_config)


def request(request_id="test-request-id"):
    return {
        "model": "test-model",
        "prompt": "user: Hello",
        "request_id": request_id,
    }


def test_init(vllm_backend, backend_config, backend_module):
    assert vllm_backend.backend_config == backend_config
    assert vllm_backend.status.name == "UNINITIALIZED"
    assert vllm_backend.engine is None
    assert backend_module.test_engines == []


@pytest.mark.asyncio
async def test_init_backend(vllm_backend, backend_module):
    await vllm_backend.init_backend()
    engine = vllm_backend.engine
    assert vllm_backend.status.name == "RUNNING"
    assert engine.args is vllm_backend.engine_args
    assert engine.args.model == "test-model"
    assert engine.args.load_format == "sharded_state"
    await vllm_backend.init_backend()
    assert backend_module.test_engines == [engine]
    await vllm_backend.shutdown()
    engine.shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_generate_without_init(vllm_backend, backend_module):
    assert await vllm_backend.generate(request()) == {
        "error": "Engine is not running"
    }
    assert backend_module.test_engines == []


@pytest.mark.asyncio
async def test_generate(vllm_backend):
    await vllm_backend.init_backend()
    data = {**request(), "temperature": 0.0, "max_tokens": 3}
    original = deepcopy(data)
    response = await vllm_backend.generate(data)
    assert data == original
    assert response["model"] == "test-model"
    assert response["id"] == "test-request-id"
    assert response["object"] == "chat.completion"
    assert isinstance(response["created"], int)
    assert response["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "test output result"},
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    assert response["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "total_tokens": 6,
    }
    assert vllm_backend.engine.calls == [
        (
            "user: Hello",
            {"temperature": 0.0, "max_tokens": 3},
            "test-request-id",
        )
    ]
    assert await vllm_backend.get_current_tokens() == []
    await vllm_backend.shutdown()


@pytest.mark.asyncio
async def test_shutdown(model_name, backend_config, backend_module):
    backend = backend_module.VllmBackend(
        model_name, {**backend_config, "trace_debug": True}
    )
    await backend.init_backend()
    engine = backend.engine
    await backend.generate(request())
    await backend.shutdown()
    await backend.shutdown()
    assert backend.status.name == "DELETING"
    assert backend.engine is None
    engine.shutdown.assert_called_once_with()
    engine.abort.assert_not_awaited()
    assert await backend.get_debug_tokens() == {
        "test-request-id": [1, 2, 3, 4, 5, 6]
    }


@pytest.mark.asyncio
async def test_shutdown_aborts_active_request(vllm_backend):
    await vllm_backend.init_backend()
    engine = vllm_backend.engine
    engine.release.clear()
    task = asyncio.create_task(vllm_backend.generate(request()))
    try:
        await asyncio.wait_for(engine.started.wait(), 1)
        await asyncio.wait_for(vllm_backend.shutdown(), 1)
    finally:
        engine.release.set()
        await asyncio.wait_for(vllm_backend.shutdown(), 1)
        await asyncio.gather(task, return_exceptions=True)
    engine.abort.assert_awaited_once_with("test-request-id")
    engine.shutdown.assert_called_once_with()
    assert task.done()
    assert vllm_backend.status.name == "DELETING"
    assert await vllm_backend.request_trace.request_count() == 0


@pytest.mark.asyncio
async def test_stop(vllm_backend):
    await vllm_backend.init_backend()
    engine = vllm_backend.engine
    engine.finish.clear()
    task = asyncio.create_task(vllm_backend.generate(request()))
    stop = None
    try:
        await asyncio.wait_for(engine.first_output.wait(), 1)
        stop = asyncio.create_task(vllm_backend.stop())
        await asyncio.sleep(0)
        assert vllm_backend.status.name == "STOPPING"
        assert not stop.done()
        assert not task.done()
        engine.shutdown.assert_not_called()
        assert await vllm_backend.generate(request("late")) == {
            "error": "Engine is not running"
        }
        engine.abort.assert_not_awaited()
    finally:
        engine.finish.set()
        response = await asyncio.wait_for(task, 1)
        if stop is not None:
            await asyncio.wait_for(stop, 1)
        await vllm_backend.shutdown()
    assert "error" not in response
    engine.abort.assert_not_awaited()
    engine.shutdown.assert_called_once_with()
    assert vllm_backend.status.name == "DELETING"


@pytest.mark.asyncio
async def test_get_current_tokens(vllm_backend):
    await vllm_backend.init_backend()
    engine = vllm_backend.engine
    engine.release.clear()
    engine.finish.clear()
    task = asyncio.create_task(
        vllm_backend.generate({**request(), "input_tokens": [11, 12]})
    )
    try:
        await asyncio.wait_for(engine.started.wait(), 1)
        assert await vllm_backend.get_current_tokens() == [[11, 12]]
        engine.release.set()
        await asyncio.wait_for(engine.first_output.wait(), 1)
        assert await vllm_backend.get_current_tokens() == [[11, 12, 4]]
        tokens = await vllm_backend.get_current_tokens()
        tokens[0].append(999)
        assert await vllm_backend.get_current_tokens() == [[11, 12, 4]]
    finally:
        engine.release.set()
        engine.finish.set()
        await asyncio.wait_for(task, 1)
        await vllm_backend.shutdown()
    assert await vllm_backend.get_current_tokens() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_debug", [False, True])
async def test_get_debug_tokens(
    model_name, backend_config, backend_module, trace_debug
):
    backend = backend_module.VllmBackend(
        model_name, {**backend_config, "trace_debug": trace_debug}
    )
    await backend.init_backend()
    for i in range(3):
        await backend.generate(request(f"test-request-id{i}"))
    assert await backend.get_current_tokens() == []
    assert await backend.get_debug_tokens() == (
        {f"test-request-id{i}": [1, 2, 3, 4, 5, 6] for i in range(3)}
        if trace_debug
        else {}
    )
    await backend.shutdown()


@pytest.mark.asyncio
async def test_resume_kv_cache(vllm_backend):
    await vllm_backend.init_backend()
    engine = vllm_backend.engine
    snapshots = [[1, 2, 3], [4, 5, 6]]
    original = deepcopy(snapshots)
    assert await vllm_backend.resume_kv_cache(snapshots) is None
    assert snapshots == original
    assert [call[0] for call in engine.calls] == [
        {"prompt_token_ids": tokens} for tokens in snapshots
    ]
    assert [call[1] for call in engine.calls] == [{"max_tokens": 1}] * 2
    assert len({call[2] for call in engine.calls}) == 2
    assert await vllm_backend.get_current_tokens() == []
    await vllm_backend.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_inputs",
    [
        ("hello", ["hello"]),
        (["hello", "world"], ["hello", "world"]),
        ([1, 2], [{"prompt_token_ids": [1, 2]}]),
        (
            [[1, 2], [3, 4]],
            [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3, 4]}],
        ),
    ],
)
async def test_encode(
    backend_config, backend_module, model_name, query, expected_inputs
):
    backend = backend_module.VllmBackend(
        model_name,
        {
            **backend_config,
            "task": "embed",
            "enforce_eager": True,
            "enable_prefix_caching": False,
        },
    )
    await backend.init_backend()
    engine = backend.engine
    assert engine.args.runner == "pooling"
    assert engine.args.convert == "embed"
    data = {"model": model_name, "input": query}
    original = deepcopy(data)
    result = await backend.encode(data)
    assert data == original
    assert result["model"] == model_name
    assert result["object"] == "list"
    assert result["data"] == [
        {
            "object": "embedding",
            "index": i,
            "embedding": [float(i + 1), 0.2, 0.3],
        }
        for i in range(len(expected_inputs))
    ]
    assert [call[0] for call in engine.encode_calls] == expected_inputs
    assert len({call[2] for call in engine.encode_calls}) == len(
        expected_inputs
    )
    token_count = sum(
        len(item["prompt_token_ids"]) if isinstance(item, dict) else 3
        for item in expected_inputs
    )
    assert result["usage"] == {
        "query_tokens": token_count,
        "total_tokens": token_count,
    }
    assert await backend.get_current_tokens() == []
    await backend.shutdown()
