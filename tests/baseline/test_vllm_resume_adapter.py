"""CPU resume contracts, executing selected classes from the pinned source.

Set SLLM_VLLM_031_SOURCE to a read-only v0.31.0 source checkout for source
execution checks. No vLLM package, model weights, torch or CUDA is imported.
"""

import ast
import asyncio
import copy
import enum
import importlib
import io
import itertools
import json
import math
import os
import sys
import types
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import fetch_vllm_resume_source as fetcher
import pytest
from fetch_vllm_resume_source import MANIFEST, verify_source

from sllm.backends import vllm_resume_adapter as adapter


def _source_module(monkeypatch, root, path, module_name, names, namespace):
    tree = ast.parse((root / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    module = types.ModuleType(module_name)
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, module_name, module)
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + nodes
    exec(
        compile(ast.fix_missing_locations(tree), str(root / path), "exec"),
        module.__dict__,
    )
    return module


def _source_method(root, path, class_name, method_name, namespace):
    tree = ast.parse((root / path).read_text())
    owner = next(
        node for node in tree.body if getattr(node, "name", None) == class_name
    )
    method = next(
        node
        for node in owner.body
        if getattr(node, "name", None) == method_name
    )
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        method,
    ]
    exec(
        compile(ast.fix_missing_locations(tree), str(root / path), "exec"),
        namespace,
    )
    return namespace[method_name]


@pytest.fixture
def pinned(monkeypatch):
    root = os.environ.get("SLLM_VLLM_031_SOURCE")
    if not root:
        pytest.skip("Set SLLM_VLLM_031_SOURCE for the pinned-source CPU checks")
    root = Path(root)
    verify_source(root, json.loads(MANIFEST.read_text()))
    msgspec = pytest.importorskip("msgspec")
    logger = types.SimpleNamespace(
        warning=lambda *a: None, warning_once=lambda *a: None
    )

    class ValidationError(ValueError):
        def __init__(self, message, **kwargs):
            super().__init__(message)

    namespace = dict(
        Any=Any,
        msgspec=msgspec,
        dataclass=dataclass,
        field=field,
        Enum=enum.Enum,
        IntEnum=enum.IntEnum,
        cached_property=cached_property,
        PydanticMsgspecMixin=type("PydanticMsgspecMixin", (), {}),
        VLLMValidationError=ValidationError,
        logger=logger,
        math=math,
        copy=copy,
        _MAX_TEMP=1e-2,
        _SAMPLING_EPS=1e-5,
        MAX_LOGPROB_TOKEN_IDS=128,
        ThinkingTokenBudget=int | None,
        envs=types.SimpleNamespace(
            VLLM_MAX_N_SEQUENCES=128, VLLM_BATCH_INVARIANT=False
        ),
    )
    sampling = _source_module(
        monkeypatch,
        root,
        "vllm/sampling_params.py",
        "vllm.sampling_params",
        [
            "SamplingType",
            "RequestOutputKind",
            "StructuredOutputsParams",
            "RepetitionDetectionParams",
            "SamplingParams",
            "_verify_num_sequences",
            "validate_thinking_token_budget",
        ],
        namespace,
    )
    logprobs = types.ModuleType("vllm.logprobs")
    monkeypatch.setitem(sys.modules, "vllm.logprobs", logprobs)
    # This source module uses only Python standard-library imports.
    exec(
        compile(
            (root / "vllm/logprobs.py").read_text(), "vllm/logprobs.py", "exec"
        ),
        logprobs.__dict__,
    )
    detokenizer = _source_module(
        monkeypatch,
        root,
        "vllm/v1/engine/detokenizer.py",
        "_pinned_detokenizer",
        [
            "IncrementalDetokenizer",
            "BaseIncrementalDetokenizer",
            "check_stop_strings",
        ],
        dict(ABC=ABC, abstractmethod=abstractmethod, sys=sys),
    )
    processors = _source_module(
        monkeypatch,
        root,
        "vllm/v1/engine/logprobs.py",
        "_pinned_logprobs_processor",
        ["LogprobsProcessor"],
        dict(
            NONES=itertools.repeat(None),
            **{
                key: value
                for key, value in vars(logprobs).items()
                if not key.startswith("__")
            },
        ),
    )
    output = _source_module(
        monkeypatch,
        root,
        "vllm/outputs.py",
        "vllm.outputs",
        ["CompletionOutput", "RequestOutput"],
        dict(dataclass=dataclass, logger=logger),
    )
    frontend = _source_module(
        monkeypatch,
        root,
        "vllm/v1/engine/output_processor.py",
        "_pinned_frontend",
        ["RequestState"],
        dict(
            RequestOutputKind=sampling.RequestOutputKind,
            CompletionOutput=output.CompletionOutput,
            RequestOutput=output.RequestOutput,
        ),
    )
    return types.SimpleNamespace(
        root=root,
        sampling=sampling,
        logprobs=logprobs,
        detokenizer=detokenizer,
        processors=processors,
        output=output,
        frontend=frontend,
    )


def _metadata(params):
    return dict(
        request_id="public",
        prompt="P",
        prompt_token_ids=[80],
        sampling_params=adapter._encode_sampling(params),
    )


def _state(pinned, params):
    request = types.SimpleNamespace(sampling_params=params)

    class Decoder(pinned.detokenizer.BaseIncrementalDetokenizer):
        def decode_next(self, token):
            return chr(token)

    state = pinned.frontend.RequestState.__new__(pinned.frontend.RequestState)
    state.detokenizer = Decoder(request)
    state.logprobs_processor = (
        pinned.processors.LogprobsProcessor.from_new_request(None, request)
    )
    state.output_kind = pinned.sampling.RequestOutputKind.CUMULATIVE
    state.request_index = 0
    state.sampling_mask_chunks = []
    state.routed_experts_chunks = []
    state.spec_decode_metrics = None
    return state


def test_effective_sampling_roundtrip_retains_derived_fields(pinned):
    params = pinned.sampling.SamplingParams(
        n=3,
        max_tokens=50,
        min_tokens=12,
        seed=9,
        presence_penalty=0.5,
        frequency_penalty=0.4,
        repetition_penalty=1.2,
        stop=["END"],
        stop_token_ids=[44],
        bad_words=["bad"],
    )
    params._eos_token_id = 2
    params._all_stop_token_ids.add(2)
    params._bad_words_token_ids = [[3, 4]]
    restored = adapter._decode_sampling(adapter._encode_sampling(params))
    assert restored.n == 3 and restored.seed == 9
    assert restored.max_tokens == 50 and restored.min_tokens == 12
    assert restored._eos_token_id == 2
    assert restored._all_stop_token_ids == {2, 44}
    assert restored._bad_words_token_ids == [[3, 4]]
    assert restored.presence_penalty == 0.5
    assert restored.frequency_penalty == 0.4
    assert restored.repetition_penalty == 1.2


def test_structured_output_sampling_retains_selected_backend(pinned):
    structured = pinned.sampling.StructuredOutputsParams(regex="[a-z]+")
    structured._backend = "xgrammar"
    structured._backend_was_auto = True
    params = pinned.sampling.SamplingParams(structured_outputs=structured)
    restored = adapter._decode_sampling(adapter._encode_sampling(params))
    assert restored.structured_outputs.regex == "[a-z]+"
    assert restored.structured_outputs._backend == "xgrammar"
    assert restored.structured_outputs._backend_was_auto is True


def test_empty_choice_does_not_duplicate_prompt_scores(pinned):
    params = pinned.sampling.SamplingParams(prompt_logprobs=1)
    snapshot = adapter.snapshot_request(
        _metadata(params), None, model="m", identity={}
    )
    snapshot["prompt_logprobs"] = [
        None,
        {1: dict(logprob=-0.5, rank=1, decoded_token="p")},
    ]
    state = _state(pinned, params)
    adapter._restore_frontend(state, snapshot, snapshot["choices"][0])
    assert state.logprobs_processor.prompt_logprobs == [None]


@pytest.mark.parametrize("tokens", [[], [65], [65, 66, 69]])
@pytest.mark.parametrize("include_stop", [False, True])
def test_abort_flush_is_normalized_using_real_completion_output(
    pinned, tokens, include_stop
):
    params = pinned.sampling.SamplingParams(
        stop=["END"], include_stop_str_in_output=include_stop
    )
    state = _state(pinned, params)
    state.detokenizer.update(tokens, False)
    completion = state._new_completion_output([], "abort", None)
    output = pinned.output.RequestOutput(
        "public", "P", [80], None, [completion], True
    )
    normalized = adapter.normalize_abort_output(output, _metadata(params))
    assert normalized.outputs[0].text == state.detokenizer.get_next_output_text(
        False, False
    )
    assert normalized.outputs[0].finish_reason is None
    assert not normalized.finished
    assert output.outputs[0].finish_reason == "abort"
    snapshot = adapter.snapshot_request(
        _metadata(params), normalized, model="m", identity={}
    )
    resumed = _state(pinned, params)
    adapter._restore_frontend(resumed, snapshot, snapshot["choices"][0])
    assert resumed.detokenizer.output_text == state.detokenizer.output_text


def test_abort_normalization_preserves_naturally_stopped_choice(pinned):
    params = pinned.sampling.SamplingParams(stop=["END"], stop_token_ids=[0])
    state = _state(pinned, params)
    state.detokenizer.update([65, 0], True)
    completion = state._new_completion_output([], "stop", 0)
    output = pinned.output.RequestOutput(
        "public", "P", [80], None, [completion], True
    )
    normalized = adapter.normalize_abort_output(output, _metadata(params))
    assert normalized.finished
    assert normalized.outputs[0].text == "A"
    assert normalized.outputs[0].token_ids == [65, 0]
    assert normalized.outputs[0].finish_reason == "stop"
    assert normalized.outputs[0].stop_reason == 0


def test_delta_abort_only_removes_the_newly_flushed_buffer(pinned):
    params = pinned.sampling.SamplingParams(
        stop=["END"], output_kind=pinned.sampling.RequestOutputKind.DELTA
    )
    state = _state(pinned, params)
    state.output_kind = params.output_kind
    state.detokenizer.update([65, 66, 69], False)
    initial = state._new_completion_output([65, 66, 69], None, None)
    assert initial.text == "A"
    completion = state._new_completion_output([], "abort", None)
    assert completion.text == "BE"
    output = pinned.output.RequestOutput(
        "public", "P", [80], None, [completion], True
    )
    normalized = adapter.normalize_abort_output(output, _metadata(params))
    assert normalized.outputs[0].text == ""
    assert normalized.outputs[0].token_ids == []


def test_detokenize_false_restores_real_token_only_detokenizer(pinned):
    params = pinned.sampling.SamplingParams(detokenize=False)
    snapshot = adapter.snapshot_request(
        _metadata(params), None, model="m", identity={}
    )
    snapshot["choices"][0]["token_ids"] = [1, 2, 3]
    state = _state(pinned, params)
    state.detokenizer = pinned.detokenizer.IncrementalDetokenizer()
    adapter._restore_frontend(state, snapshot, snapshot["choices"][0])
    assert state.detokenizer.output_token_ids == [1, 2, 3]
    assert state.detokenizer.get_next_output_text(False, False) == ""


@pytest.mark.parametrize("field", ["mm_features", "prompt_embeds"])
def test_nontext_requests_cannot_silently_drop_input_state(pinned, field):
    request = types.SimpleNamespace(
        request_id="id",
        external_req_id=None,
        prompt_token_ids=[1],
        sampling_params=pinned.sampling.SamplingParams(),
    )
    setattr(request, field, [object()])
    metadata = adapter._metadata(request)
    assert field in metadata["capture_error"]
    with pytest.raises(ValueError, match=field):
        adapter.snapshot_request(metadata, None, model="m", identity={})


@pytest.mark.parametrize("flat", [False, True])
def test_real_frontend_restores_stop_buffer_and_logprobs_without_repeating(
    pinned, flat
):
    params = pinned.sampling.SamplingParams(
        max_tokens=8,
        min_tokens=1,
        stop=["END"],
        logprobs=1,
        prompt_logprobs=1,
        flat_logprobs=flat,
    )
    before = _state(pinned, params)
    before.detokenizer.update([65, 66, 69], False)  # ABE, with BE withheld.
    rows = [
        {token: pinned.logprobs.Logprob(-0.5, 1, chr(token))}
        for token in [65, 66, 69]
    ]
    before.logprobs_processor.logprobs.extend(rows)
    before.logprobs_processor.cumulative_logprob = -1.5
    completion = before._new_completion_output([], None, None)
    request_output = pinned.output.RequestOutput(
        "public", "P", [80], [None], [completion], False
    )
    snapshot = adapter.snapshot_request(
        _metadata(params), request_output, model="m", identity={"revision": "a"}
    )
    after = _state(pinned, params)
    adapter._restore_frontend(after, snapshot, snapshot["choices"][0])
    assert after.detokenizer.output_text == "ABE"
    assert after.detokenizer.get_next_output_text(False, False) == "A"
    assert after.logprobs_processor.cumulative_logprob == -1.5
    assert len(after.logprobs_processor.logprobs) == 3
    assert len(after.logprobs_processor.prompt_logprobs) == 1
    assert (
        isinstance(
            after.logprobs_processor.logprobs, pinned.logprobs.FlatLogprobs
        )
        == flat
    )
    assert after.detokenizer.update([78, 68], False) == "END"
    assert after.detokenizer.output_token_ids == [65, 66, 69, 78, 68]
    assert after.detokenizer.get_next_output_text(True, False) == "AB"


def test_zero_output_snapshot_keeps_all_choices_and_exact_budget(pinned):
    params = pinned.sampling.SamplingParams(
        n=3, seed=77, max_tokens=30, min_tokens=12
    )
    snapshot = adapter.snapshot_request(
        _metadata(params), None, model="m", identity={}
    )
    assert len(snapshot["choices"]) == 3
    assert adapter.child_request_ids(snapshot, "epoch") == [
        "epoch:choice:0",
        "epoch:choice:1",
        "epoch:choice:2",
    ]
    snapshot["choices"][1].update(
        finish_reason="stop", stop_reason="END", text="done", token_ids=[1]
    )
    snapshot["choices"][2]["token_ids"] = [2] * 30
    assert adapter.child_request_ids(snapshot, "epoch") == ["epoch:choice:0"]
    result = adapter._request_output(snapshot)
    assert result.outputs[1].finish_reason == "stop"
    assert result.outputs[2].finish_reason == "length"
    assert snapshot["sampling_params"]["max_tokens"] == 30


@pytest.fixture
def scheduler_types(pinned, monkeypatch):
    stats = _source_module(
        monkeypatch,
        pinned.root,
        "vllm/v1/metrics/stats.py",
        "_pinned_prefill_stats",
        ["PrefillStats"],
        dict(dataclass=dataclass),
    )
    request_module = _source_module(
        monkeypatch,
        pinned.root,
        "vllm/v1/request.py",
        "_pinned_request",
        ["Request", "RequestStatus"],
        dict(
            enum=enum,
            time=types.SimpleNamespace(time=lambda: 0),
            deque=deque,
            StructuredOutputRequest=types.SimpleNamespace(
                from_sampling_params=lambda params: None
            ),
            length_from_prompt_token_ids_or_embeds=lambda ids, embeds: len(ids),
            ConstantList=lambda value: value,
            PrefillStats=stats.PrefillStats,
        ),
    )

    class Scheduler:
        def __init__(self, vllm_config):
            self.use_v2_model_runner = vllm_config.use_v2_model_runner
            self.requests = {}

        def add_request(self, request):
            self.requests[request.request_id] = request

        def _handle_blocked_waiting_request(self, request):
            return True

    class AsyncScheduler(Scheduler):
        def __init__(self, vllm_config):
            super().__init__(vllm_config)
            self.async_initialized = True

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.sched.scheduler",
        types.SimpleNamespace(Scheduler=Scheduler),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.sched.async_scheduler",
        types.SimpleNamespace(AsyncScheduler=AsyncScheduler),
    )
    monkeypatch.delitem(
        sys.modules, "sllm.backends.vllm_resume_scheduler", raising=False
    )
    module = importlib.import_module("sllm.backends.vllm_resume_scheduler")
    return request_module, module, Scheduler


def test_scheduler_uses_real_request_append_and_v2_prefill_boundary(
    pinned, scheduler_types
):
    request_module, module, Scheduler = scheduler_types
    for async_scheduling in [False, True]:
        config = types.SimpleNamespace(
            use_v2_model_runner=True,
            scheduler_config=types.SimpleNamespace(
                async_scheduling=async_scheduling
            ),
        )
        scheduler = module.ResumeScheduler(config)
        assert hasattr(scheduler, "async_initialized") == async_scheduling
        params = pinned.sampling.SamplingParams(
            max_tokens=10,
            min_tokens=5,
            extra_args={
                adapter.RESUME_KEY: {
                    "schema_version": 1,
                    "output_token_ids": [3, 4],
                }
            },
        )
        hashes = []

        def hasher(request):
            hashes.append(tuple(request.all_token_ids))
            return [tuple(request.all_token_ids)]

        request = request_module.Request(
            "id", [1, 2], params, None, block_hasher=hasher
        )
        scheduler.add_request(request)
        assert request.prompt_token_ids == [1, 2]
        assert request.output_token_ids == [3, 4]
        assert request.all_token_ids == [1, 2, 3, 4]
        assert request.num_computed_tokens == 0
        assert request.max_tokens == 10 and params.min_tokens == 5
        assert hashes == [(1, 2), (1, 2, 3, 4)]
        # Exercise actual upstream readiness and grammar acceptance methods.
        Scheduler._handle_blocked_waiting_request = _source_method(
            pinned.root,
            "vllm/v1/core/sched/scheduler.py",
            "Scheduler",
            "_handle_blocked_waiting_request",
            dict(RequestStatus=request_module.RequestStatus),
        )
        accepted = []
        manager = types.SimpleNamespace(
            _get_constraint_start=lambda *args, **kwargs: 0
        )
        manager.accept_tokens = types.MethodType(
            _source_method(
                pinned.root,
                "vllm/v1/structured_output/__init__.py",
                "StructuredOutputManager",
                "accept_tokens",
                dict(TYPE_CHECKING=False),
            ),
            manager,
        )
        scheduler.structured_output_manager = manager
        request.structured_output_request = types.SimpleNamespace(
            grammar=None, reasoning_ended=None
        )
        request.status = (
            request_module.RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        )
        assert not scheduler._handle_blocked_waiting_request(request)
        assert accepted == []
        request.structured_output_request.grammar = types.SimpleNamespace(
            accept_tokens=lambda request_id, tokens: accepted.append(
                (request_id, tokens)
            )
            is None
        )
        assert scheduler._handle_blocked_waiting_request(request)
        assert scheduler._handle_blocked_waiting_request(request)
        assert accepted == [("id", [3, 4])]
        assert request.structured_output_request.reasoning_ended
    # Upstream keeps prefill history separate from the original prompt.
    tree = ast.parse(
        (pinned.root / "vllm/v1/core/sched/scheduler.py").read_text()
    )
    source = ast.unparse(tree)
    assert "req._all_token_ids" in source
    data = (pinned.root / "vllm/v1/core/sched/output.py").read_text()
    assert "prefill_token_ids=prefill_token_ids" in data
    assert "return len(self.prompt_token_ids)" in data


@pytest.mark.asyncio
async def test_registration_waits_for_core_submit_and_all_children(pinned):
    registrations = []
    gate = asyncio.Event()

    class Processor:
        def add_request(
            self, request, prompt, parent_req=None, request_index=0, queue=None
        ):
            registrations.append(request.request_id)

    class Core:
        async def add_request_async(self, request):
            await gate.wait()

    first, second = Processor(), Processor()
    original = second.add_request.__func__
    engine = types.SimpleNamespace(output_processor=first, engine_core=Core())
    adapter.install(engine)
    installed = first.add_request
    adapter.install(engine)
    assert first.add_request is installed
    assert second.add_request.__func__ is original
    params = pinned.sampling.SamplingParams(n=2)
    parent = types.SimpleNamespace(
        external_req_id="public", sampling_params=params, n=2
    )
    waiter = asyncio.create_task(adapter.wait_registered(engine, "public"))
    children = [
        types.SimpleNamespace(
            request_id=f"{index}_random",
            external_req_id="public",
            sampling_params=params,
            prompt_token_ids=[1],
        )
        for index in range(2)
    ]
    for index, request in enumerate(children):
        first.add_request(request, "P", parent, index)
    submitted = asyncio.create_task(
        engine.engine_core.add_request_async(children[0])
    )
    await asyncio.sleep(0)
    assert not waiter.done()
    gate.set()
    await submitted
    assert not waiter.done()
    await engine.engine_core.add_request_async(children[1])
    await waiter
    assert registrations == ["0_random", "1_random"]
    assert (
        adapter.request_metadata(engine, "public")["sampling_params"]["n"] == 2
    )


@pytest.mark.asyncio
async def test_prepare_uses_normalized_input_processor_result(pinned):
    params = pinned.sampling.SamplingParams(max_tokens=None)
    calls = []

    class InputProcessor:
        async def process_inputs_async(
            self, request_id, inputs, original, *, supported_tasks
        ):
            calls.append((request_id, inputs, original, supported_tasks))
            effective = copy.deepcopy(original)
            effective.max_tokens = 96
            return types.SimpleNamespace(
                request_id=request_id,
                external_req_id=None,
                prompt_token_ids=[10, 11, 12],
                sampling_params=effective,
            )

    async def tasks():
        return ("generate",)

    engine = types.SimpleNamespace(
        input_processor=InputProcessor(), get_supported_tasks=tasks
    )
    request, metadata = await adapter.prepare_request(
        engine, "raw", params, "id"
    )
    assert len(calls) == 1
    assert (
        metadata["prompt_token_ids"] == request.prompt_token_ids == [10, 11, 12]
    )
    assert metadata["sampling_params"]["max_tokens"] == 96
    assert params.max_tokens is None


@pytest.mark.asyncio
async def test_unencodable_metadata_keeps_normal_admission(pinned):
    params = pinned.sampling.SamplingParams()
    params.extra_args = {"custom_plugin_state": object()}
    request = types.SimpleNamespace(
        request_id="internal",
        external_req_id="public",
        prompt_token_ids=[1],
        sampling_params=params,
    )
    metadata = adapter._metadata(request)
    assert metadata["prompt_token_ids"] == [1]
    assert metadata["sampling_params"]["output_kind"] == 0
    assert metadata["sampling_params"]["max_tokens"] == 16
    assert metadata["capture_error"]
    calls = []

    class Processor:
        def add_request(self, *args):
            calls.append("frontend")

    class Core:
        async def add_request_async(self, request):
            calls.append("core")

    engine = types.SimpleNamespace(
        output_processor=Processor(), engine_core=Core()
    )
    adapter.install(engine)
    engine.output_processor.add_request(request, "P")
    await engine.engine_core.add_request_async(request)
    await adapter.wait_registered(engine, "public")
    assert calls == ["frontend", "core"]
    with pytest.raises(ValueError, match="cannot be migrated"):
        adapter.snapshot_request(metadata, None, model="m", identity={})


def test_capability_rejects_v1_before_freezing():
    engine = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(use_v2_model_runner=False)
    )
    with pytest.raises(ValueError, match="V2"):
        adapter.validate_engine(engine)


def test_source_fetcher_checks_downloads_and_existing_cache(
    pinned, monkeypatch, tmp_path
):
    calls = []

    def download(url, timeout):
        name = url.split("/v0.31.0/", 1)[1]
        calls.append(name)
        return io.BytesIO((pinned.root / name).read_bytes())

    monkeypatch.setattr(fetcher, "urlopen", download)
    fetcher.fetch_source(tmp_path)
    count = len(json.loads(MANIFEST.read_text())["files"])
    assert len(calls) == count
    fetcher.fetch_source(tmp_path)
    assert len(calls) == count
    (tmp_path / calls[0]).write_text("wrong version")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_source(tmp_path, json.loads(MANIFEST.read_text()))
    monkeypatch.setattr(
        fetcher, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong version")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        fetcher.fetch_source(tmp_path)


@pytest.mark.asyncio
async def test_resume_aggregates_choices_keeps_seeds_and_original_budgets(
    pinned,
):
    params = pinned.sampling.SamplingParams(
        n=3, seed=10, max_tokens=4, min_tokens=2
    )
    snapshot = adapter.snapshot_request(
        _metadata(params), None, model="m", identity={}
    )
    snapshot["choices"][0].update(token_ids=[65], text="A")
    snapshot["choices"][1].update(
        token_ids=[68], text="D", finish_reason="stop", stop_reason=0
    )
    snapshot["choices"][2].update(token_ids=[66, 67], text="BC")
    submitted = []
    aborted = []

    class Processor:
        def __init__(self):
            self.request_states = {}

        def add_request(
            self, request, prompt, parent_req=None, request_index=0, queue=None
        ):
            self.request_states[request.request_id] = _state(
                pinned, request.sampling_params
            )

    class Core:
        async def add_request_async(self, request):
            submitted.append(
                (
                    request.external_req_id,
                    copy.deepcopy(request.sampling_params),
                )
            )

    class Engine:
        def __init__(self):
            self.vllm_config = types.SimpleNamespace(
                use_v2_model_runner=True,
                scheduler_config=types.SimpleNamespace(
                    scheduler_cls=adapter.SCHEDULER_CLS
                ),
            )
            self.model_config = types.SimpleNamespace(max_model_len=100)
            self.input_processor = self
            self.output_processor = Processor()
            self.engine_core = Core()

        async def get_supported_tasks(self):
            return ("generate",)

        async def process_inputs_async(
            self, request_id, inputs, params, *, supported_tasks
        ):
            return types.SimpleNamespace(
                request_id=request_id,
                external_req_id=None,
                prompt_token_ids=list(inputs["prompt_token_ids"]),
                sampling_params=copy.deepcopy(params),
            )

        async def generate(self, request, params, request_id):
            request.external_req_id = request_id
            request.request_id += "-random"
            self.output_processor.add_request(request, None)
            await self.engine_core.add_request_async(request)
            state = self.output_processor.request_states[request.request_id]
            remaining = params.max_tokens - len(
                state.detokenizer.output_token_ids
            )
            state.detokenizer.update([90] * remaining, False)
            completion = state._new_completion_output([], "length", None)
            yield pinned.output.RequestOutput(
                request_id,
                None,
                request.prompt_token_ids,
                None,
                [completion],
                True,
            )

        async def abort(self, request_id):
            aborted.append(request_id)

    engine = Engine()
    adapter.install(engine)
    outputs = [
        output async for output in adapter.resume(engine, snapshot, "epoch")
    ]
    final = outputs[-1]
    assert final.finished
    assert final.request_id == "public"
    assert [output.index for output in final.outputs] == [0, 1, 2]
    assert [output.text for output in final.outputs] == ["AZZZ", "D", "BCZZ"]
    assert final.outputs[1].finish_reason == "stop"
    assert final.outputs[1].stop_reason == 0
    assert sorted(
        (
            request_id,
            params.seed,
            params.n,
            params.max_tokens,
            params.min_tokens,
        )
        for request_id, params in submitted
    ) == [("epoch:choice:0", 10, 1, 4, 2), ("epoch:choice:2", 12, 1, 4, 2)]
    assert sorted(aborted) == ["epoch:choice:0", "epoch:choice:2"]
    assert not engine._sllm_resume_adapter["pending"]
    assert snapshot["sampling_params"]["n"] == 3


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "history, cache_hit",
    [
        (None, 0),
        (None, 1),
        ([], 0),
        ([], 1),
        ([3, 4], 0),
        ([3, 4], 1),
        ([3, 4], 3),
    ],
)
def test_resume_prefill_accounting_with_output_history_cache_hit(
    pinned, scheduler_types, async_scheduling, history, cache_hit
):
    request_module, module, _ = scheduler_types
    config = types.SimpleNamespace(
        use_v2_model_runner=True,
        scheduler_config=types.SimpleNamespace(
            async_scheduling=async_scheduling
        ),
    )
    params = pinned.sampling.SamplingParams(max_tokens=10, min_tokens=5)
    if history is not None:
        params.extra_args = {
            adapter.RESUME_KEY: {
                "schema_version": 1,
                "output_token_ids": history,
            }
        }
    request = request_module.Request("id", [1, 2], params, None)
    first_prefill = request.prefill_stats
    scheduler = module.ResumeScheduler(config)
    scheduler.add_request(request)

    # Execute the exact upstream first-prefill branch which failed on GPU.
    # The supplied hit is a target-local KV lookup result, not source KV.
    tree = ast.parse(
        (pinned.root / "vllm/v1/core/sched/scheduler.py").read_text()
    )
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    schedule = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == "schedule"
    )
    branches = [
        n
        for n in ast.walk(schedule)
        if isinstance(n, ast.If)
        and ast.unparse(n.test)
        == "request.prefill_stats and request.num_preemptions <= 0"
    ]
    assert len(branches) == 1
    block = ast.Module(body=branches, type_ignores=[])
    namespace = dict(
        request=request,
        num_computed_tokens=cache_hit,
        num_new_local_computed_tokens=cache_hit,
        num_external_computed_tokens=0,
    )
    exec(
        compile(
            ast.fix_missing_locations(block), "pinned-prefill-branch", "exec"
        ),
        namespace,
    )

    assert request.prompt_token_ids == [1, 2]
    assert request.output_token_ids == (history or [])
    assert request.num_computed_tokens == request.num_preemptions == 0
    assert request.max_tokens == 10 and params.min_tokens == 5
    assert namespace["num_computed_tokens"] == cache_hit
    if history:
        assert request.take_prefill_stats() is None
    else:
        assert request.take_prefill_stats() is first_prefill
        assert first_prefill.num_prompt_tokens == 2
        assert first_prefill.num_cached_tokens == cache_hit
        assert first_prefill.num_computed_tokens == 2 - cache_hit
