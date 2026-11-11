"""CPU state-machine contracts for freezing and handing off active requests."""

import asyncio
import copy
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import pytest_asyncio
from test_legacy_contract import legacy as legacy
from test_vllm_migration_router import migration as _router_migration
from test_vllm_migration_router import request as router_request
from test_vllm_migration_router import rpc

router_migration = _router_migration


def output(request_id, tokens, reason=None):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[10, 11],
        outputs=[
            SimpleNamespace(
                index=0,
                token_ids=list(tokens),
                text=" ".join(map(str, tokens)),
                logprobs=None,
                finish_reason=reason,
                stop_reason=None,
                cumulative_logprob=None,
            )
        ],
        metrics=None,
        finished=reason is not None,
    )


class Stream:
    def __init__(self):
        self.started = asyncio.Event()
        self.registration = asyncio.Event()
        self.registration.set()
        self.registered = asyncio.Event()
        self.observed = asyncio.Event()
        self.finish = asyncio.Event()
        self.tail = asyncio.Event()
        self.tail.set()
        self.aborted = False
        self.natural_on_abort = False


class Engine:
    """A controllable async iterator, including output racing with abort."""

    _sllm_resume_adapter = True

    def __init__(self):
        self.streams = {}
        self.calls = []
        self.prewarm_calls = []
        self.prewarm_started = asyncio.Event()
        self.prewarm_release = asyncio.Event()
        self.prewarm_release.set()
        self.resume_calls = []
        self.abort_calls = []
        self.abort_errors = set()
        self.close_calls = 0
        self.supported = True
        self.with_logprobs = False

    def stream(self, key):
        return self.streams.setdefault(key, Stream())

    def get_tokenizer(self):
        return SimpleNamespace(get_vocab=lambda: {"hello": 10, "world": 11})

    def result(self, request_id, tokens, reason=None):
        result = output(request_id, tokens, reason)
        if self.with_logprobs:
            choice = result.outputs[0]
            choice.logprobs = [{token: -0.5} for token in tokens]
            choice.cumulative_logprob = -0.5 * len(tokens)
        return result

    async def iterate(self, key, request_id, previous):
        stream = self.stream(key)
        stream.started.set()
        await stream.registration.wait()
        stream.registered.set()
        yield self.result(request_id, previous)
        stream.observed.set()
        await stream.finish.wait()
        await stream.tail.wait()
        reason = (
            "abort"
            if stream.aborted and not stream.natural_on_abort
            else "stop"
        )
        yield self.result(request_id, previous + [previous[-1] + 1], reason)

    async def generate(self, inputs, params, request_id):
        if params.values.get("max_tokens") == 1:
            self.prewarm_calls.append((inputs, request_id))
            self.prewarm_started.set()
            await self.prewarm_release.wait()
            yield output(request_id, [99], "length")
            return
        self.calls.append(request_id)
        emitted_tokens = emitted_text = 0
        async for item in self.iterate(request_id, request_id, [20]):
            if params.values.get("output_kind") == 1:
                choice = item.outputs[0]
                token_count, text_count = (
                    len(choice.token_ids),
                    len(choice.text),
                )
                choice.token_ids = choice.token_ids[emitted_tokens:]
                choice.text = choice.text[emitted_text:]
                if choice.logprobs is not None:
                    choice.logprobs = choice.logprobs[emitted_tokens:]
                emitted_tokens, emitted_text = token_count, text_count
            yield item

    async def abort(self, request_id):
        self.abort_calls.append(request_id)
        if request_id in self.abort_errors:
            raise RuntimeError("abort failed")
        stream = self.stream(request_id)
        stream.aborted = True
        stream.finish.set()

    def shutdown(self):
        self.close_calls += 1


def validate_engine(engine):
    if not engine.supported:
        raise RuntimeError("unsupported runner")


def validate_snapshot(engine, snapshot):
    validate_engine(engine)
    if snapshot["sampling_params"].get("logits_processors"):
        raise ValueError("unsupported sampling params")


async def prepare_request(engine, inputs, params, request_id):
    return inputs, {
        "request_id": request_id,
        "prompt": inputs if isinstance(inputs, str) else None,
        "prompt_token_ids": list(inputs.get("prompt_token_ids", [10, 11]))
        if isinstance(inputs, dict)
        else [10, 11],
        "sampling_params": {
            "output_text_buffer_length": 0,
            **copy.deepcopy(params.values),
        },
    }


def snapshot_request(metadata, merged_output, *, model, identity):
    choices = []
    if merged_output is not None:
        choices = [
            {
                "index": item.index,
                "token_ids": list(item.token_ids),
                "text": item.text,
                "logprobs": item.logprobs,
                "cumulative_logprob": item.cumulative_logprob,
                "stop_reason": item.stop_reason,
                "finish_reason": None
                if item.finish_reason == "abort"
                else item.finish_reason,
            }
            for item in merged_output.outputs
        ]
    return {
        "schema_version": 1,
        "request_id": metadata["request_id"],
        "prompt": metadata["prompt"],
        "prompt_token_ids": list(metadata["prompt_token_ids"]),
        "sampling_params": copy.deepcopy(metadata["sampling_params"]),
        "choices": choices,
        "model": model,
        "identity": identity,
        "prompt_logprobs": None,
        "prompt_token_id_logprobs": None,
    }


async def resume(engine, snapshot, internal_root):
    engine.resume_calls.append((snapshot["request_id"], internal_root))
    previous = (
        snapshot["choices"][0]["token_ids"] if snapshot["choices"] else [20]
    )
    async for item in engine.iterate(
        internal_root + ":0", snapshot["request_id"], previous
    ):
        yield item


async def wait_registered(engine, request_id):
    await engine.stream(request_id).registered.wait()


async def joined(awaitable):
    return await asyncio.wait_for(awaitable, 2)


async def turns():
    for _ in range(3):
        await asyncio.sleep(0)


@pytest_asyncio.fixture
async def migration(monkeypatch, request):
    try:
        adapter = importlib.import_module("sllm.backends.vllm_resume_adapter")
    except ModuleNotFoundError:
        adapter = ModuleType("sllm.backends.vllm_resume_adapter")
        monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    for name, value in {
        "install": lambda engine: None,
        "validate_engine": validate_engine,
        "prepare_request": prepare_request,
        "snapshot_request": snapshot_request,
        "validate_snapshot": validate_snapshot,
        "child_request_ids": lambda snapshot, root: [root + ":0"],
        "resume": resume,
        "wait_registered": wait_registered,
        "forget_request": lambda engine, request_id: None,
    }.items():
        monkeypatch.setattr(adapter, name, value, raising=False)
    factory = request.getfixturevalue("legacy")
    backends, tasks = [], []

    def create():
        backend = factory()
        backend.engine = Engine()
        backends.append(backend)
        return backend

    def spawn(coroutine):
        task = asyncio.create_task(coroutine)
        tasks.append(task)
        return task

    yield SimpleNamespace(create=create, spawn=spawn)
    for backend in backends:
        if backend.engine is not None:
            backend.engine.prewarm_release.set()
            for stream in backend.engine.streams.values():
                stream.finish.set()
                stream.tail.set()
                stream.registration.set()
    for task in tasks:
        if not task.done():
            task.cancel()
    await joined(asyncio.gather(*tasks, return_exceptions=True))
    for backend in backends:
        await joined(backend.shutdown())


@pytest.fixture
def connected_router(migration, router_migration, monkeypatch):
    """Use both state machines with fake actor transport and engine."""
    import sllm.routers.migration_router as router_module

    source, target = migration.create(), migration.create()

    def actor(backend):
        result = SimpleNamespace(init_backend=rpc(lambda: None))
        for name in (
            "generate",
            "get_current_tokens",
            "resume_kv_cache",
            "shutdown",
            "freeze_migration",
            "prepare_migration",
            "commit_migration",
            "rollback_migration",
            "resume_migration",
            "cancel_migration_request",
            "forget_migration_request",
        ):
            setattr(result, name, rpc(getattr(backend, name)))
        return result

    router_migration.handle.backend_instance = actor(source)
    target_actor = actor(target)
    monkeypatch.setattr(
        router_module,
        "start_instance",
        SimpleNamespace(
            options=lambda **kwargs: SimpleNamespace(
                remote=rpc(lambda *args: target_actor).remote
            )
        ),
    )
    return SimpleNamespace(
        state=router_migration,
        source=source,
        target=target,
    )


def request_data(request_id="request"):
    return {
        "request_id": request_id,
        "model": "logical",
        "input_tokens": [10, 11],
        "max_tokens": 5,
    }


async def active(migration, backend, request_id="request"):
    task = migration.spawn(backend.generate(request_data(request_id)))
    await joined(backend.engine.stream(request_id).observed.wait())
    return task


async def frozen(migration, backend, epoch="epoch"):
    task = await active(migration, backend)
    envelope = await joined(backend.freeze_migration(epoch))
    return task, envelope


@pytest.mark.asyncio
async def test_freeze_waits_for_abort_tail_and_commit_releases_ticket(
    migration,
):
    backend = migration.create()
    engine = backend.engine
    task = await active(migration, backend)
    engine.stream("request").tail.clear()
    freeze = migration.spawn(backend.freeze_migration("epoch"))
    await joined(engine.stream("request").finish.wait())
    assert not freeze.done()
    assert not task.done()
    engine.stream("request").tail.set()
    envelope = await joined(freeze)
    assert envelope["schema_version"] == 1
    assert envelope["epoch"] == "epoch"
    assert envelope["requests"][0]["choices"][0]["token_ids"] == [20, 21]
    assert not task.done()
    assert engine.close_calls == 0
    await backend.commit_migration("epoch")
    assert await joined(task) == {
        "_sllm_migration": {"epoch": "epoch", "request_id": "request"}
    }
    assert engine.abort_calls == ["request"]


@pytest.mark.asyncio
async def test_rollback_keeps_original_call_and_resumes_same_engine(migration):
    backend = migration.create()
    engine = backend.engine
    task, _ = await frozen(migration, backend)
    await joined(backend.rollback_migration("epoch"))
    await turns()
    assert not task.done()
    assert len(engine.resume_calls) == 1
    key = engine.resume_calls[0][1] + ":0"
    await joined(engine.stream(key).observed.wait())
    engine.stream(key).finish.set()
    result = await joined(task)
    assert result["id"] == "request"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 3
    assert backend.engine is engine
    assert engine.close_calls == 0


@pytest.mark.asyncio
async def test_prepare_is_idle_and_resume_retries_share_one_generation(
    migration,
):
    source, target = migration.create(), migration.create()
    source_call, envelope = await frozen(migration, source)
    assert await target.prepare_migration("epoch", envelope["requests"]) == [
        "request"
    ]
    assert target.engine.resume_calls == target.engine.calls == []
    await source.commit_migration("epoch")
    await joined(source_call)
    calls = [
        migration.spawn(target.resume_migration("epoch", "request"))
        for _ in range(2)
    ]
    await turns()
    assert len(target.engine.resume_calls) == 1
    key = target.engine.resume_calls[0][1] + ":0"
    await joined(target.engine.stream(key).observed.wait())
    target.engine.stream(key).finish.set()
    results = await joined(asyncio.gather(*calls))
    assert results[0] == results[1]
    assert results[0]["usage"]["completion_tokens"] == 3
    assert await target.resume_migration("epoch", "request") == results[0]
    assert len(target.engine.resume_calls) == 1


@pytest.mark.asyncio
async def test_freeze_includes_a_dispatched_request_that_admits_late(migration):
    backend = migration.create()
    freeze = migration.spawn(
        backend.freeze_migration("epoch", request_ids=["late"])
    )
    await turns()
    assert not freeze.done()
    task = migration.spawn(backend.generate(request_data("late")))
    envelope = await joined(freeze)
    assert [item["request_id"] for item in envelope["requests"]] == ["late"]
    assert not task.done()
    await backend.commit_migration("epoch")
    assert (await joined(task))["_sllm_migration"]["request_id"] == "late"


@pytest.mark.asyncio
async def test_late_cancel_tombstone_unblocks_freeze_and_prevents_admission(
    migration,
):
    backend = migration.create()
    freeze = migration.spawn(
        backend.freeze_migration("epoch", request_ids=["late"])
    )
    await turns()
    await backend.cancel_migration_request(None, "late")
    assert (await joined(freeze))["requests"] == []
    outcome = (
        await asyncio.gather(
            backend.generate(request_data("late")), return_exceptions=True
        )
    )[0]
    assert isinstance(outcome, asyncio.CancelledError) or "error" in outcome
    assert backend.engine.calls == []


@pytest.mark.asyncio
async def test_cancelled_prepared_request_cannot_be_resurrected(migration):
    source, target = migration.create(), migration.create()
    source_call, envelope = await frozen(migration, source)
    await target.prepare_migration("epoch", envelope["requests"])
    await target.cancel_migration_request("epoch", "request")
    outcome = (
        await asyncio.gather(
            target.resume_migration("epoch", "request"), return_exceptions=True
        )
    )[0]
    assert isinstance(outcome, BaseException) or "error" in outcome
    assert target.engine.resume_calls == []
    await source.commit_migration("epoch")
    await joined(source_call)


@pytest.mark.asyncio
async def test_force_shutdown_unblocks_frozen_external_request(migration):
    backend = migration.create()
    engine = backend.engine
    task, _ = await frozen(migration, backend)
    await joined(backend.shutdown())
    outcome = (await joined(asyncio.gather(task, return_exceptions=True)))[0]
    assert (
        isinstance(outcome, BaseException) or "_sllm_migration" not in outcome
    )
    assert backend.engine is None
    assert engine.close_calls == 1


@pytest.mark.asyncio
async def test_natural_completion_racing_with_freeze_is_not_migrated(migration):
    backend = migration.create()
    task = await active(migration, backend)
    backend.engine.stream("request").natural_on_abort = True
    envelope = await joined(backend.freeze_migration("epoch"))
    assert envelope["requests"] == []
    result = await joined(task)
    assert result["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_unsupported_runner_is_rejected_before_aborting_source(migration):
    backend = migration.create()
    task = await active(migration, backend)
    backend.engine.supported = False
    with pytest.raises(RuntimeError, match="unsupported runner"):
        await backend.freeze_migration("epoch")
    assert backend.engine.abort_calls == []
    assert not task.done()
    backend.engine.stream("request").finish.set()
    assert (await joined(task))["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_freeze_waits_for_core_registration_before_abort(migration):
    backend = migration.create()
    stream = backend.engine.stream("request")
    stream.registration.clear()
    task = migration.spawn(backend.generate(request_data()))
    await joined(stream.started.wait())
    freeze = migration.spawn(backend.freeze_migration("epoch"))
    await turns()
    assert backend.engine.abort_calls == []
    assert not freeze.done()
    stream.registration.set()
    envelope = await joined(freeze)
    assert envelope["requests"][0]["choices"][0]["token_ids"] == [20, 21]
    await backend.commit_migration("epoch")
    assert "_sllm_migration" in await joined(task)


@pytest.mark.asyncio
async def test_resumed_request_can_migrate_again_with_full_history(migration):
    source, middle, target = [migration.create() for _ in range(3)]
    source_call, first = await frozen(migration, source, "first")
    await middle.prepare_migration("first", first["requests"])
    await source.commit_migration("first")
    await joined(source_call)
    middle_call = migration.spawn(middle.resume_migration("first", "request"))
    await turns()
    middle_key = middle.engine.resume_calls[0][1] + ":0"
    await joined(middle.engine.stream(middle_key).observed.wait())
    second = await joined(middle.freeze_migration("second"))
    assert second["requests"][0]["choices"][0]["token_ids"] == [20, 21, 22]
    await target.prepare_migration("second", second["requests"])
    await middle.commit_migration("second")
    assert await joined(middle_call) == {
        "_sllm_migration": {"epoch": "second", "request_id": "request"}
    }
    final_call = migration.spawn(target.resume_migration("second", "request"))
    await turns()
    final_key = target.engine.resume_calls[0][1] + ":0"
    await joined(target.engine.stream(final_key).observed.wait())
    target.engine.stream(final_key).finish.set()
    result = await joined(final_call)
    assert result["id"] == "request"
    assert result["choices"][0]["message"]["content"] == "20 21 22 23"
    assert result["usage"]["completion_tokens"] == 4
    assert not source.active_requests
    assert not middle.active_requests
    assert not target.active_requests


@pytest.mark.asyncio
async def test_freeze_waits_for_late_resume_and_preserves_previous_history(
    migration,
):
    source, target = migration.create(), migration.create()
    source_call, first = await frozen(migration, source, "first")
    await target.prepare_migration("first", first["requests"])
    await source.commit_migration("first")
    await joined(source_call)
    freeze = migration.spawn(
        target.freeze_migration("second", request_ids=["request"])
    )
    await turns()
    assert not freeze.done()
    resumed = migration.spawn(target.resume_migration("first", "request"))
    second = await joined(freeze)
    assert second["requests"] == first["requests"]
    assert target.engine.resume_calls == []
    await target.commit_migration("second")
    assert (await joined(resumed))["_sllm_migration"]["epoch"] == "second"


@pytest.mark.asyncio
async def test_cancelled_frozen_source_is_not_resumed_by_rollback(migration):
    backend = migration.create()
    task, _ = await frozen(migration, backend)
    await joined(backend.cancel_migration_request(None, "request"))
    await joined(backend.rollback_migration("epoch"))
    assert task.cancelled()
    assert not backend.active_requests
    assert backend.engine.resume_calls == []


@pytest.mark.asyncio
async def test_freeze_rpc_cancellation_retains_drain_for_rollback(migration):
    backend = migration.create()
    task = await active(migration, backend)
    stream = backend.engine.stream("request")
    stream.tail.clear()
    freeze = migration.spawn(backend.freeze_migration("epoch"))
    await joined(stream.finish.wait())
    freeze.cancel()
    await asyncio.gather(freeze, return_exceptions=True)
    assert not task.done()
    rollback = migration.spawn(backend.rollback_migration("epoch"))
    await turns()
    assert not rollback.done()
    stream.tail.set()
    await joined(rollback)
    await turns()
    key = backend.engine.resume_calls[0][1] + ":0"
    await joined(backend.engine.stream(key).observed.wait())
    backend.engine.stream(key).finish.set()
    assert (await joined(task))["usage"]["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_shutdown_settles_freeze_waiting_for_late_admission(migration):
    backend = migration.create()
    engine = backend.engine
    freeze = migration.spawn(
        backend.freeze_migration("epoch", request_ids=["late"])
    )
    await turns()
    await joined(backend.shutdown())
    with pytest.raises(RuntimeError, match="shut down"):
        await joined(freeze)
    assert engine.close_calls == 1
    assert backend.engine is None


@pytest.mark.asyncio
async def test_concurrent_prepare_retries_wait_for_same_prewarm(migration):
    source, target = migration.create(), migration.create()
    source_call, envelope = await frozen(migration, source)
    target.engine.prewarm_release.clear()
    first = migration.spawn(
        target.prepare_migration("epoch", envelope["requests"])
    )
    await joined(target.engine.prewarm_started.wait())
    second = migration.spawn(
        target.prepare_migration("epoch", envelope["requests"])
    )
    await turns()
    assert not first.done()
    assert not second.done()
    assert len(target.engine.prewarm_calls) == 1
    target.engine.prewarm_release.set()
    assert await joined(asyncio.gather(first, second)) == [
        ["request"],
        ["request"],
    ]
    await source.commit_migration("epoch")
    await joined(source_call)


@pytest.mark.asyncio
async def test_rollback_late_admission_starts_original_iterator(migration):
    backend = migration.create()
    freeze = migration.spawn(
        backend.freeze_migration("epoch", request_ids=["late"])
    )
    await turns()
    task = migration.spawn(backend.generate(request_data("late")))
    assert (await joined(freeze))["requests"][0]["choices"] == []
    assert backend.engine.calls == []
    await joined(backend.rollback_migration("epoch"))
    await joined(backend.engine.stream("late").observed.wait())
    assert backend.engine.calls == ["late"]
    assert backend.engine.resume_calls == []
    backend.engine.stream("late").finish.set()
    assert (await joined(task))["usage"]["completion_tokens"] == 2


@pytest.mark.asyncio
async def test_failed_abort_rollback_retains_original_running_iterator(
    migration,
):
    backend = migration.create()
    task = await active(migration, backend)
    backend.engine.abort_errors.add("request")
    with pytest.raises(RuntimeError, match="abort failed"):
        await joined(backend.freeze_migration("epoch"))
    await joined(backend.rollback_migration("epoch"))
    assert not task.done()
    assert backend.engine.resume_calls == []
    backend.engine.stream("request").finish.set()
    assert (await joined(task))["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_failed_freeze_joins_other_abort_tails_before_rollback(migration):
    backend = migration.create()
    failed = await active(migration, backend, "failed")
    paused = await active(migration, backend, "paused")
    backend.engine.abort_errors.add("failed")
    stream = backend.engine.stream("paused")
    stream.tail.clear()
    freeze = migration.spawn(backend.freeze_migration("epoch"))
    await joined(stream.finish.wait())
    await turns()
    assert not freeze.done()
    stream.tail.set()
    with pytest.raises(RuntimeError, match="abort failed"):
        await joined(freeze)
    await joined(backend.rollback_migration("epoch"))
    await turns()
    assert [item[0] for item in backend.engine.resume_calls] == ["paused"]
    key = backend.engine.resume_calls[0][1] + ":0"
    await joined(backend.engine.stream(key).observed.wait())
    backend.engine.stream(key).finish.set()
    backend.engine.stream("failed").finish.set()
    assert (await joined(paused))["usage"]["completion_tokens"] == 3
    assert (await joined(failed))["usage"]["completion_tokens"] == 2


@pytest.mark.asyncio
async def test_unsupported_params_validate_every_request_before_any_abort(
    migration,
):
    backend = migration.create()
    eligible = await active(migration, backend, "eligible")
    unsupported = migration.spawn(
        backend.generate(
            {**request_data("unsupported"), "logits_processors": ["custom"]}
        )
    )
    await joined(backend.engine.stream("unsupported").observed.wait())
    with pytest.raises(ValueError, match="unsupported sampling params"):
        await backend.freeze_migration("epoch")
    assert backend.engine.abort_calls == []
    assert not eligible.done()
    assert not unsupported.done()
    for stream in backend.engine.streams.values():
        stream.finish.set()
    results = await joined(asyncio.gather(eligible, unsupported))
    assert all(
        result["choices"][0]["finish_reason"] == "stop" for result in results
    )


@pytest.mark.asyncio
async def test_shutdown_joins_target_prepare_prewarm(migration):
    source, target = migration.create(), migration.create()
    source_call, envelope = await frozen(migration, source)
    engine = target.engine
    engine.prewarm_release.clear()
    prepare = migration.spawn(
        target.prepare_migration("epoch", envelope["requests"])
    )
    await joined(engine.prewarm_started.wait())
    await joined(target.shutdown())
    outcome = (await joined(asyncio.gather(prepare, return_exceptions=True)))[0]
    assert isinstance(outcome, BaseException)
    assert engine.close_calls == 1
    assert target.engine is None
    assert not target.active_requests
    await source.commit_migration("epoch")
    await joined(source_call)


@pytest.mark.asyncio
async def test_direct_and_prewarm_completion_do_not_retain_migration_records(
    migration,
):
    backend = migration.create()
    task = await active(migration, backend)
    backend.engine.stream("request").finish.set()
    await joined(task)
    await joined(backend.resume_kv_cache([[10, 11]]))
    assert not backend._request_terminal
    assert not backend._migration_tracked
    assert not backend.active_requests


@pytest.mark.asyncio
async def test_tracked_terminal_is_retained_until_router_forgets_it(migration):
    backend = migration.create()
    task = migration.spawn(
        backend.generate(request_data(), migration_tracked=True)
    )
    await joined(backend.engine.stream("request").observed.wait())
    backend.engine.stream("request").finish.set()
    await joined(task)
    assert backend._request_terminal == {"request"}
    assert backend._migration_tracked == {"request"}
    assert (
        await joined(backend.freeze_migration("epoch", request_ids=["request"]))
    )["requests"] == []
    await backend.forget_migration_request("request")
    assert backend._request_terminal == {"request"}
    await backend.rollback_migration("epoch")
    assert not backend._request_terminal
    assert not backend._migration_tracked


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_on_source", [False, True])
async def test_delta_generation_resumes_cumulative_history_without_duplication(
    migration,
    resume_on_source,
):
    source = migration.create()
    source.trace_debug = True
    source.engine.with_logprobs = True
    source_call = migration.spawn(
        source.generate({**request_data(), "max_tokens": 3, "output_kind": 1})
    )
    await joined(source.engine.stream("request").observed.wait())
    envelope = await joined(source.freeze_migration("epoch"))
    snapshot = envelope["requests"][0]
    assert snapshot["sampling_params"]["max_tokens"] == 3
    assert snapshot["sampling_params"]["output_kind"] == 1
    assert snapshot["choices"][0]["token_ids"] == [20, 21]
    assert snapshot["choices"][0]["text"] == "20 21"
    assert snapshot["choices"][0]["logprobs"] == [{20: -0.5}, {21: -0.5}]
    if resume_on_source:
        target = source
        await source.rollback_migration("epoch")
        resumed = source_call
    else:
        target = migration.create()
        target.trace_debug = True
        target.engine.with_logprobs = True
        await target.prepare_migration("epoch", envelope["requests"])
        await source.commit_migration("epoch")
        await joined(source_call)
        resumed = migration.spawn(target.resume_migration("epoch", "request"))
    await turns()
    key = target.engine.resume_calls[0][1] + ":0"
    await joined(target.engine.stream(key).observed.wait())
    target.engine.stream(key).finish.set()
    result = await joined(resumed)
    assert result["choices"][0]["message"]["content"] == "20 21 22"
    assert result["choices"][0]["logprobs"] == [
        {20: -0.5},
        {21: -0.5},
        {22: -0.5},
    ]
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert target.request_history["request"].outputs[0].token_ids == [
        20,
        21,
        22,
    ]
    assert snapshot["choices"][0]["token_ids"] == [20, 21]


def test_delta_merge_tracks_each_choice_and_preserves_completed_choices():
    from sllm.backends.vllm_migration import merge_output

    first = output("request", [20])
    first.outputs[0].logprobs = [{20: -0.5}]
    first.outputs[0].cumulative_logprob = -0.5
    another = output("request", [30], "stop").outputs[0]
    another.index = 1
    another.logprobs = [{30: -0.3}]
    another.cumulative_logprob = -0.3
    first.outputs.append(another)
    update = output("request", [21], "length")
    update.outputs[0].text = " 21"
    update.outputs[0].logprobs = [{21: -0.7}]
    update.outputs[0].cumulative_logprob = -1.2
    merged = merge_output(first, update, delta=True)
    assert [choice.token_ids for choice in merged.outputs] == [[20, 21], [30]]
    assert [choice.finish_reason for choice in merged.outputs] == [
        "length",
        "stop",
    ]
    assert merged.outputs[0].text == "20 21"
    assert merged.outputs[0].logprobs == [{20: -0.5}, {21: -0.7}]
    assert merged.outputs[0].cumulative_logprob == -1.2
    assert first.outputs[0].token_ids == [20]
    assert first.outputs[0].logprobs == [{20: -0.5}]


@pytest.mark.asyncio
async def test_forget_terminal_during_freeze_does_not_lose_expected_completion(
    migration,
):
    backend = migration.create()
    done = migration.spawn(
        backend.generate(request_data("done"), migration_tracked=True)
    )
    await joined(backend.engine.stream("done").observed.wait())
    backend.engine.stream("done").finish.set()
    await joined(done)
    freeze = migration.spawn(
        backend.freeze_migration("epoch", request_ids=["done", "late"])
    )
    await turns()
    assert not freeze.done()
    await backend.forget_migration_request("done")
    await backend.cancel_migration_request(None, "late")
    assert (await joined(freeze))["requests"] == []
    await backend.rollback_migration("epoch")
    assert "done" not in backend._request_terminal
    assert "done" not in backend._migration_tracked


@pytest.mark.asyncio
async def test_router_backend_handoff_preserves_one_response_and_slot(
    connected_router,
):
    state = connected_router.state
    source, target = connected_router.source, connected_router.target
    source_engine = source.engine
    task, allocation = await router_request(state, dispatch=False)
    try:
        await joined(source_engine.stream("request").observed.wait())
        assert (
            await joined(state.router.execute_migration_plan(state.plan))
            == "target"
        )
        assert source.engine is None
        assert source_engine.close_calls == 1
        assert len(target.engine.resume_calls) == 1
        owner = state.router.allocated_instances[allocation]
        assert owner.instance_id == "target"
        assert owner.concurrency == state.router.request_count == 1
        assert state.handle.concurrency == 0
        assert state.reservations == {"target"}
        assert not task.done()
        key = target.engine.resume_calls[0][1] + ":0"
        await joined(target.engine.stream(key).observed.wait())
        target.engine.stream(key).finish.set()
        result = await joined(task)
        assert result["id"] == "request"
        assert result["choices"][0]["message"]["content"] == "20 21 22"
        assert result["usage"]["completion_tokens"] == 3
        assert "_sllm_migration" not in result
        assert owner.concurrency == state.router.request_count == 0
        assert not state.router.allocated_instances
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_router_backend_prepare_failure_rolls_back_original_request(
    connected_router,
):
    state = connected_router.state
    source, target = connected_router.source, connected_router.target
    engine = source.engine
    target.engine.supported = False
    task, allocation = await router_request(state, dispatch=False)
    try:
        await joined(engine.stream("request").observed.wait())
        with pytest.raises(RuntimeError, match="unsupported runner"):
            await joined(state.router.execute_migration_plan(state.plan))
        assert source.engine is engine
        assert engine.close_calls == 0
        assert target.engine is None
        assert state.router.allocated_instances[allocation] is state.handle
        assert state.handle.concurrency == state.router.request_count == 1
        assert state.reservations == {"source"}
        assert len(engine.resume_calls) == 1
        key = engine.resume_calls[0][1] + ":0"
        await joined(engine.stream(key).observed.wait())
        engine.stream(key).finish.set()
        assert (await joined(task))["usage"]["completion_tokens"] == 3
        assert state.handle.concurrency == state.router.request_count == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_router_backend_cancel_during_prepare_never_resumes_request(
    connected_router,
):
    state = connected_router.state
    source, target = connected_router.source, connected_router.target
    task, _ = await router_request(state, dispatch=False)
    plan = None
    try:
        await joined(source.engine.stream("request").observed.wait())
        target.engine.prewarm_release.clear()
        plan = asyncio.create_task(
            state.router.execute_migration_plan(state.plan)
        )
        await joined(target.engine.prewarm_started.wait())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await joined(task)
        assert source.engine is not None
        assert not source.active_requests
        target.engine.prewarm_release.set()
        assert await joined(plan) == "target"
        assert target.engine.resume_calls == []
        assert state.router.request_count == 0
        assert state.router.ready_inference_instances["target"].concurrency == 0
        assert state.reservations == {"target"}
    finally:
        if target.engine is not None:
            target.engine.prewarm_release.set()
        task.cancel()
        if plan is not None:
            plan.cancel()
        await asyncio.gather(
            *(item for item in (task, plan) if item is not None),
            return_exceptions=True,
        )
