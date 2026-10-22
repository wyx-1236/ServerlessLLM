"""Migration ownership regressions; all RPCs are CPU in-process doubles."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_migration_contract import chain as chain
from test_migration_contract import legacy as legacy
from test_migration_contract import rpc

import sllm.routers.migration_router as router_module


async def join(task):
    return await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_replay_error_body_rolls_back_target(chain):
    chain.target.resume_kv_cache = rpc(
        lambda tokens: {"error": "replay refused"}
    )
    with pytest.raises(RuntimeError, match="replay refused"):
        await join(chain.router.execute_migration_plan(chain.plan))
    assert list(chain.router.ready_inference_instances) == ["source"]
    assert chain.reservations == {"source": "0"}


@pytest.mark.asyncio
async def test_backend_replay_error_is_not_discarded(legacy):
    backend = legacy()
    backend.generate = AsyncMock(return_value={"error": "engine closed"})
    with pytest.raises(RuntimeError, match="engine closed"):
        await backend.resume_kv_cache([[1, 2]])
    backend.generate.assert_awaited_once_with(
        {"input_tokens": [1, 2], "max_tokens": 1}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["reservation", "creation", "replay"])
async def test_cancel_observes_late_ownership_and_releases_target(
    chain, monkeypatch, stage
):
    started, release = asyncio.Event(), asyncio.Event()

    async def pause():
        started.set()
        await release.wait()
        if stage == "reservation":
            chain.reservations["target"] = "1"
            return True
        if stage == "creation":
            return chain.target

    if stage == "reservation":
        chain.router.model_loading_scheduler.mark_resource.remote = (
            lambda *a: pause()
        )
    elif stage == "creation":
        monkeypatch.setattr(
            router_module,
            "start_instance",
            SimpleNamespace(
                options=lambda **kw: SimpleNamespace(remote=lambda *a: pause())
            ),
        )
    else:
        chain.target.resume_kv_cache.remote = lambda *a: pause()
    task = asyncio.create_task(chain.router.execute_migration_plan(chain.plan))
    try:
        await join(started.wait())
        task.cancel()
        if stage != "replay":
            # Cancellation reaches the transaction before its RPC returns.
            for _ in range(3):
                await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await join(task)
        assert chain.reservations == {"source": "0"}
        assert list(chain.router.ready_inference_instances) == ["source"]
        if stage != "reservation":
            assert ("kill_target", None) in chain.events
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_cancels_replay_and_prevents_late_publication(chain):
    started, release = asyncio.Event(), asyncio.Event()

    async def pause(*args):
        started.set()
        await release.wait()

    chain.target.resume_kv_cache.remote = pause
    task = asyncio.create_task(chain.router.execute_migration_plan(chain.plan))
    try:
        await join(started.wait())
        await join(chain.router.shutdown())
        assert task.done()
        assert not chain.router.ready_inference_instances
        assert not chain.reservations
        assert not chain.router.migration_record
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cleanup_failure_still_kills_and_deallocates_target(chain):
    chain.target.init_backend = rpc(
        lambda: (_ for _ in ()).throw(RuntimeError("init failed"))
    )
    chain.target.shutdown = rpc(
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed"))
    )
    with pytest.raises(RuntimeError):
        await join(chain.router.execute_migration_plan(chain.plan))
    assert ("kill_target", None) in chain.events
    assert chain.reservations == {"source": "0"}


@pytest.mark.asyncio
@pytest.mark.parametrize("allocated", [False, True])
async def test_request_cancel_releases_pending_or_allocated_slot(
    chain, allocated
):
    router = chain.router
    started = asyncio.Event()

    async def generate(**kw):
        started.set()
        await asyncio.Event().wait()

    chain.source.generate = SimpleNamespace(remote=generate)
    task = asyncio.create_task(router.inference({}, "generate"))
    handle = router.ready_inference_instances["source"]
    try:
        future = await join(router.request_queue.get())
        if allocated:
            handle.concurrency = 1
            router.allocated_instances[future] = handle
            future.set_result("source")
            await join(started.wait())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await join(task)
        assert router.request_count == handle.concurrency == 0
        assert not router.pending_allocations
        assert not router.allocated_instances
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_status_rpc_does_not_hold_metadata_lock(chain):
    async def tokens():
        assert not chain.router.instance_management_lock.locked()
        return [[1]]

    chain.source.get_current_tokens.remote = tokens
    assert (
        await join(chain.router.get_instance_status("source"))
    ).num_current_tokens == 1


@pytest.mark.asyncio
async def test_missing_instance_releases_reserved_request(chain):
    router = chain.router
    handle = router.ready_inference_instances.pop("source")
    handle.concurrency = 1
    task = asyncio.create_task(router.inference({}, "generate"))
    try:
        future = await join(router.request_queue.get())
        router.allocated_instances[future] = handle
        future.set_result("source")
        assert await join(task) == {"error": "Instance not found"}
        assert router.request_count == handle.concurrency == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_preempted_transformers_branch_copies_token_budget(chain):
    router = chain.router
    router.backend = "transformers"
    chain.source.generate = rpc(
        lambda **kw: {
            "preempted": True,
            "completed_tokens": 2,
            "current_output": [1],
        }
    )
    chain.target.resume_generate = SimpleNamespace(
        remote=AsyncMock(return_value={"done": True})
    )
    router.migration_record["source"] = "target"
    router.ready_inference_instances["target"] = SimpleNamespace(
        backend_instance=chain.target
    )
    data = {"max_tokens": 4}
    task = asyncio.create_task(router.inference(data, "generate"))
    try:
        future = await join(router.request_queue.get())
        router.ready_inference_instances["source"].concurrency = 1
        future.set_result("source")
        assert await join(task) == {"done": True}
        assert data == {"max_tokens": 4}
        assert chain.target.resume_generate.remote.call_args.kwargs[
            "request_data"
        ] == {"max_tokens": 2}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_source_disappears_during_replay_releases_target(chain):
    def replay(tokens):
        chain.router.ready_inference_instances.pop("source")

    chain.target.resume_kv_cache = rpc(replay)
    assert await join(chain.router.execute_migration_plan(chain.plan)) is None
    assert chain.reservations == {"source": "0"}
    assert not chain.router.migration_record
    assert ("kill_target", None) in chain.events


@pytest.mark.asyncio
async def test_repeated_source_plan_has_only_one_target_owner(chain):
    results = await join(
        asyncio.gather(
            chain.router.execute_migration_plan(chain.plan),
            chain.router.execute_migration_plan(chain.plan),
        )
    )
    assert sorted(results, key=str) == [None, "target"]
    assert sum(name == "mark" for name, _ in chain.events) == 1
    assert chain.reservations == {"target": "1"}


@pytest.mark.asyncio
async def test_source_request_releases_slot_after_cutover(chain):
    started, release = asyncio.Event(), asyncio.Event()

    async def generate(**kw):
        started.set()
        await release.wait()
        return {"choices": [{"finish_reason": "abort"}]}

    chain.source.generate = SimpleNamespace(remote=generate)
    router = chain.router
    source = router.ready_inference_instances["source"]
    task = asyncio.create_task(router.inference({}, "generate"))
    try:
        future = await join(router.request_queue.get())
        source.concurrency = 1
        router.allocated_instances[future] = source
        future.set_result("source")
        await join(started.wait())
        assert await join(router.execute_migration_plan(chain.plan)) == "target"
        release.set()
        assert (await join(task))["choices"][0]["finish_reason"] == "abort"
        assert router.request_count == source.concurrency == 0
        assert not router.allocated_instances
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_creation_failure_returns_reservation_without_touching_source(
    chain, monkeypatch
):
    def fail(*a):
        raise RuntimeError("actor creation failed")

    monkeypatch.setattr(
        router_module,
        "start_instance",
        SimpleNamespace(options=lambda **kw: SimpleNamespace(remote=fail)),
    )
    with pytest.raises(RuntimeError, match="actor creation failed"):
        await join(chain.router.execute_migration_plan(chain.plan))
    assert chain.reservations == {"source": "0"}
    assert list(chain.router.ready_inference_instances) == ["source"]


@pytest.mark.asyncio
async def test_close_wakes_queued_requests(chain):
    router = chain.router
    task = asyncio.create_task(router.inference({}, "generate"))
    try:
        await join(router.request_queue.get())
        await join(router.shutdown())
        assert await join(task) == {"error": "Instance cancelled"}
        assert router.request_count == 0
        assert not router.pending_allocations
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["init_backend", "resume_kv_cache"])
async def test_actual_scheduler_loop_keeps_request_pending_after_router_failure(
    chain, monkeypatch, failure_at
):
    from test_storage_aware_lifecycle import cancel, enqueue, run_iteration

    import sllm.schedulers.storage_aware_scheduler as scheduler_module

    scheduler = scheduler_module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 0}, "1": {"free_gpu": 1}}
    scheduler.model_instance = {"model": {"source": "0"}}
    chain.router.model_loading_scheduler = SimpleNamespace(
        mark_resource=SimpleNamespace(remote=scheduler.mark_resource),
        deallocate_resource=SimpleNamespace(
            remote=scheduler.deallocate_resource
        ),
    )

    def fail(*a):
        raise RuntimeError("target failed")

    setattr(chain.target, failure_at, rpc(fail))
    plans = scheduler_module.MigrationPlans()
    plans.append(chain.plan, 1)
    monkeypatch.setattr(
        scheduler_module.ray,
        "get_actor",
        lambda *a, **kw: SimpleNamespace(
            execute_migration_plan=SimpleNamespace(
                remote=chain.router.execute_migration_plan
            )
        ),
    )
    request = await enqueue(scheduler)
    try:
        await run_iteration(
            scheduler, scheduler_module.AllocationPlan("0", 1, plans)
        )
        assert not request.done()
        assert scheduler.worker_nodes == {
            "0": {"free_gpu": 0},
            "1": {"free_gpu": 1},
        }
        assert scheduler.model_instance == {"model": {"source": "0"}}
        assert list(chain.router.ready_inference_instances) == ["source"]
    finally:
        await cancel(request)


@pytest.mark.asyncio
async def test_source_snapshot_rpc_failure_releases_only_target(chain):
    def fail():
        raise RuntimeError("source actor died")

    chain.source.get_current_tokens = rpc(fail)
    with pytest.raises(RuntimeError, match="source actor died"):
        await join(chain.router.execute_migration_plan(chain.plan))
    assert chain.reservations == {"source": "0"}
    assert list(chain.router.ready_inference_instances) == ["source"]
    assert not chain.router.migration_record
    assert ("kill_target", None) in chain.events


@pytest.mark.asyncio
async def test_kill_failure_still_returns_target_reservation(
    chain, monkeypatch
):
    import ray

    def fail(*args, **kw):
        raise RuntimeError("kill failed")

    chain.target.init_backend = rpc(
        lambda: (_ for _ in ()).throw(RuntimeError("init failed"))
    )
    monkeypatch.setattr(ray, "kill", fail)
    with pytest.raises(RuntimeError, match="kill failed"):
        await join(chain.router.execute_migration_plan(chain.plan))
    assert chain.reservations == {"source": "0"}
    assert list(chain.router.ready_inference_instances) == ["source"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_request_finalization_waits_for_locked_slot_after_cancel(
    chain, cancel_count
):
    router = chain.router
    handle = router.ready_inference_instances["source"]
    started, complete, finalizing = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def generate(**kwargs):
        started.set()
        await complete.wait()
        return {"done": True}

    add_requests = handle.add_requests

    async def release_slot(delta):
        finalizing.set()
        return await add_requests(delta)

    handle.add_requests = release_slot
    chain.source.generate = SimpleNamespace(remote=generate)
    task = asyncio.create_task(router.inference({}, "generate"))
    acquired = False
    try:
        future = await join(router.request_queue.get())
        handle.concurrency = 1
        router.allocated_instances[future] = handle
        future.set_result("source")
        await join(started.wait())
        await handle.lock.acquire()
        acquired = True
        complete.set()
        await join(finalizing.wait())
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        assert (
            not task.done()
        ), "request lost cleanup ownership while slot return was blocked"
        handle.lock.release()
        acquired = False
        with pytest.raises(asyncio.CancelledError):
            await join(task)
        assert router.request_count == handle.concurrency == 0
        assert not router.pending_allocations
        assert not router.allocated_instances
    finally:
        if acquired:
            handle.lock.release()
        complete.set()
        task.cancel()
        await join(asyncio.gather(task, return_exceptions=True))
