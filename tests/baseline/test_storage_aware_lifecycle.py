"""CPU reservation ownership regressions for the storage-aware scheduler."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import sllm.schedulers.storage_aware_scheduler as module
from sllm.utils import InstanceStatus


async def run_iteration(scheduler, plan, before_plan=None, host_load=None):
    async def schedule(*args):
        scheduler.running = False
        if before_plan:
            await before_plan()
        return [plan]

    scheduler.schedule = schedule
    scheduler._get_worker_nodes = AsyncMock(
        side_effect=lambda: copy.deepcopy(scheduler.worker_nodes)
    )
    scheduler.store_manager = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=AsyncMock(return_value={}))
            for name in (
                "get_model_info",
                "get_store_info",
                "get_hardware_info",
                "load_to_host",
            )
        }
    )
    if host_load is not None:
        scheduler.store_manager.load_to_host.remote = host_load
    scheduler.running = True
    await asyncio.wait_for(scheduler._control_loop(), 2)


async def enqueue(scheduler, name="new", instance="new_i", count=1):
    task = asyncio.create_task(
        scheduler.allocate_resource(name, instance, {"num_gpus": count})
    )
    for _ in range(100):
        if (name, instance) in scheduler.pending_allocations:
            return task
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    pytest.fail("allocation did not enqueue")


async def cancel(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_storage_allocation_installs_owner_before_publication_and_returns_capacity():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 2}}
    request = await enqueue(scheduler, count=2)
    try:
        await run_iteration(scheduler, module.AllocationPlan("0", 1))
        assert await asyncio.wait_for(request, 1) == "0"
        assert scheduler.model_instance == {"new": {"new_i": "0"}}
        assert scheduler.worker_nodes["0"]["free_gpu"] == 0
        await scheduler.deallocate_resource("new", "new_i", {"num_gpus": 2})
        await scheduler.deallocate_resource("new", "new_i", {"num_gpus": 2})
        assert scheduler.worker_nodes["0"]["free_gpu"] == 2
    finally:
        await cancel(request)


@pytest.mark.asyncio
async def test_mark_reserves_actual_migrant_size_and_is_idempotent():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 3}, "1": {"free_gpu": 3}}
    assert await scheduler.mark_resource("old", "migrant", "0", {"num_gpus": 2})
    assert await scheduler.mark_resource("old", "migrant", "0", {"num_gpus": 2})
    assert not await scheduler.mark_resource(
        "old", "migrant", "1", {"num_gpus": 2}
    )
    assert not await scheduler.mark_resource(
        "old", "other", "0", {"num_gpus": 2}
    )
    assert scheduler.worker_nodes["0"]["free_gpu"] == 1
    assert scheduler.worker_nodes["1"]["free_gpu"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node,count", [("missing", 1), ("0", 0), ("0", -1), ("0", 1.5)]
)
async def test_mark_rejects_invalid_reservation(node, count):
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 2}}
    assert not await scheduler.mark_resource(
        "old", "migrant", node, {"num_gpus": count}
    )
    assert not scheduler.model_instance
    assert scheduler.worker_nodes["0"]["free_gpu"] == 2


@pytest.mark.asyncio
async def test_failed_migration_keeps_dependent_allocation_pending(monkeypatch):
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 0}, "1": {"free_gpu": 2}}
    plans = module.MigrationPlans()
    plans.append(
        module.MigrationPlan(
            "1", InstanceStatus("old", "0", 1, 2, model_name="old")
        ),
        1,
    )
    monkeypatch.setattr(
        module.ray,
        "get_actor",
        lambda *a, **kw: SimpleNamespace(
            execute_migration_plan=SimpleNamespace(
                remote=AsyncMock(return_value=None)
            )
        ),
    )
    request = await enqueue(scheduler)
    try:
        await run_iteration(scheduler, module.AllocationPlan("0", 1, plans))
        assert not request.done()
        assert scheduler.worker_nodes == {
            "0": {"free_gpu": 0},
            "1": {"free_gpu": 2},
        }
        assert not scheduler.model_instance
    finally:
        await cancel(request)


@pytest.mark.asyncio
async def test_concurrent_reservation_cannot_be_overwritten_or_overallocated():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 1}}
    request = await enqueue(scheduler)

    async def competing_reservation():
        assert await scheduler.mark_resource(
            "old", "migrant", "0", {"num_gpus": 1}
        )

    try:
        await run_iteration(
            scheduler, module.AllocationPlan("0", 1), competing_reservation
        )
        assert not request.done()
        assert scheduler.model_instance == {"old": {"migrant": "0"}}
        assert scheduler.worker_nodes["0"]["free_gpu"] == 0
    finally:
        await cancel(request)


@pytest.mark.asyncio
async def test_cancel_during_plan_does_not_allocate_or_leave_queue():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 1}}
    request = await enqueue(scheduler)
    await run_iteration(
        scheduler, module.AllocationPlan("0", 1), lambda: cancel(request)
    )
    assert request.cancelled()
    assert scheduler.model_loading_queues["new"] == []
    assert not scheduler.pending_allocations
    assert not scheduler.model_instance
    assert scheduler.worker_nodes["0"]["free_gpu"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_failure", [False, True])
async def test_migration_callbacks_commit_actual_size_without_snapshot_overwrite(
    monkeypatch, partial_failure
):
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 0}, "1": {"free_gpu": 3}}
    scheduler.model_instance = {"old": {"source": "0"}}
    plans = module.MigrationPlans()
    plans.append(
        module.MigrationPlan(
            "1", InstanceStatus("source", "0", 1, 2, model_name="old")
        ),
        1,
    )
    if partial_failure:
        plans.append(
            module.MigrationPlan(
                "1", InstanceStatus("second", "0", 1, 1, model_name="old")
            ),
            1,
        )

    async def migrate(plan):
        if plan.source_instance.instance_id == "second":
            return None
        assert await scheduler.mark_resource(
            "old", "target", "1", {"num_gpus": 2}
        )
        await scheduler.deallocate_resource("old", "source", {"num_gpus": 2})
        return "target"

    monkeypatch.setattr(
        module.ray,
        "get_actor",
        lambda *a, **kw: SimpleNamespace(
            execute_migration_plan=SimpleNamespace(remote=migrate)
        ),
    )
    request = await enqueue(scheduler)
    try:
        await run_iteration(scheduler, module.AllocationPlan("0", 1, plans))
        assert scheduler.model_instance["old"] == {"target": "1"}
        assert scheduler.worker_nodes["1"]["free_gpu"] == 1
        if partial_failure:
            assert not request.done()
            assert scheduler.worker_nodes["0"]["free_gpu"] == 2
        else:
            assert await asyncio.wait_for(request, 1) == "0"
            assert scheduler.worker_nodes["0"]["free_gpu"] == 1
    finally:
        await cancel(request)


@pytest.mark.asyncio
async def test_instance_status_callback_can_reserve_without_lock_deadlock(
    monkeypatch,
):
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 0}, "1": {"free_gpu": 1}}
    scheduler.model_instance = {"old": {"source": "0"}}

    async def status(instance):
        assert await scheduler.mark_resource("other", "callback", "1")
        return None

    monkeypatch.setattr(
        module.ray,
        "get_actor",
        lambda *a, **kw: SimpleNamespace(
            get_instance_status=SimpleNamespace(remote=status)
        ),
    )
    assert (
        await asyncio.wait_for(
            scheduler.get_migration_plans(
                "new",
                1,
                "0",
                scheduler.worker_nodes,
                {},
                {},
                {},
            ),
            1,
        )
        is None
    )
    assert scheduler.worker_nodes["1"]["free_gpu"] == 0


@pytest.mark.asyncio
async def test_cancel_during_host_load_returns_committed_reservation_once():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 1}}
    request = await enqueue(scheduler)

    async def load(node, model):
        assert scheduler.model_instance == {"new": {"new_i": "0"}}
        assert scheduler.worker_nodes["0"]["free_gpu"] == 0
        await cancel(request)

    await run_iteration(
        scheduler,
        module.AllocationPlan("0", 1),
        host_load=load,
    )
    assert request.cancelled()
    assert not scheduler.pending_allocations
    assert scheduler.model_loading_queues["new"] == []
    assert scheduler.model_instance["new"] == {}
    assert scheduler.worker_nodes["0"]["free_gpu"] == 1


@pytest.mark.asyncio
async def test_two_same_model_requests_use_identity_after_queue_removal():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 2}}
    first = await enqueue(scheduler, instance="first")
    second = await enqueue(scheduler, instance="second")
    try:
        await run_iteration(scheduler, module.AllocationPlan("0", 1))
        assert await asyncio.wait_for(first, 1) == "0"
        assert await asyncio.wait_for(second, 1) == "0"
        assert scheduler.model_instance == {
            "new": {"first": "0", "second": "0"}
        }
        assert scheduler.worker_nodes["0"]["free_gpu"] == 0
    finally:
        await cancel(first)
        await cancel(second)


@pytest.mark.asyncio
async def test_host_load_failure_preserves_published_owner_and_next_allocation():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 2}}
    first = await enqueue(scheduler, instance="first")
    second = await enqueue(scheduler, instance="second")
    loads = []

    async def load(node, model):
        loads.append((node, model))
        if len(loads) == 1:
            raise RuntimeError("host load failed")

    try:
        await run_iteration(
            scheduler,
            module.AllocationPlan("0", 1),
            host_load=load,
        )
        assert await asyncio.wait_for(first, 1) == "0"
        assert await asyncio.wait_for(second, 1) == "0"
        assert loads == [("0", "new"), ("0", "new")]
        assert scheduler.model_instance == {
            "new": {"first": "0", "second": "0"}
        }
        assert scheduler.worker_nodes["0"]["free_gpu"] == 0
        await scheduler.deallocate_resource("new", "first", {"num_gpus": 1})
        await scheduler.deallocate_resource("new", "second", {"num_gpus": 1})
        assert scheduler.worker_nodes["0"]["free_gpu"] == 2
    finally:
        await cancel(first)
        await cancel(second)


@pytest.mark.asyncio
async def test_cancel_waiting_for_metadata_lock_preserves_loop_and_capacity():
    scheduler = module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"0": {"free_gpu": 1}}
    first = await enqueue(scheduler, instance="first")
    metadata_held = asyncio.Event()
    released = False
    second = None

    async def hold_metadata():
        await scheduler.metadata_lock.acquire()
        metadata_held.set()

    iteration = asyncio.create_task(
        run_iteration(
            scheduler,
            module.AllocationPlan("0", 1),
            hold_metadata,
        )
    )
    try:
        await asyncio.wait_for(metadata_held.wait(), 1)
        for _ in range(100):
            if scheduler.queue_lock.locked():
                break
            await asyncio.sleep(0)
        assert scheduler.queue_lock.locked()
        first.cancel()
        await asyncio.sleep(0)
        assert scheduler.pending_allocations[("new", "first")].cancelled()
        scheduler.metadata_lock.release()
        released = True
        await asyncio.wait_for(iteration, 2)
        await cancel(first)
        assert not scheduler.model_instance
        assert scheduler.worker_nodes["0"]["free_gpu"] == 1
        assert scheduler.model_loading_queues["new"] == []
        second = await enqueue(scheduler, instance="second")
        await run_iteration(scheduler, module.AllocationPlan("0", 1))
        assert await asyncio.wait_for(second, 1) == "0"
        assert scheduler.model_instance == {"new": {"second": "0"}}
        assert scheduler.worker_nodes["0"]["free_gpu"] == 0
        await scheduler.deallocate_resource("new", "second", {"num_gpus": 1})
        assert scheduler.worker_nodes["0"]["free_gpu"] == 1
    finally:
        if metadata_held.is_set() and not released:
            scheduler.metadata_lock.release()
        await cancel(iteration)
        await cancel(first)
        if second is not None:
            await cancel(second)
