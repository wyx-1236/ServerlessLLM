"""CPU evidence for service lifecycle ownership, independent of CUDA."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sllm.routers.roundrobin_router import RoundRobinRouter


def method(fn):
    return SimpleNamespace(remote=fn)


@pytest.mark.asyncio
async def test_router_releases_counts_when_backend_raises():
    router = RoundRobinRouter(
        "test", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    router.running = True
    backend = SimpleNamespace(
        generate=method(AsyncMock(side_effect=ValueError("bad request")))
    )
    from sllm.utils import InstanceHandle

    handle = InstanceHandle(
        "instance", 2, 1, backend_instance=backend, ready=True
    )
    router.ready_inference_instances["instance"] = handle

    async def allocate():
        future = await router.request_queue.get()
        await handle.add_requests(1)
        future.set_result("instance")

    allocation = asyncio.create_task(allocate())
    with pytest.raises(ValueError, match="bad request"):
        await router.inference({}, "generate")
    await allocation
    assert router.request_count == 0
    assert handle.concurrency == 0


@pytest.mark.asyncio
async def test_router_shutdown_wakes_queued_request_and_stops_loops():
    router = RoundRobinRouter(
        "test", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    router.running = True
    router.auto_scaler = asyncio.create_task(asyncio.sleep(60))
    router.load_balancer = asyncio.create_task(asyncio.sleep(60))
    pending = asyncio.create_task(router.inference({}, "generate"))
    await asyncio.sleep(0)
    await asyncio.wait_for(router.shutdown(), 1)
    assert (await asyncio.wait_for(pending, 1))["error"] == "Instance cancelled"
    assert router.request_count == 0
    assert router.auto_scaler.done() and router.load_balancer.done()


@pytest.mark.asyncio
async def test_controller_delete_awaits_router_cleanup(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    monkeypatch.setitem(
        sys.modules, "sllm.store_manager", SimpleNamespace(StoreManager=object)
    )
    path = Path(__file__).parents[2] / "sllm/controller.py"
    spec = importlib.util.spec_from_file_location(
        "controller_lifecycle_probe", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SllmController = module.SllmController
    import ray

    shutdown = AsyncMock(return_value=[])
    router = SimpleNamespace(shutdown=method(shutdown))
    controller = SllmController.__new__(SllmController)
    controller.request_routers = {"test": router}
    controller.registered_models = {"test": {}}
    killed = []
    monkeypatch.setattr(ray, "kill", lambda actor, **kw: killed.append(actor))
    controller.metadata_lock = asyncio.Lock()
    await controller.delete("test")
    shutdown.assert_awaited_once()
    assert killed == [router]
    assert not controller.request_routers and not controller.registered_models


@pytest.mark.asyncio
async def test_router_delete_escalates_and_joins_autoscale_drain(monkeypatch):
    import ray

    from sllm.utils import InstanceHandle

    router = RoundRobinRouter(
        "test", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    router.running = True
    draining = asyncio.Event()
    released = asyncio.Event()

    async def stop():
        draining.set()
        await released.wait()

    async def shutdown():
        released.set()

    actor = SimpleNamespace(stop=method(stop), shutdown=method(shutdown))
    router.ready_inference_instances["i"] = InstanceHandle(
        "i", 1, 1, backend_instance=actor, ready=True
    )
    deallocate = AsyncMock()
    router.model_loading_scheduler = SimpleNamespace(
        deallocate_resource=method(deallocate)
    )
    killed = []
    monkeypatch.setattr(ray, "kill", lambda actor, **kw: killed.append(actor))
    await router._stop_instance("i")
    await draining.wait()
    await asyncio.wait_for(router.shutdown(), 1)
    assert killed == [actor]
    deallocate.assert_awaited_once()
    assert not router.deleting_inference_instances
    assert not router.finish_tasks


@pytest.mark.asyncio
async def test_backend_shutdown_error_still_returns_actor_and_resources(
    monkeypatch,
):
    import ray

    from sllm.utils import InstanceHandle

    router = RoundRobinRouter(
        "test", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    actor = SimpleNamespace(
        shutdown=method(AsyncMock(side_effect=RuntimeError("engine died")))
    )
    router.ready_inference_instances["i"] = InstanceHandle(
        "i", 1, 1, backend_instance=actor, ready=True
    )
    deallocate = AsyncMock()
    router.model_loading_scheduler = SimpleNamespace(
        deallocate_resource=method(deallocate)
    )
    killed = []
    monkeypatch.setattr(ray, "kill", lambda actor, **kw: killed.append(actor))
    with pytest.raises(RuntimeError, match="engine died"):
        await router._shutdown_instance("i")
    assert killed == [actor]
    deallocate.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_cancelled_wait_removes_pending_request():
    from sllm.schedulers.fcfs_scheduler import FcfsScheduler

    scheduler = FcfsScheduler({})
    task = asyncio.create_task(
        scheduler.allocate_resource("test", "i", {"num_gpus": 1})
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not scheduler.pending_allocations
    assert not scheduler.model_loading_queues["test"]
    assert not scheduler.model_instance


@pytest.mark.asyncio
async def test_scheduler_reservation_is_returned_once(monkeypatch):
    import sllm.schedulers.fcfs_scheduler as module

    scheduler = module.FcfsScheduler({})
    monkeypatch.setattr(
        module,
        "get_worker_nodes",
        lambda: {"0": {"free_gpu": 1, "total_gpu": 1}},
    )
    await scheduler.start()
    assert (
        await asyncio.wait_for(
            scheduler.allocate_resource("test", "i", {"num_gpus": 1}), 2
        )
        == "0"
    )
    assert scheduler.worker_nodes["0"]["free_gpu"] == 0
    await scheduler.deallocate_resource("test", "i", {"num_gpus": 1})
    await scheduler.deallocate_resource("test", "i", {"num_gpus": 1})
    assert scheduler.worker_nodes["0"]["free_gpu"] == 1
    await scheduler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_fails", [False, True])
async def test_controller_shutdown_releases_all_owned_actors(
    monkeypatch, shutdown_fails
):
    import importlib.util
    import sys
    from pathlib import Path

    import ray

    monkeypatch.setitem(
        sys.modules, "sllm.store_manager", SimpleNamespace(StoreManager=object)
    )
    spec = importlib.util.spec_from_file_location(
        "controller_shutdown_probe",
        Path(__file__).parents[2] / "sllm/controller.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller = module.SllmController.__new__(module.SllmController)
    controller.running = True
    controller.running_lock = asyncio.Lock()
    controller.metadata_lock = asyncio.Lock()
    failure = RuntimeError("engine died") if shutdown_fails else None
    router = SimpleNamespace(shutdown=method(AsyncMock(side_effect=failure)))
    scheduler_shutdown = AsyncMock()
    controller.scheduler = SimpleNamespace(shutdown=method(scheduler_shutdown))
    controller.store_manager = object()
    controller.ft_job_store = object()
    controller.request_routers = {"test": router}
    controller.registered_models = {"test": {}}
    controller.ft_processor = asyncio.create_task(asyncio.sleep(60))
    killed = []
    monkeypatch.setattr(ray, "kill", lambda actor, **kw: killed.append(actor))
    if shutdown_fails:
        with pytest.raises(RuntimeError, match="Model cleanup failed"):
            await controller.shutdown()
    else:
        await controller.shutdown()
    assert killed == [
        router,
        controller.scheduler,
        controller.store_manager,
        controller.ft_job_store,
    ]
    scheduler_shutdown.assert_awaited_once()
    assert controller.ft_processor.done()
    assert not controller.request_routers


@pytest.mark.asyncio
async def test_failed_startup_releases_even_when_shutdown_raises(monkeypatch):
    import ray

    import sllm.routers.roundrobin_router as module
    from sllm.utils import InstanceHandle

    router = RoundRobinRouter(
        "test", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    actor = SimpleNamespace(
        init_backend=method(AsyncMock(side_effect=RuntimeError("init failed"))),
        shutdown=method(AsyncMock(side_effect=RuntimeError("close failed"))),
    )
    monkeypatch.setattr(
        module,
        "start_instance",
        SimpleNamespace(
            options=lambda **kwargs: SimpleNamespace(
                remote=AsyncMock(return_value=actor)
            )
        ),
    )
    router.starting_inference_instances["i"] = InstanceHandle("i", 1, 1)
    deallocate = AsyncMock()
    router.model_loading_scheduler = SimpleNamespace(
        allocate_resource=method(AsyncMock(return_value="0")),
        deallocate_resource=method(deallocate),
    )
    killed = []
    monkeypatch.setattr(ray, "kill", lambda actor, **kw: killed.append(actor))
    with pytest.raises(RuntimeError, match="close failed"):
        await router._start_instance("i")
    assert killed == [actor]
    deallocate.assert_awaited_once()
    assert not router.starting_inference_instances


@pytest.mark.asyncio
async def test_cancel_after_allocation_before_instance_lookup_returns_slot():
    from sllm.utils import InstanceHandle

    router = RoundRobinRouter(
        "test", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    router.running = True
    handle = InstanceHandle("i", 1, 1, ready=True)
    router.ready_inference_instances["i"] = handle
    request = asyncio.create_task(router.inference({}, "generate"))
    future = await router.request_queue.get()
    await handle.add_requests(1)
    router.allocated_instances[future] = handle
    async with router.instance_management_lock:
        future.set_result("i")
        await asyncio.sleep(0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    assert handle.concurrency == 0
    assert router.request_count == 0
    assert not router.allocated_instances


@pytest.mark.asyncio
async def test_router_close_waits_for_all_instances_despite_one_failure(
    monkeypatch,
):
    import ray

    from sllm.utils import InstanceHandle

    router = RoundRobinRouter(
        "test", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    router.running = True
    release = asyncio.Event()
    started = asyncio.Event()

    async def delayed():
        started.set()
        await release.wait()

    first = SimpleNamespace(
        shutdown=method(AsyncMock(side_effect=RuntimeError("failed")))
    )
    second = SimpleNamespace(shutdown=method(delayed))
    for name, actor in [("a", first), ("b", second)]:
        router.ready_inference_instances[name] = InstanceHandle(
            name, 1, 1, backend_instance=actor, ready=True
        )
    deallocate = AsyncMock()
    router.model_loading_scheduler = SimpleNamespace(
        deallocate_resource=method(deallocate)
    )
    killed = []
    monkeypatch.setattr(ray, "kill", lambda actor, **kw: killed.append(actor))
    close = asyncio.create_task(router.shutdown())
    await started.wait()
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    with pytest.raises(RuntimeError, match="Instance cleanup failed"):
        await close
    assert len(killed) == 2
    assert deallocate.await_count == 2
