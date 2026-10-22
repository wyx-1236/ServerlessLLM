"""CPU regressions for original migration semantics and lifecycle ownership."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import ray
from test_legacy_contract import legacy as legacy

import sllm.routers.migration_router as router_module
import sllm.schedulers.storage_aware_scheduler as scheduler_module
from sllm.utils import InstanceHandle, InstanceStatus


class Ref:
    """Minimal eager RPC result supporting both await and ray.get."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def result():
            return self.value

        return result().__await__()


def rpc(fn):
    return SimpleNamespace(remote=lambda *a, **kw: Ref(fn(*a, **kw)))


def new_router():
    return router_module.MigrationRouter(
        "model",
        {"num_cpus": 1, "num_gpus": 1},
        "vllm",
        {},
        {"migration_delta": 2},
    )


@pytest_asyncio.fixture
async def chain(monkeypatch):
    router = new_router()
    events = []
    snapshots = [[[10, 11, 20, 21]], [[10, 11, 20, 21, 22]]]
    reservations = {}

    def event(name, value=None):
        events.append((name, copy.deepcopy(value)))

    def mark(model, instance, node, resources=None):
        assert resources == router.resource_requirements
        reservations[instance] = node
        event("mark", instance)
        return True

    def deallocate(model, instance, resources):
        reservations.pop(instance, None)
        event("deallocate", instance)

    def snapshot():
        value = snapshots.pop(0)
        event("snapshot", value)
        return value

    source = SimpleNamespace(
        get_current_tokens=rpc(snapshot),
        shutdown=rpc(
            lambda: event(
                "source_shutdown", list(router.ready_inference_instances)
            )
        ),
    )
    target = SimpleNamespace(
        init_backend=rpc(lambda: event("init")),
        resume_kv_cache=rpc(lambda tokens: event("replay", tokens)),
        shutdown=rpc(lambda: event("target_shutdown")),
    )
    handle = InstanceHandle(
        "source",
        10,
        1,
        node_id="0",
        backend_instance=source,
        ready=True,
        lock=asyncio.Lock(),
    )
    router.ready_inference_instances = {"source": handle}
    router.running = True
    router.model_loading_scheduler = SimpleNamespace(
        mark_resource=rpc(mark),
        deallocate_resource=rpc(deallocate),
    )
    reservations["source"] = "0"
    monkeypatch.setattr(router, "_new_instance_id", lambda: "target")
    monkeypatch.setattr(
        router_module,
        "start_instance",
        SimpleNamespace(
            options=lambda **kw: SimpleNamespace(
                remote=rpc(lambda *a: (event("start"), target)[1]).remote,
            ),
        ),
    )

    def forbidden(*a, **kw):
        raise AssertionError(
            "Migration must await RPCs and use the returned actor"
        )

    monkeypatch.setattr(ray, "get_actor", forbidden)
    monkeypatch.setattr(ray, "get", forbidden)
    monkeypatch.setattr(
        ray,
        "kill",
        lambda actor, **kw: event(
            "kill_source" if actor is source else "kill_target",
        ),
    )
    plan = scheduler_module.MigrationPlan(
        "1",
        InstanceStatus("source", "0", 1, 1, model_name="model"),
    )
    return SimpleNamespace(
        router=router,
        source=source,
        target=target,
        plan=plan,
        events=events,
        snapshots=snapshots,
        reservations=reservations,
    )


@pytest.mark.asyncio
async def test_router_constructor_matches_controller_and_uses_parent_pool():
    router = router_module.MigrationRouter(
        "model", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}, False, {}
    )
    assert await router.get_instance_status("source") is None
    assert not hasattr(router, "ready_instances")


@pytest.mark.asyncio
async def test_characterization_actual_backend_snapshot_replayed_as_one_token(
    legacy,
):
    source = legacy()
    assert await source._admit("active", [10, 11]) is None
    source.active_requests["active"]["output"] = SimpleNamespace(
        prompt_token_ids=[10, 11],
        outputs=[SimpleNamespace(token_ids=[20, 21])],
    )
    snapshot = await source.get_current_tokens()
    assert snapshot == [[10, 11, 20, 21]]
    target = legacy()
    target.engine.release.set()
    await target.resume_kv_cache(snapshot)
    assert target.engine.calls[0][:2] == (
        {"prompt_token_ids": [10, 11, 20, 21]},
        {"max_tokens": 1},
    )
    assert not hasattr(target, "resume_generate")
    await source._finish("active")


@pytest.mark.asyncio
async def test_characterization_snapshot_replay_then_switch_then_source_cleanup(
    chain,
):
    assert await chain.router.execute_migration_plan(chain.plan) == "target"
    assert chain.events == [
        ("mark", "target"),
        ("start", None),
        ("init", None),
        ("snapshot", [[10, 11, 20, 21]]),
        ("replay", [[10, 11, 20, 21]]),
        ("snapshot", [[10, 11, 20, 21, 22]]),
        ("source_shutdown", ["target"]),
        ("kill_source", None),
        ("deallocate", "source"),
    ]
    assert chain.router.migration_record == {"source": "target"}
    assert chain.reservations == {"target": "1"}
    # The final delta is deliberately not replayed by the original algorithm.


@pytest.mark.asyncio
async def test_characterization_first_snapshot_within_delta_skips_all_replay(
    chain,
):
    chain.snapshots[:] = [[[10, 11]]]
    assert await chain.router.execute_migration_plan(chain.plan) == "target"
    assert not any(name == "replay" for name, _ in chain.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["init_backend", "resume_kv_cache"])
async def test_target_failure_releases_actor_and_reservation(
    chain,
    failure_at,
):
    def fail(*args):
        raise RuntimeError("target failure")

    setattr(chain.target, failure_at, rpc(fail))
    with pytest.raises(RuntimeError, match="target failure"):
        await chain.router.execute_migration_plan(chain.plan)
    assert list(chain.router.ready_inference_instances) == ["source"]
    assert chain.reservations == {"source": "0"}
    assert ("target_shutdown", None) in chain.events
    assert ("kill_target", None) in chain.events


@pytest.mark.asyncio
async def test_vanished_source_releases_target(chain):
    chain.router.ready_inference_instances.clear()
    assert await chain.router.execute_migration_plan(chain.plan) is None
    assert "target" not in chain.reservations


@pytest.mark.asyncio
async def test_source_close_failure_keeps_target_and_releases_source(chain):
    def fail():
        raise RuntimeError("source close failed")

    chain.source.shutdown = rpc(fail)
    with pytest.raises(RuntimeError, match="source close failed"):
        await chain.router.execute_migration_plan(chain.plan)
    assert list(chain.router.ready_inference_instances) == ["target"]
    assert chain.router.migration_record == {"source": "target"}
    assert set(chain.reservations) == {"target"}
    assert ("kill_source", None) in chain.events


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_raises", [False, True])
async def test_abort_passthrough_and_exception_release_request_count(
    chain,
    backend_raises,
):
    router = chain.router
    router.running = True
    result = {"choices": [{"finish_reason": "abort"}]}

    def generate(**kwargs):
        if backend_raises:
            raise RuntimeError("actor failed")
        return result

    chain.source.generate = rpc(generate)
    handle = router.ready_inference_instances["source"]
    handle.concurrency = 1
    request = asyncio.create_task(
        router.inference({"max_tokens": 4}, "generate")
    )
    future = await router.request_queue.get()
    future.set_result("source")
    if backend_raises:
        with pytest.raises(RuntimeError, match="actor failed"):
            await request
        assert router.request_count == handle.concurrency == 0
    else:
        assert await request is result
        assert router.request_count == handle.concurrency == 0
    assert not router.migration_record


@pytest.mark.asyncio
async def test_mark_resource_refuses_overreservation():
    scheduler = scheduler_module.StorageAwareScheduler({})
    scheduler.worker_nodes = {"1": {"free_gpu": 1, "total_gpu": 1}}
    assert await scheduler.mark_resource("model", "a", "1")
    assert not await scheduler.mark_resource("model", "b", "1")
    assert scheduler.model_instance == {"model": {"a": "1"}}
    assert scheduler.worker_nodes["1"]["free_gpu"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("migration_fails", [False, True])
async def test_storage_loop_owns_allocations_and_waits_after_failed_migration(
    monkeypatch,
    migration_fails,
):
    from test_storage_aware_lifecycle import cancel, enqueue, run_iteration

    scheduler = scheduler_module.StorageAwareScheduler({})
    scheduler.worker_nodes = {
        "0": {"free_gpu": 0 if migration_fails else 1},
        "1": {"free_gpu": 1},
    }
    migrations = None
    if migration_fails:
        migrations = scheduler_module.MigrationPlans()
        migrations.append(
            scheduler_module.MigrationPlan(
                "1",
                InstanceStatus("old", "0", 1, 1, model_name="old_model"),
            ),
            1,
        )
    monkeypatch.setattr(
        scheduler_module.ray,
        "get_actor",
        lambda *a, **kw: SimpleNamespace(
            execute_migration_plan=SimpleNamespace(
                remote=AsyncMock(return_value=None)
            )
        ),
    )
    request = await enqueue(scheduler)
    try:
        await run_iteration(
            scheduler, scheduler_module.AllocationPlan("0", 1, migrations)
        )
        if migration_fails:
            assert not request.done()
            assert scheduler.model_instance == {}
            assert scheduler.worker_nodes["0"]["free_gpu"] == 0
        else:
            assert await asyncio.wait_for(request, 1) == "0"
            assert scheduler.model_instance == {"new": {"new_i": "0"}}
            await scheduler.deallocate_resource("new", "new_i", {"num_gpus": 1})
            assert scheduler.worker_nodes["0"]["free_gpu"] == 1
        assert scheduler.worker_nodes["1"]["free_gpu"] == 1
    finally:
        await cancel(request)
