"""CPU checks for transparent vLLM request and slot handoff."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

import sllm.routers.migration_router as router_module
from sllm.schedulers.storage_aware_scheduler import MigrationPlan
from sllm.utils import InstanceHandle, InstanceStatus


def rpc(function):
    async def invoke(*args, **kwargs):
        value = function(*args, **kwargs)
        if asyncio.iscoroutine(value):
            return await value
        return value

    return SimpleNamespace(remote=invoke)


async def joined(operation):
    return await asyncio.wait_for(operation, 3)


class Backend:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.requests = {}
        self.cancelled = set()
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.prepared = asyncio.Event()
        self.prepare_release = asyncio.Event()
        self.prepare_release.set()
        self.commit_release = asyncio.Event()
        self.commit_release.set()
        self.freeze_release = asyncio.Event()
        self.freeze_release.set()
        self.frozen = asyncio.Event()
        self.commit_error = False
        self.prepare_error = False
        self.shutdown_error = False
        self.epoch = None
        self.route = asyncio.Event()
        self.cancel_signal = asyncio.Event()
        for name in (
            "generate",
            "get_current_tokens",
            "init_backend",
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
            setattr(self, name, rpc(getattr(self, "_" + name)))

    async def _generate(self, request_data, *, migration_tracked=False):
        assert migration_tracked
        request_id = request_data["request_id"]
        self.requests[request_id] = request_data
        self.started.set()
        return await self._result(request_id)

    async def _result(self, request_id):
        done = asyncio.create_task(self.finish.wait())
        route = asyncio.create_task(self.route.wait())
        cancel = asyncio.create_task(self.cancel_signal.wait())
        try:
            await asyncio.wait(
                (done, route, cancel), return_when=asyncio.FIRST_COMPLETED
            )
            if cancel.done():
                raise asyncio.CancelledError
            if done.done():
                return {"id": request_id, "text": "completed"}
            return {
                "_sllm_migration": {
                    "epoch": self.epoch,
                    "request_id": request_id,
                }
            }
        finally:
            done.cancel()
            route.cancel()
            cancel.cancel()
            await asyncio.gather(done, route, cancel, return_exceptions=True)

    def _get_current_tokens(self):
        return []

    def _init_backend(self):
        self.events.append(("init", self.name))

    def _resume_kv_cache(self, tokens):
        self.events.append(("replay", self.name))

    def _shutdown(self):
        self.events.append(("shutdown", self.name))
        if self.shutdown_error:
            raise RuntimeError("shutdown failed")

    async def _freeze_migration(self, epoch, request_ids=None):
        self.events.append(("freeze", self.name))
        self.epoch = epoch
        self.expected = request_ids
        self.frozen.set()
        await self.freeze_release.wait()
        return {
            "schema_version": 1,
            "epoch": epoch,
            "requests": [
                {"request_id": value}
                for value in request_ids
                if value not in self.cancelled and not self.finish.is_set()
            ],
        }

    async def _prepare_migration(self, epoch, envelopes):
        self.events.append(("prepare", self.name))
        self.epoch = epoch
        self.requests = {item["request_id"]: item for item in envelopes}
        self.prepared.set()
        await self.prepare_release.wait()
        if self.prepare_error:
            raise RuntimeError("prepare failed")
        return list(self.requests)

    async def _commit_migration(self, epoch):
        self.events.append(("commit", self.name))
        await self.commit_release.wait()
        self.route.set()
        if self.commit_error:
            raise RuntimeError("commit response lost")

    def _rollback_migration(self, epoch):
        self.events.append(("rollback", self.name))
        self.frozen.clear()

    async def _resume_migration(self, epoch, request_id):
        assert request_id not in self.cancelled
        self.events.append(("resume", self.name))
        self.started.set()
        return await self._result(request_id)

    def _cancel_migration_request(self, epoch, request_id):
        self.events.append(("cancel", self.name))
        self.cancelled.add(request_id)
        self.cancel_signal.set()

    def _forget_migration_request(self, request_id):
        self.events.append(("forget", self.name))
        self.requests.pop(request_id, None)


@pytest_asyncio.fixture
async def migration(monkeypatch):
    router = router_module.MigrationRouter(
        "model", {"num_cpus": 1, "num_gpus": 1}, "vllm", {}, {}
    )
    router.running = True
    events = []
    source, target = Backend("source", events), Backend("target", events)
    handle = InstanceHandle(
        "source",
        3,
        1,
        node_id="0",
        backend_instance=source,
        ready=True,
        lock=asyncio.Lock(),
    )
    router.ready_inference_instances["source"] = handle
    reservations = {"source"}

    def mark(model, instance, *args):
        reservations.add(instance)
        return True

    router.model_loading_scheduler = SimpleNamespace(
        mark_resource=rpc(mark),
        deallocate_resource=rpc(
            lambda model, instance, *args: reservations.remove(instance)
        ),
    )
    monkeypatch.setattr(router, "_new_instance_id", lambda: "target")
    monkeypatch.setattr(
        router_module,
        "start_instance",
        SimpleNamespace(
            options=lambda **kwargs: SimpleNamespace(
                remote=rpc(lambda *args: target).remote
            )
        ),
    )
    monkeypatch.setattr("ray.kill", lambda *args, **kwargs: None)
    state = SimpleNamespace(
        router=router,
        source=source,
        target=target,
        handle=handle,
        events=events,
        reservations=reservations,
        plan=MigrationPlan("1", InstanceStatus("source", "0", 1, 0)),
    )
    yield state
    source.finish.set()
    target.finish.set()
    source.freeze_release.set()
    target.prepare_release.set()
    source.commit_release.set()


async def request(state, request_id="request", dispatch=True):
    task = asyncio.create_task(
        state.router.inference({"request_id": request_id}, "generate")
    )
    allocation = await joined(state.router.request_queue.get())
    state.handle.concurrency += 1
    state.router.allocated_instances[allocation] = state.handle
    allocation.set_result("source")
    if dispatch:
        await joined(state.source.started.wait())
    return task, allocation


@pytest.mark.asyncio
async def test_handoff_keeps_original_request_and_moves_slot_before_commit(
    migration,
):
    task, allocation = await request(migration)
    try:
        assert (
            await joined(
                migration.router.execute_migration_plan(migration.plan)
            )
            == "target"
        )
        await joined(migration.target.started.wait())
        owner = migration.router.allocated_instances[allocation]
        assert owner.instance_id == "target"
        assert owner.max_queue_length == 3
        assert owner.concurrency == migration.router.request_count == 1
        assert migration.handle.concurrency == 0
        assert migration.source.expected == ["request"]
        assert not task.done()
        migration.target.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
        assert owner.concurrency == migration.router.request_count == 0
        assert migration.reservations == {"target"}
        assert ("forget", "target") in migration.events
        assert not migration.target.requests
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_prepare_failure_rolls_back_source_before_target_cleanup(
    migration,
):
    task, _ = await request(migration)
    migration.target.prepare_error = True
    try:
        with pytest.raises(RuntimeError, match="prepare failed"):
            await joined(
                migration.router.execute_migration_plan(migration.plan)
            )
        assert migration.events.index(
            ("rollback", "source")
        ) < migration.events.index(("shutdown", "target"))
        assert migration.reservations == {"source"}
        assert not task.done()
        migration.source.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["freeze", "prepare"])
async def test_plan_cancel_waits_for_late_rpc_then_rolls_back(migration, stage):
    task, _ = await request(migration)
    gate = (
        migration.source.freeze_release
        if stage == "freeze"
        else migration.target.prepare_release
    )
    entered = (
        migration.source.frozen
        if stage == "freeze"
        else migration.target.prepared
    )
    gate.clear()
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(entered.wait())
        plan.cancel()
        await asyncio.sleep(0)
        assert not plan.done()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await joined(plan)
        assert migration.events.index(
            ("rollback", "source")
        ) < migration.events.index(("shutdown", "target"))
        assert not migration.handle.migrating
        migration.source.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
    finally:
        gate.set()
        task.cancel()
        plan.cancel()
        await asyncio.gather(task, plan, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_during_prepare_does_not_transfer_or_revive_request(
    migration,
):
    task, _ = await request(migration)
    migration.target.prepare_release.clear()
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.target.prepared.wait())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await joined(task)
        migration.target.prepare_release.set()
        assert await joined(plan) == "target"
        assert "request" in migration.target.cancelled
        # The source freeze may still need its terminal marker until commit;
        # actor retirement clears it without an early forget RPC.
        assert ("forget", "source") not in migration.events
        assert not migration.target.started.is_set()
        assert migration.router.request_count == 0
        assert migration.handle.concurrency == 0
        assert (
            migration.router.ready_inference_instances["target"].concurrency
            == 0
        )
    finally:
        migration.target.prepare_release.set()
        task.cancel()
        plan.cancel()
        await asyncio.gather(task, plan, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["commit", "shutdown"])
async def test_postpublication_failure_keeps_target_owner_and_completes(
    migration, failure
):
    task, allocation = await request(migration)
    if failure == "commit":
        migration.source.commit_error = True
    else:
        migration.source.shutdown_error = True
    try:
        with pytest.raises(
            RuntimeError, match="commit response lost|shutdown failed"
        ):
            await joined(
                migration.router.execute_migration_plan(migration.plan)
            )
        await joined(migration.target.started.wait())
        assert (
            migration.router.allocated_instances[allocation].instance_id
            == "target"
        )
        assert migration.reservations == {"target"}
        assert migration.events.count(("commit", "source")) == 1
        assert ("rollback", "source") not in migration.events
        migration.target.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_during_freeze_is_not_transferred(migration):
    task, _ = await request(migration)
    migration.source.freeze_release.clear()
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.source.frozen.wait())
        migration.source.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
        migration.source.freeze_release.set()
        assert await joined(plan) == "target"
        assert not migration.target.started.is_set()
        assert (
            migration.router.ready_inference_instances["target"].concurrency
            == 0
        )
    finally:
        migration.source.freeze_release.set()
        task.cancel()
        plan.cancel()
        await asyncio.gather(task, plan, return_exceptions=True)


@pytest.mark.asyncio
async def test_allocated_but_not_dispatched_waits_then_generates_on_target(
    migration,
):
    router = migration.router
    migration.source.freeze_release.clear()
    plan = asyncio.create_task(router.execute_migration_plan(migration.plan))
    task = None
    try:
        await joined(migration.source.frozen.wait())
        task, allocation = await request(migration, dispatch=False)
        await asyncio.sleep(0)
        assert not migration.source.started.is_set()
        assert migration.source.expected == []
        migration.source.freeze_release.set()
        assert await joined(plan) == "target"
        await joined(migration.target.started.wait())
        assert router.allocated_instances[allocation].instance_id == "target"
        assert ("resume", "target") not in migration.events
        migration.target.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
    finally:
        migration.source.freeze_release.set()
        plan.cancel()
        if task is not None:
            task.cancel()
        await asyncio.gather(
            plan, *([task] if task else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_frozen_source_rejects_allocation_and_autoscaler_deletion(
    migration,
):
    migration.source.freeze_release.clear()
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.source.frozen.wait())
        assert migration.handle.ready
        assert not await migration.handle.add_requests(1)
        assert not await migration.handle.check_request_queue()
        await joined(migration.router._stop_instance())
        await joined(migration.router._stop_instance("source"))
        assert migration.router.ready_inference_instances == {
            "source": migration.handle
        }
        migration.source.freeze_release.set()
        assert await joined(plan) == "target"
    finally:
        migration.source.freeze_release.set()
        plan.cancel()
        await asyncio.gather(plan, return_exceptions=True)


@pytest.mark.asyncio
async def test_request_can_cross_two_migrations(migration, monkeypatch):
    task, allocation = await request(migration)
    try:
        assert (
            await joined(
                migration.router.execute_migration_plan(migration.plan)
            )
            == "target"
        )
        second = Backend("second", migration.events)
        monkeypatch.setattr(
            migration.router, "_new_instance_id", lambda: "second"
        )
        monkeypatch.setattr(
            router_module,
            "start_instance",
            SimpleNamespace(
                options=lambda **kwargs: SimpleNamespace(
                    remote=rpc(lambda *args: second).remote
                )
            ),
        )
        plan = MigrationPlan("2", InstanceStatus("target", "1", 1, 1))
        assert (
            await joined(migration.router.execute_migration_plan(plan))
            == "second"
        )
        assert (
            migration.router.allocated_instances[allocation].instance_id
            == "second"
        )
        second.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
        assert migration.router.request_count == 0
        assert migration.reservations == {"second"}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_waiting_for_slot_lock_does_not_partially_publish(
    migration,
):
    task, allocation = await request(migration)
    await migration.handle.lock.acquire()
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.target.prepared.wait())
        for _ in range(5):
            await asyncio.sleep(0)
        plan.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        migration.handle.lock.release()
        with pytest.raises(asyncio.CancelledError):
            await joined(plan)
        assert (
            migration.router.allocated_instances[allocation] is migration.handle
        )
        assert migration.handle.concurrency == 1
        migration.source.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
    finally:
        if migration.handle.lock.locked():
            migration.handle.lock.release()
        plan.cancel()
        task.cancel()
        await asyncio.gather(plan, task, return_exceptions=True)


@pytest.mark.asyncio
async def test_source_rpc_failure_after_publication_uses_prepared_target(
    migration,
):
    crash = asyncio.Event()

    async def generate(request_data, *, migration_tracked=False):
        migration.source.started.set()
        await crash.wait()
        raise RuntimeError("source actor lost")

    async def commit(epoch):
        crash.set()
        for _ in range(5):
            await asyncio.sleep(0)
        raise RuntimeError("commit response lost")

    migration.source.generate = rpc(generate)
    migration.source.commit_migration = rpc(commit)
    task, _ = await request(migration)
    try:
        with pytest.raises(RuntimeError, match="commit response lost"):
            await joined(
                migration.router.execute_migration_plan(migration.plan)
            )
        migration.target.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_during_commit_settles_handoff_and_does_not_revive_request(
    migration,
):
    task, allocation = await request(migration)
    migration.source.commit_release.clear()
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.target.prepared.wait())
        while ("commit", "source") not in migration.events:
            await asyncio.sleep(0)
        assert (
            migration.router.allocated_instances[allocation].instance_id
            == "target"
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await joined(task)
        plan.cancel()
        await asyncio.sleep(0)
        assert not plan.done()
        migration.source.commit_release.set()
        with pytest.raises(asyncio.CancelledError):
            await joined(plan)
        assert "request" in migration.target.cancelled
        assert not migration.target.started.is_set()
        assert migration.reservations == {"target"}
        assert migration.router.request_count == 0
    finally:
        migration.source.commit_release.set()
        task.cancel()
        plan.cancel()
        await asyncio.gather(task, plan, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_waits_for_real_rpc_completion_then_forgets(migration):
    task, _ = await request(migration)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await joined(task)
    assert ("cancel", "source") in migration.events
    assert ("forget", "source") in migration.events
    assert not migration.source.requests
    assert migration.router.request_count == migration.handle.concurrency == 0


@pytest.mark.asyncio
async def test_terminal_forget_waits_until_freeze_rollback(migration):
    task, _ = await request(migration)
    migration.source.freeze_release.clear()
    migration.target.prepare_error = True
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.source.frozen.wait())
        migration.source.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
        assert ("forget", "source") not in migration.events
        migration.source.freeze_release.set()
        with pytest.raises(RuntimeError, match="prepare failed"):
            await joined(plan)
        assert migration.events.index(
            ("rollback", "source")
        ) < migration.events.index(("forget", "source"))
        assert not migration.router.migration_forget
        assert not migration.source.requests
    finally:
        migration.source.freeze_release.set()
        plan.cancel()
        task.cancel()
        await asyncio.gather(task, plan, return_exceptions=True)


@pytest.mark.asyncio
async def test_removed_source_cannot_publish_partial_slot_transfer(migration):
    task, allocation = await request(migration)
    migration.target.prepare_release.clear()
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.target.prepared.wait())
        async with migration.router.instance_management_lock:
            migration.router.ready_inference_instances.pop("source")
        migration.target.prepare_release.set()
        with pytest.raises(RuntimeError, match="source is no longer ready"):
            await joined(plan)
        assert (
            migration.router.allocated_instances[allocation] is migration.handle
        )
        assert migration.handle.concurrency == 1
        assert not migration.router.migration_record
        migration.source.finish.set()
        assert await joined(task) == {"id": "request", "text": "completed"}
    finally:
        migration.target.prepare_release.set()
        task.cancel()
        plan.cancel()
        await asyncio.gather(task, plan, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_in_flight_at_publication_tombstones_target(migration):
    task, allocation = await request(migration)
    migration.target.prepare_release.clear()
    entered, release = asyncio.Event(), asyncio.Event()
    cancel = migration.source.cancel_migration_request.remote

    async def delayed_cancel(epoch, request_id):
        entered.set()
        await release.wait()
        await cancel(epoch, request_id)

    migration.source.cancel_migration_request = rpc(delayed_cancel)
    plan = asyncio.create_task(
        migration.router.execute_migration_plan(migration.plan)
    )
    try:
        await joined(migration.target.prepared.wait())
        task.cancel()
        await joined(entered.wait())
        migration.target.prepare_release.set()
        while ("commit", "source") not in migration.events:
            await asyncio.sleep(0)
        assert "request" in migration.target.cancelled
        assert (
            migration.router.allocated_instances[allocation] is migration.handle
        )
        release.set()
        assert await joined(plan) == "target"
        with pytest.raises(asyncio.CancelledError):
            await joined(task)
        assert not migration.router.migration_forget
        assert (
            migration.router.request_count == migration.handle.concurrency == 0
        )
    finally:
        release.set()
        migration.target.prepare_release.set()
        task.cancel()
        plan.cancel()
        await asyncio.gather(task, plan, return_exceptions=True)
