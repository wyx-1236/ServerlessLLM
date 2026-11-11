# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from sllm.inference_instance import start_instance
from sllm.logger import init_logger
from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.utils import InstanceHandle, InstanceStatus

logger = init_logger(__name__)


@dataclass
class _MigrationRequest:
    request_id: str
    dispatched: bool = False
    epoch: Optional[str] = None
    resume: bool = False
    completed: bool = False
    cancelled_settled: bool = False
    cancel_requested: bool = False
    route: asyncio.Event = field(default_factory=asyncio.Event)
    dispatch: asyncio.Event = field(default_factory=asyncio.Event)
    operation: Optional[asyncio.Future] = None
    rpc_instance: Optional[InstanceHandle] = None
    rpc_epoch: Optional[str] = None
    previous_operations: list = field(default_factory=list)


class MigrationRouter(RoundRobinRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
        enable_lora: bool = False,
        lora_adapters: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            model_name,
            resource_requirements,
            backend,
            backend_config,
            router_config,
            enable_lora,
            lora_adapters,
        )
        self.migration_record = {}
        self.migration_delta = self.router_config.get("migration_delta", 20)
        self.migration_lock = asyncio.Lock()
        self.migration_tasks = set()
        self.migration_requests = {}
        self.migration_barriers = {}
        self.migration_epochs = {}
        self.migration_forget = {}

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}
            async with self.request_count_lock:
                self.request_count += 1
            allocation = self.loop.create_future()
            if self.backend == "vllm" and action == "generate":
                request_data = dict(request_data)
                request_data.setdefault("request_id", uuid.uuid4().hex)
                self.migration_requests[allocation] = _MigrationRequest(
                    request_data["request_id"]
                )
            self.pending_allocations.add(allocation)
            self.request_queue.put_nowait(allocation)
        instance = None
        try:
            async with self.idle_time_lock:
                self.idle_time = 0
            instance_id = await allocation
            if instance_id is None:
                return {"error": "Instance cancelled"}
            if self.backend == "vllm" and action == "generate":
                return await self._vllm_inference(allocation, request_data)
            async with self.instance_management_lock:
                instance = self.ready_inference_instances.get(instance_id)
            if instance is None:
                return {"error": "Instance not found"}
            if action == "encode":
                return await instance.backend_instance.encode.remote(
                    request_data=request_data
                )
            if action != "generate":
                return {"error": "Invalid action"}
            result = await instance.backend_instance.generate.remote(
                request_data=request_data
            )
            # Transformers retains its existing continuation envelope.
            if self.backend == "transformers" and "preempted" in result:
                async with self.instance_management_lock:
                    target_id = self.migration_record.get(instance_id)
                    target = self.ready_inference_instances.get(target_id)
                if target is None:
                    return {"error": "No target instance found"}
                resumed_data = dict(request_data)
                # The Transformers backend's omitted max_tokens means a total
                # budget of 10, not a fresh budget after migration.
                resumed_data["max_tokens"] = (
                    request_data.get("max_tokens", 10)
                    - result["completed_tokens"]
                )
                result = await target.backend_instance.resume_generate.remote(
                    request_data=resumed_data,
                    current_output=result["current_output"],
                )
            return result
        except asyncio.CancelledError:
            if allocation in self.migration_requests:
                try:
                    await self._settle_owned(
                        self._cancel_vllm_request(allocation)
                    )
                except Exception:
                    logger.exception("Failed to cancel a migrating request")
            raise
        finally:
            _, cancelled = await self._settle_owned(
                self._finalize_inference(allocation, instance)
            )
            if cancelled:
                raise asyncio.CancelledError

    async def _finalize_inference(self, allocation, instance):
        # Keep ownership until both the instance slot and request count return,
        # including cancellation while either accounting lock is contended.
        try:
            async with self.instance_management_lock:
                self.pending_allocations.discard(allocation)
                state = self.migration_requests.pop(allocation, None)
                if state is not None:
                    state.dispatch.set()
                reserved = self.allocated_instances.pop(allocation, instance)
                if reserved is not None:
                    await reserved.add_requests(-1)
        finally:
            async with self.request_count_lock:
                self.request_count -= 1
        if state is not None and (state.completed or state.cancelled_settled):
            try:
                async with self.instance_management_lock:
                    if state.rpc_instance.migrating:
                        if (
                            state.rpc_instance.instance_id
                            not in self.migration_barriers
                        ):
                            # Committed source retirement owns cleanup even
                            # when its caller finalizes after the handoff.
                            return
                        self.migration_forget.setdefault(
                            state.rpc_instance.instance_id, set()
                        ).add(state.request_id)
                        return
                    operation = state.rpc_instance.backend_instance.forget_migration_request.remote(
                        state.request_id
                    )
                await operation
            except Exception:
                logger.exception("Failed to forget a completed request")

    async def _vllm_inference(self, allocation, request_data):
        state = self.migration_requests[allocation]
        while True:
            async with self.instance_management_lock:
                instance = self.allocated_instances.get(allocation)
                if instance is None or (
                    self.ready_inference_instances.get(instance.instance_id)
                    is not instance
                ):
                    return {"error": "Instance not found"}
                barrier = self.migration_barriers.get(instance.instance_id)
                if barrier is None:
                    if not self.running:
                        return {"error": "Instance stopped"}
                    state.route.clear()
                    state.rpc_instance = instance
                    state.rpc_epoch = state.epoch if state.resume else None
                    if state.resume:
                        operation = (
                            instance.backend_instance.resume_migration.remote(
                                state.epoch, state.request_id
                            )
                        )
                    else:
                        operation = instance.backend_instance.generate.remote(
                            request_data=request_data, migration_tracked=True
                        )
                    state.operation = asyncio.ensure_future(operation)
                    state.dispatched = True
                    state.dispatch.set()
            if barrier is not None:
                await barrier.wait()
                continue
            notification = asyncio.create_task(state.route.wait())
            try:
                await asyncio.wait(
                    (state.operation, notification),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                async with self.instance_management_lock:
                    transferred = (
                        self.allocated_instances.get(allocation)
                        is not state.rpc_instance
                    )
                if transferred:
                    # A source actor failure after publication must not
                    # discard the target's already prepared request.
                    await state.route.wait()
                elif state.operation.done():
                    result = state.operation.result()
                    if not (
                        isinstance(result, dict) and "_sllm_migration" in result
                    ):
                        state.completed = True
                        return result
                    envelope = result["_sllm_migration"]
                    if envelope.get("request_id") != state.request_id:
                        raise RuntimeError("Mismatched migration request ID")
                    # The commit RPC can finish before its caller publishes
                    # the wakeup. Never leak the internal envelope to HTTP.
                    await state.route.wait()
            finally:
                notification.cancel()
                await asyncio.gather(notification, return_exceptions=True)
            # Do not cancel a local Ray Future and mistake that for remote
            # completion. The handoff owner joins it after source retirement.
            state.previous_operations.append(state.operation)

    async def _cancel_vllm_request(self, allocation):
        async with self.instance_management_lock:
            state = self.migration_requests.get(allocation)
            if state is None:
                return
            state.cancel_requested = True
            owner = self.allocated_instances.get(allocation)
            instances = {}
            if state.rpc_instance is not None:
                instances[state.rpc_instance.instance_id] = (
                    state.rpc_instance,
                    self.migration_epochs.get(
                        state.rpc_instance.instance_id, state.rpc_epoch
                    ),
                )
            if owner is not None:
                instances[owner.instance_id] = (
                    owner,
                    self.migration_epochs.get(owner.instance_id, state.epoch),
                )
        try:
            outcomes = await asyncio.gather(
                *(
                    item.backend_instance.cancel_migration_request.remote(
                        epoch, state.request_id
                    )
                    for item, epoch in instances.values()
                ),
                return_exceptions=True,
            )
            failures = [
                value for value in outcomes if isinstance(value, BaseException)
            ]
            if failures:
                raise RuntimeError("Request cancellation failed") from failures[
                    0
                ]
        finally:
            if state.operation is not None:
                await asyncio.gather(state.operation, return_exceptions=True)
                state.cancelled_settled = True
            state.dispatch.set()

    @staticmethod
    async def _settle_owned(operation):
        """Observe ownership-changing RPCs even if the caller is cancelled."""
        task = asyncio.ensure_future(operation)
        cancelled = False
        while True:
            try:
                return await asyncio.shield(task), cancelled
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True

    async def _cleanup_migration_instance(self, instance_id, instance):
        try:
            if instance.backend_instance is not None:
                await instance.backend_instance.shutdown.remote()
        finally:
            await self._release_instance(instance_id, instance)

    async def execute_migration_plan(self, migration_plan):
        async with self.running_lock:
            if not self.running:
                return None
            task = asyncio.create_task(self._execute_migration(migration_plan))
            self.migration_tasks.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            await self._settle_owned(task)
            raise
        finally:
            self.migration_tasks.discard(task)

    async def _execute_migration(self, migration_plan):
        # A second plan must recheck the source after the first one commits.
        async with self.migration_lock:
            source_id = migration_plan.source_instance.instance_id
            async with self.instance_management_lock:
                source = self.ready_inference_instances.get(source_id)
                if source is None or not self.running:
                    return None
            target_id = self._new_instance_id()
            node_id = migration_plan.target_node_id
            target = InstanceHandle(
                instance_id=target_id,
                max_queue_length=source.max_queue_length,
                num_gpu=self.resource_requirements["num_gpus"],
                node_id=node_id,
                lock=asyncio.Lock(),
            )
            reserved = False
            committed = False
            frozen = False
            epoch = uuid.uuid4().hex
            barrier = None
            try:
                reserved, cancelled = await self._settle_owned(
                    self.model_loading_scheduler.mark_resource.remote(
                        self.model_name,
                        target_id,
                        node_id,
                        self.resource_requirements,
                    )
                )
                if cancelled:
                    raise asyncio.CancelledError
                if not reserved:
                    return None
                startup_config = {
                    "num_cpus": self.resource_requirements["num_cpus"],
                    "num_gpus": self.resource_requirements["num_gpus"],
                    "resources": {
                        "worker_node": 0.1,
                        f"worker_id_{node_id}": 0.1,
                    },
                }
                if self.backend == "vllm":
                    startup_config["max_concurrency"] = (
                        target.max_queue_length + 4
                    )
                target.backend_instance, cancelled = await self._settle_owned(
                    start_instance.options(
                        resources=startup_config["resources"]
                    ).remote(
                        target_id,
                        self.backend,
                        self.model_name,
                        self.backend_config,
                        startup_config,
                    )
                )
                if cancelled:
                    raise asyncio.CancelledError
                await target.backend_instance.init_backend.remote()
                n_previous_tokens = 0
                migration_iter = 0
                while True:
                    async with self.instance_management_lock:
                        if (
                            self.ready_inference_instances.get(source_id)
                            is not source
                        ):
                            return None
                    current_tokens = await source.backend_instance.get_current_tokens.remote()
                    if self.backend_config.get("trace_debug", False):
                        logger.info(
                            "Migration token snapshot: %s",
                            {
                                "source_instance_id": source_id,
                                "target_instance_id": target_id,
                                "iteration": migration_iter,
                                "tokens": current_tokens,
                            },
                        )
                    if not current_tokens:
                        break
                    n_current_tokens = len(current_tokens[0])
                    n_delta_tokens = n_current_tokens - n_previous_tokens
                    n_previous_tokens = n_current_tokens
                    # Preserve the original algorithm: do not replay the
                    # final delta.
                    if n_delta_tokens <= self.migration_delta:
                        break
                    if self.backend_config.get("trace_debug", False):
                        logger.info(
                            "Migration replay input: %s",
                            {
                                "source_instance_id": source_id,
                                "target_instance_id": target_id,
                                "iteration": migration_iter,
                                "tokens": current_tokens,
                            },
                        )
                    replay = (
                        await target.backend_instance.resume_kv_cache.remote(
                            current_tokens
                        )
                    )
                    if isinstance(replay, dict) and "error" in replay:
                        raise RuntimeError(
                            f"Migration replay failed: {replay['error']}"
                        )
                    migration_iter += 1
                if self.backend == "vllm":
                    async with self.instance_management_lock:
                        if (
                            not self.running
                            or self.ready_inference_instances.get(source_id)
                            is not source
                        ):
                            return None
                        source.migrating = True
                        barrier = asyncio.Event()
                        self.migration_barriers[source_id] = barrier
                        self.migration_epochs[source_id] = epoch
                        request_ids = [
                            state.request_id
                            for allocation, state in self.migration_requests.items()
                            if state.dispatched
                            and self.allocated_instances.get(allocation)
                            is source
                        ]
                    # freeze may acquire ownership before its RPC reports an
                    # error, so every precommit exit must attempt rollback.
                    frozen = True
                    snapshot, cancelled = await self._settle_owned(
                        source.backend_instance.freeze_migration.remote(
                            epoch, request_ids=request_ids
                        )
                    )
                    if cancelled:
                        raise asyncio.CancelledError
                    if (
                        snapshot.get("schema_version") != 1
                        or snapshot.get("epoch") != epoch
                    ):
                        raise RuntimeError("Invalid migration snapshot")
                    envelopes = snapshot["requests"]
                    prepared, cancelled = await self._settle_owned(
                        target.backend_instance.prepare_migration.remote(
                            epoch, envelopes
                        )
                    )
                    if cancelled:
                        raise asyncio.CancelledError
                    snapshot_ids = {item["request_id"] for item in envelopes}
                    if set(prepared) != snapshot_ids:
                        raise RuntimeError("Incomplete migration preparation")
                    transferred = await self._publish_vllm_handoff(
                        source, target, snapshot_ids, epoch, barrier
                    )
                    committed = True
                    _, cancelled = await self._settle_owned(
                        self._finish_vllm_handoff(
                            source,
                            target,
                            snapshot_ids,
                            transferred,
                            epoch,
                            barrier,
                        )
                    )
                    if cancelled:
                        raise asyncio.CancelledError
                    return target_id
                async with self.instance_management_lock:
                    if (
                        not self.running
                        or self.ready_inference_instances.get(source_id)
                        is not source
                    ):
                        return None
                    self.ready_inference_instances.pop(source_id)
                    source.status = False
                    target.ready = True
                    self.ready_inference_instances[target_id] = target
                    self.migration_record[source_id] = target_id
                    committed = True
                # Ownership has transferred. A retirement failure must not roll
                # back the published target or retain the source reservation.
                _, cancelled = await self._settle_owned(
                    self._cleanup_migration_instance(source_id, source)
                )
                if cancelled:
                    raise asyncio.CancelledError
                return target_id
            finally:
                if reserved and not committed:
                    try:
                        if frozen:
                            await self._settle_owned(
                                self._rollback_vllm_migration(source, epoch)
                            )
                    finally:
                        try:
                            await self._settle_owned(
                                self._cleanup_migration_instance(
                                    target_id, target
                                )
                            )
                        finally:
                            await self._settle_owned(
                                self._release_vllm_barrier(source, barrier)
                            )

    async def _publish_vllm_handoff(
        self, source, target, snapshot_ids, epoch, barrier
    ):
        # Acquire every contended lock before the first ledger mutation. A
        # cancellation while waiting for a slot lock must leave source owner.
        async with self.instance_management_lock, source.lock, target.lock:
            if not self.running:
                raise asyncio.CancelledError
            if (
                self.ready_inference_instances.get(source.instance_id)
                is not source
            ):
                raise RuntimeError("Migration source is no longer ready")
            transferred = []
            for allocation, state in self.migration_requests.items():
                if (
                    self.allocated_instances.get(allocation) is source
                    and (
                        not state.dispatched or state.request_id in snapshot_ids
                    )
                    and not state.cancel_requested
                ):
                    transferred.append(state)
                    self.allocated_instances[allocation] = target
                    state.resume = state.dispatched
                    state.epoch = epoch if state.resume else None
                    state.dispatched = False
                    state.dispatch.clear()
            source.concurrency -= len(transferred)
            target.concurrency += len(transferred)
            self.ready_inference_instances.pop(source.instance_id)
            target.ready = True
            target.migrating = True
            self.ready_inference_instances[target.instance_id] = target
            self.migration_barriers[target.instance_id] = barrier
            self.migration_record[source.instance_id] = target.instance_id
            return transferred

    async def _finish_vllm_handoff(
        self, source, target, snapshot_ids, transferred, epoch, barrier
    ):
        failure = None
        # This entire post-publication operation is shielded. Once the owner
        # changes, cancellation cannot interrupt activation or source cleanup.
        transferred_ids = {state.request_id for state in transferred}
        for request_id in snapshot_ids - transferred_ids:
            try:
                await target.backend_instance.cancel_migration_request.remote(
                    epoch, request_id
                )
            except BaseException as error:
                failure = error
        try:
            await source.backend_instance.commit_migration.remote(epoch)
        except BaseException as error:
            failure = error
        finally:
            # Even a lost commit response cannot restore source ownership or
            # strand the original request waiting on the retired actor.
            async with self.instance_management_lock:
                self.migration_barriers.pop(source.instance_id, None)
                self.migration_barriers.pop(target.instance_id, None)
                target.migrating = False
                barrier.set()
                for state in transferred:
                    state.route.set()
        try:
            # The next migration must see each resume RPC as dispatched or
            # cancelled; otherwise a prepared request could be missed.
            await asyncio.gather(
                *(state.dispatch.wait() for state in transferred)
            )
            async with self.instance_management_lock:
                completed_source_rpcs = [
                    state.operation
                    for allocation, state in self.migration_requests.items()
                    if self.allocated_instances.get(allocation) is source
                    and state.operation is not None
                ]
            # A naturally completed request can be absent from freeze while
            # its Ray response is still in flight. Observe that response before
            # killing the source actor; its original caller owns finalization.
            await asyncio.gather(*completed_source_rpcs, return_exceptions=True)
        finally:
            try:
                await self._cleanup_migration_instance(
                    source.instance_id, source
                )
            finally:
                await asyncio.gather(
                    *(
                        operation
                        for state in transferred
                        for operation in state.previous_operations
                    ),
                    return_exceptions=True,
                )
                async with self.instance_management_lock:
                    # Retired actors need no per-request tombstone cleanup.
                    self.migration_forget.pop(source.instance_id, None)
                    self.migration_epochs.pop(source.instance_id, None)
        if failure is not None:
            raise failure

    async def _release_vllm_barrier(self, source, barrier):
        async with self.instance_management_lock:
            source.migrating = False
            self.migration_epochs.pop(source.instance_id, None)
            self.migration_barriers.pop(source.instance_id, None)
            forgotten = self.migration_forget.pop(source.instance_id, ())
            if barrier is not None:
                barrier.set()
        for request_id in forgotten:
            await source.backend_instance.forget_migration_request.remote(
                request_id
            )

    async def _rollback_vllm_migration(self, source, epoch):
        if not self.running:
            async with self.instance_management_lock:
                request_ids = [
                    state.request_id
                    for allocation, state in self.migration_requests.items()
                    if self.allocated_instances.get(allocation) is source
                ]
            await asyncio.gather(
                *(
                    source.backend_instance.cancel_migration_request.remote(
                        epoch, request_id
                    )
                    for request_id in request_ids
                )
            )
        await source.backend_instance.rollback_migration.remote(epoch)

    async def _close(self):
        migrations = list(self.migration_tasks)
        for task in migrations:
            task.cancel()
        outcomes = await asyncio.gather(*migrations, return_exceptions=True)
        deleted = await super()._close()
        failures = [item for item in outcomes if isinstance(item, Exception)]
        if failures:
            raise RuntimeError("Migration cleanup failed") from failures[0]
        return deleted

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        async with self.instance_management_lock:
            instance = self.ready_inference_instances.get(instance_id)
        if instance is None:
            return None
        instance_status = await instance.get_status()
        instance_status.model_name = self.model_name
        current_tokens = (
            await instance.backend_instance.get_current_tokens.remote()
        )
        instance_status.num_current_tokens = (
            len(current_tokens[0]) if current_tokens else 0
        )
        return instance_status

    async def fine_tuning(self, request_data: dict):
        raise NotImplementedError(
            "Fine-tuning is only applicable on roundrobin router"
        )
