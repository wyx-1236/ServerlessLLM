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
from typing import Dict, Optional

from sllm.inference_instance import start_instance
from sllm.logger import init_logger
from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.utils import InstanceHandle, InstanceStatus

logger = init_logger(__name__)


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

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}
            async with self.request_count_lock:
                self.request_count += 1
            allocation = self.loop.create_future()
            self.pending_allocations.add(allocation)
            self.request_queue.put_nowait(allocation)
        instance = None
        try:
            async with self.idle_time_lock:
                self.idle_time = 0
            instance_id = await allocation
            if instance_id is None:
                return {"error": "Instance cancelled"}
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
            # Keep the inherited Transformers continuation branch. vLLM's
            # abort/cancel response is terminal and has no resume_generate API.
            if "preempted" in result:
                async with self.instance_management_lock:
                    target_id = self.migration_record.get(instance_id)
                    target = self.ready_inference_instances.get(target_id)
                if target is None:
                    return {"error": "No target instance found"}
                resumed_data = dict(request_data)
                if "max_tokens" in resumed_data:
                    resumed_data["max_tokens"] -= result["completed_tokens"]
                result = await target.backend_instance.resume_generate.remote(
                    request_data=resumed_data,
                    current_output=result["current_output"],
                )
            return result
        finally:
            _, cancelled = await self._settle_owned(
                self._finalize_inference(allocation, instance)
            )
            if cancelled:
                raise asyncio.CancelledError

    async def _finalize_inference(self, allocation, instance):
        # Keep ownership until both the instance slot and request count return,
        # including cancellation while either accounting lock is contended.
        self.pending_allocations.discard(allocation)
        reserved = self.allocated_instances.pop(allocation, instance)
        try:
            if reserved is not None:
                await reserved.add_requests(-1)
        finally:
            async with self.request_count_lock:
                self.request_count -= 1

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
                max_queue_length=10,
                num_gpu=self.resource_requirements["num_gpus"],
                node_id=node_id,
                lock=asyncio.Lock(),
            )
            reserved = False
            committed = False
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
                    await self._settle_owned(
                        self._cleanup_migration_instance(target_id, target)
                    )

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
