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
import logging
import uuid
from typing import Dict, List, Optional

import ray

from sllm.fine_tuning_instance import start_ft_instance
from sllm.inference_instance import start_instance
from sllm.logger import init_logger

from ..utils import InstanceHandle
from .router_utils import SllmRouter

logger = init_logger(__name__)


async def auto_scaler(
    auto_scaling_metrics: Dict[str, int], auto_scaling_config: Dict[str, int]
) -> int:
    """
    Returns desired number of instances for a model based on the auto-scaling policy
    """

    request_count = auto_scaling_metrics.get("request_count", 0)

    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 2)

    desired_instances = (
        request_count + target_ongoing_requests - 1
    ) // target_ongoing_requests
    desired_instances = min(
        max_instances, max(min_instances, desired_instances)
    )

    return desired_instances


class RoundRobinRouter(SllmRouter):
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
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config
        self.router_config = router_config

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        # Inference instance pools
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        # Fine-tuning instance pools
        self.starting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.fine_tuning_count = 0
        self.fine_tuning_count_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.enable_lora = enable_lora
        self.loaded_lora_adapters = lora_adapters
        self.lora_lock = asyncio.Lock()

        self.auto_scaler = None
        self.load_balancer = None
        self.start_tasks = set()
        self.finish_tasks = set()
        self.pending_allocations = set()
        self.allocated_instances = {}
        self.close_task = None
        logger.info(f"Created new handler for model {self.model_name}")

    async def start(
        self, auto_scaling_config: Dict[str, int], mode: str = "inference"
    ):
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        if mode == "inference":
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config
            self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
            self.load_balancer = asyncio.create_task(self._load_balancer_loop())
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    async def update(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
        lora_adapters: Optional[Dict[str, str]] = None,
    ):
        if auto_scaling_config is not None:
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config

        if lora_adapters is not None:
            async with self.lora_lock:
                self.loaded_lora_adapters = lora_adapters

        logger.info(
            f"Model {self.model_name}'s auto scaling config updated to {auto_scaling_config}"
        )

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}
            async with self.request_count_lock:
                self.request_count += 1
            instance_allocation = self.loop.create_future()
            self.pending_allocations.add(instance_allocation)
            self.request_queue.put_nowait(instance_allocation)
        instance = None
        try:
            async with self.idle_time_lock:
                self.idle_time = 0
            instance_id = await instance_allocation
            if instance_id is None:
                return {"error": "Instance cancelled"}
            logger.info(f"{request_data}, type: {type(request_data)}")
            async with self.instance_management_lock:
                if instance_id not in self.ready_inference_instances:
                    logger.error(f"Instance {instance_id} not found")
                    return {"error": "Instance not found"}
                instance = self.ready_inference_instances[instance_id]

            # sanity check
            if self.enable_lora and "lora_adapter_name" in request_data:
                lora_adapter_name = request_data["lora_adapter_name"]
                if lora_adapter_name not in self.loaded_lora_adapters:
                    logger.error(f"Lora adapter {lora_adapter_name} not found")
                    return {
                        "error": f"Lora adapter {lora_adapter_name} not found"
                    }
                await instance.backend_instance.load_lora_adapter.remote(
                    lora_name=lora_adapter_name,
                    lora_path=self.loaded_lora_adapters[lora_adapter_name],
                )
            # NOTE: `.remote(request_data)` does not work, don't know why.
            # Looks like a known issue:
            # https://github.com/ray-project/ray/issues/26283#issuecomment-1780691475
            if action == "generate":
                result = await instance.backend_instance.generate.remote(
                    request_data=request_data
                )
            elif action == "encode":
                result = await instance.backend_instance.encode.remote(
                    request_data=request_data
                )
            else:
                result = {"error": "Invalid action"}
            return result
        finally:
            self.pending_allocations.discard(instance_allocation)
            reserved = self.allocated_instances.pop(
                instance_allocation, instance
            )
            if reserved is not None:
                await reserved.add_requests(-1)
            async with self.request_count_lock:
                self.request_count -= 1

    async def fine_tuning(self, request_data: dict):
        logger.info(f"Starting fine-tuning for model {self.model_name}")
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.fine_tuning_count_lock:
            self.fine_tuning_count += 1

        try:
            instance_id = await self._create_ft_instance()
        except Exception as e:
            logger.error(f"Failed to create fine-tuning instance: {str(e)}")
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {"error": f"Failed to create fine-tuning instance: {str(e)}"}

        max_wait_time = 300
        wait_time = 0
        while wait_time < max_wait_time:
            async with self.instance_management_lock:
                if instance_id in self.ready_ft_instances:
                    instance = self.ready_ft_instances[instance_id]
                    break
                elif instance_id not in self.starting_ft_instances:
                    logger.error(
                        f"Fine tuning instance {instance_id} not found in starting or ready instances"
                    )
                    async with self.fine_tuning_count_lock:
                        self.fine_tuning_count -= 1
                    return {"error": "Fine tuning instance not found"}
            await asyncio.sleep(0.1)
            wait_time += 0.1
        else:
            logger.error(
                f"Timeout waiting for fine tuning instance {instance_id} to be ready"
            )
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {
                "error": "Timeout waiting for fine tuning instance to be ready"
            }

        try:
            logger.info(f"Calling fine_tuning method on instance {instance_id}")
            result = await instance.backend_instance.fine_tuning.remote(
                request_data=request_data
            )

            logger.info(f"Finished processing fine-tuning {self.model_name}")
            await instance.add_requests(-1)

            await self._shutdown_instance(instance_id, is_ft=True)

            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1

            return result
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            await instance.add_requests(-1)
            await self._shutdown_instance(instance_id, is_ft=True)
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            logger.info(
                f"Fine-tuning failed and cleaned up for model {self.model_name}"
            )
            return {"error": f"Fine-tuning failed: {str(e)}"}

    async def delete_adapters(self, lora_adapters: List[str]):
        async with self.lora_lock:
            for adapter_name in lora_adapters:
                if adapter_name in self.loaded_lora_adapters:
                    del self.loaded_lora_adapters[adapter_name]
        logger.info(
            f"Deleted LoRA adapters {lora_adapters} on model {self.model_name}"
        )

    async def shutdown(self):
        async with self.running_lock:
            if self.close_task is None:
                self.running = False
                self.close_task = asyncio.create_task(self._close())
            task = self.close_task
        return await asyncio.shield(task)

    async def _close(self):
        loops = [
            task for task in (self.auto_scaler, self.load_balancer) if task
        ]
        for task in loops:
            task.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
        for future in list(self.pending_allocations):
            if not future.done():
                future.set_result(None)
        while not self.request_queue.empty():
            self.request_queue.get_nowait()
        # Starting actors own cleanup in _start_instance, including cancellation.
        starts = list(self.start_tasks)
        for task in starts:
            task.cancel()
        await asyncio.gather(*starts, return_exceptions=True)
        # Escalate any graceful drains before joining their owning tasks.
        async with self.instance_management_lock:
            draining = list(self.deleting_inference_instances.values())
        await asyncio.gather(
            *(item.backend_instance.shutdown.remote() for item in draining),
            return_exceptions=True,
        )
        await asyncio.gather(*list(self.finish_tasks), return_exceptions=True)
        async with self.instance_management_lock:
            deleted = list(self.ready_inference_instances)
        outcomes = await asyncio.gather(
            *(self._shutdown_instance(item) for item in deleted),
            return_exceptions=True,
        )
        failures = [
            item for item in outcomes if isinstance(item, BaseException)
        ]
        if failures:
            raise RuntimeError("Instance cleanup failed") from failures[0]
        return deleted

    async def _load_balancer_loop(self):
        # this is a simple round-robin load balancer
        round_robin_index = 0
        while True:
            instance_allocation = await self.request_queue.get()
            if instance_allocation.done():
                continue
            allocated = False
            logger.info(f"A request is waiting for model {self.model_name}")
            while not allocated and not instance_allocation.done():
                # 1. get ready instances
                instance_options = None
                while not instance_options and not instance_allocation.done():
                    await asyncio.sleep(1)
                    async with self.instance_management_lock:
                        instance_options = list(
                            self.ready_inference_instances.keys()
                        )
                    logger.info(f"{instance_options}")
                if instance_allocation.done():
                    break
                logger.info(f"Got ready instances {instance_options}")
                instance_id = instance_options[
                    round_robin_index % len(instance_options)
                ]
                round_robin_index += 1
                async with self.instance_management_lock:
                    if instance_id not in self.ready_inference_instances:
                        continue
                    instance = self.ready_inference_instances[instance_id]
                    # check if the request queue reaches max length
                    if await instance.check_request_queue():
                        allocated = await instance.add_requests(1)
                        if allocated:
                            if instance_allocation.done():
                                await instance.add_requests(-1)
                            else:
                                self.allocated_instances[
                                    instance_allocation
                                ] = instance
                                instance_allocation.set_result(instance_id)
                    else:
                        logger.info(
                            f"Instance {instance_id} cannot add another request"
                        )
                if not allocated:
                    await asyncio.sleep(self.loop_interval)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            async with self.request_count_lock:
                request_count = self.request_count
            auto_scaling_metrics = {"request_count": request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            async with self.instance_management_lock:
                num_running_instances = len(
                    self.starting_inference_instances
                ) + len(self.ready_inference_instances)
            logger.info(
                f"{self.model_name}: {num_running_instances} instances,"
                f"need {desired_instances} instances",
            )
            if desired_instances > num_running_instances:
                logger.info("Creating new instance")
                await self._create_instance()
            elif desired_instances < num_running_instances:
                keep_alive = auto_scaling_config.get("keep_alive", 0)
                if self.idle_time >= keep_alive:
                    logger.info(
                        f"Stopping instance, idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    await self._stop_instance()
                    async with self.idle_time_lock:
                        self.idle_time = 0
                else:
                    logger.info(
                        f"idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    async with self.idle_time_lock:
                        self.idle_time += self.loop_interval
            else:
                # logger.info("No scaling needed")
                pass
            await asyncio.sleep(self.loop_interval)

    async def _create_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # get max_queue_length from auto_scaling_config
        if self.auto_scaling_config.get("metric", "") == "concurrency":
            max_request_length = self.auto_scaling_config.get("target", 1)
        else:
            max_request_length = 1
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}, max queue length is {max_request_length}"
        )
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        task = self.loop.create_task(self._start_instance(instance_id))
        self.start_tasks.add(task)
        task.add_done_callback(self.start_tasks.discard)

        return instance_id

    async def _create_ft_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new FT instance {instance_id} for model {self.model_name}"
        )

        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=1,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_ft_instances[instance_id] = instance
        self.loop.create_task(self._start_ft_instance(instance_id))
        logger.info(f"Created task for starting FT instance {instance_id}")
        return instance_id

    async def _start_instance(self, instance_id):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]
        # Now ask model loading scheduler to load the model
        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        startup_node = None
        try:
            startup_node = (
                await self.model_loading_scheduler.allocate_resource.remote(
                    self.model_name, instance_id, self.resource_requirements
                )
            )
            startup_config = {
                "num_cpus": self.resource_requirements["num_cpus"],
                "num_gpus": self.resource_requirements["num_gpus"],
                "resources": {
                    "worker_node": 0.1,
                    f"worker_id_{startup_node}": 0.1,
                },
            }
            logger.info(
                f"Startup config: {startup_config}, {self.backend_config}"
            )

            creation = asyncio.ensure_future(
                start_instance.options(
                    resources={
                        "worker_node": 0.1,
                        f"worker_id_{startup_node}": 0.1,
                    }
                ).remote(
                    instance_id,
                    self.backend,
                    self.model_name,
                    self.backend_config,
                    startup_config,
                )
            )
            try:
                instance.backend_instance = await asyncio.shield(creation)
            except asyncio.CancelledError:
                instance.backend_instance = await creation
                raise
            logger.info(
                f"Started instance {instance_id} for model {self.model_name}"
            )
            async with instance.lock:
                instance.node_id = startup_node
            await instance.backend_instance.init_backend.remote()

            async with self.instance_management_lock:
                instance.ready = True
                self.ready_inference_instances[instance_id] = instance
                self.starting_inference_instances.pop(instance_id)
            return instance_id
        except BaseException:
            try:
                if instance.backend_instance is not None:
                    await instance.backend_instance.shutdown.remote()
            finally:
                try:
                    await self._release_instance(instance_id, instance)
                finally:
                    async with self.instance_management_lock:
                        self.starting_inference_instances.pop(instance_id, None)
            logger.exception("Instance startup failed: %s", instance_id)

    async def _start_ft_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.starting_ft_instances:
                logger.error(f"FT Instance {instance_id} not found")
                return
            instance = self.starting_ft_instances[instance_id]

        logger.info(
            f"Allocating FT resources for model {self.model_name} on {instance_id}"
        )
        try:
            startup_node = (
                await self.model_loading_scheduler.allocate_resource.remote(
                    self.model_name, instance_id, self.resource_requirements
                )
            )
            logger.debug(
                f"Allocated resources on node {startup_node} for FT instance {instance_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to allocate resources for FT instance {instance_id}: {str(e)}"
            )
            raise

        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }

        try:
            instance.backend_instance = await start_ft_instance.options(
                resources=startup_config["resources"]
            ).remote(
                instance_id,
                self.backend,
                self.model_name,
                self.backend_config,
                startup_config,
            )
        except Exception as e:
            logger.error(
                f"Failed to create Ray actor for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        try:
            await instance.backend_instance.init_backend.remote()
        except Exception as e:
            logger.error(
                f"Failed to initialize backend for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise

        async with self.instance_management_lock:
            self.ready_ft_instances[instance_id] = instance
            self.starting_ft_instances.pop(instance_id)
        logger.info(f"Fine-tuning instance {instance_id} is now ready")
        return instance_id

    async def _stop_instance(self, instance_id: Optional[str] = None):
        while len(self.ready_inference_instances) <= 0:
            await asyncio.sleep(1)

        async with self.instance_management_lock:
            if instance_id is None:
                instance_id, instance = self.ready_inference_instances.popitem()
            elif instance_id in self.ready_inference_instances:
                instance = self.ready_inference_instances.pop(instance_id)
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            self.deleting_inference_instances[instance_id] = instance
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        task = self.loop.create_task(self._finish_instance(instance_id))
        self.finish_tasks.add(task)
        task.add_done_callback(self.finish_tasks.discard)

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances[instance_id]
        async with instance.lock:
            instance.status = False
        try:
            await instance.backend_instance.stop.remote()
        finally:
            try:
                await self._release_instance(instance_id, instance)
            finally:
                async with self.instance_management_lock:
                    self.deleting_inference_instances.pop(instance_id, None)

    async def _release_instance(self, instance_id, instance):
        try:
            if instance.backend_instance is not None:
                ray.kill(instance.backend_instance, no_restart=True)
        finally:
            await self.model_loading_scheduler.deallocate_resource.remote(
                self.model_name, instance_id, self.resource_requirements
            )

    async def _shutdown_instance(self, instance_id: str, is_ft: bool = False):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            if is_ft:
                pool = self.ready_ft_instances
            else:
                pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
            async with instance.lock:
                instance.status = False
        try:
            await instance.backend_instance.shutdown.remote()
        finally:
            await self._release_instance(instance_id, instance)
        return
