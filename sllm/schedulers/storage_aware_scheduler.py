# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import copy
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import ray

from sllm.logger import init_logger

from ..utils import InstanceStatus
from .fcfs_scheduler import FcfsScheduler

logger = init_logger(__name__)


@dataclass
class MigrationPlan:
    target_node_id: int
    source_instance: InstanceStatus


class MigrationPlans:
    def __init__(self):
        # self.evictedInstances = []  # List to store the instances that are evicted
        self.total_latency = float(
            "inf"
        )  # Initialize total_latency to infinity
        self.evictedGPUs = 0  # Counter for the total num_gpu freed
        self.plans = []
        self.store_info = None

    def append(self, plan: MigrationPlan, total_latency: float):
        # Update the total_latency estimate for the current plan
        self.total_latency = total_latency
        # Update the total num_gpu freed
        self.evictedGPUs += plan.source_instance.num_gpu
        # Add a instance to the Migration plan
        self.plans.append(plan)


@dataclass
class AllocationPlan:
    node_id: int
    latency: float
    migration_plans: Optional[MigrationPlans] = None


class StorageAwareScheduler(FcfsScheduler):
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__(scheduler_config)

        self.enable_migration = scheduler_config.get("enable_migration", False)

        self.store_manager = None

        self.model_scheduler_config = {}

    async def _control_loop(self):
        logger.info("Starting control loop")
        while self.running:
            loading_requests = []
            async with self.queue_lock:
                for (
                    model_name,
                    loading_queue,
                ) in self.model_loading_queues.items():
                    for idx, entry in enumerate(loading_queue):
                        loading_requests.append((model_name, idx, *entry))
            if self.store_manager is None:
                try:
                    self.store_manager = ray.get_actor("store_manager")
                except ValueError:
                    logger.error("Store manager not found")
                    await asyncio.sleep(1)
                    continue
            if loading_requests:
                await self._get_worker_nodes()
                model_info = await self.store_manager.get_model_info.remote()
                store_info = await self.store_manager.get_store_info.remote()
                hardware_info = (
                    await self.store_manager.get_hardware_info.remote()
                )
                loading_requests.sort(key=lambda x: x[1])
                for (
                    model_name,
                    idx,
                    request_time,
                    num_gpus,
                    allocation_result,
                ) in loading_requests:
                    if allocation_result.done():
                        continue
                    async with self.metadata_lock:
                        worker_nodes = copy.deepcopy(self.worker_nodes)
                    scheduling_options = await self.schedule(
                        model_name,
                        num_gpus,
                        worker_nodes,
                        model_info,
                        store_info,
                        hardware_info,
                    )
                    if not scheduling_options or allocation_result.done():
                        continue
                    scheduling_options.sort(
                        key=lambda x: (x.latency, x.node_id)
                    )
                    allocation_plan = scheduling_options[0]
                    migrations_succeeded = True
                    if allocation_plan.migration_plans is not None:
                        for (
                            migration_plan
                        ) in allocation_plan.migration_plans.plans:
                            if allocation_result.done():
                                migrations_succeeded = False
                                break
                            try:
                                router = ray.get_actor(
                                    migration_plan.source_instance.model_name,
                                    namespace="models",
                                )
                                target_instance_id = (
                                    await router.execute_migration_plan.remote(
                                        migration_plan
                                    )
                                )
                            except Exception as exc:
                                logger.error(
                                    f"Failed to execute migration plan: {exc}"
                                )
                                target_instance_id = None
                            if target_instance_id is None:
                                migrations_succeeded = False
                                break
                    if not migrations_succeeded:
                        continue
                    node_id = allocation_plan.node_id
                    # Router callbacks own migration reservations. Never write a
                    # speculative planning snapshot back over committed metadata.
                    async with self.queue_lock:
                        if allocation_result.done():
                            continue
                        owner = next(
                            (
                                key
                                for key, future in self.pending_allocations.items()
                                if future is allocation_result
                            ),
                            None,
                        )
                        entry = (request_time, num_gpus, allocation_result)
                        queue = self.model_loading_queues.get(model_name, [])
                        if owner is None or entry not in queue:
                            continue
                        async with self.metadata_lock:
                            # Cancellation can run while this lock is awaited.
                            if (
                                allocation_result.done()
                                or self.pending_allocations.get(owner)
                                is not allocation_result
                            ):
                                continue
                            node = self.worker_nodes.get(node_id)
                            if node is None or node["free_gpu"] < num_gpus:
                                continue
                            if owner[1] in self.model_instance.get(
                                owner[0], {}
                            ):
                                continue
                            self.model_instance.setdefault(owner[0], {})[
                                owner[1]
                            ] = node_id
                            node["free_gpu"] -= num_gpus
                            queue.remove(entry)
                            allocation_result.set_result(node_id)
                    try:
                        await self.store_manager.load_to_host.remote(
                            node_id, model_name
                        )
                    except Exception as exc:
                        # Publication transferred cleanup ownership to the caller.
                        logger.error(
                            f"Host load failed for model {model_name} "
                            f"on node {node_id}: {exc}"
                        )
            await asyncio.sleep(1)

    async def schedule(
        self,
        model_name,
        num_gpus,
        worker_nodes,
        model_info,
        store_info,
        hardware_info,
    ) -> List[AllocationPlan]:
        scheduling_options = []
        logger.info(f"Checking model {model_name}")
        for node_id, node_info in worker_nodes.items():
            logger.info(f"Checking node {node_id}, node info: {node_info}")
            if node_id not in store_info:
                logger.error(f"Node {node_id} not found in store info")
                continue
            free_gpu = node_info["free_gpu"]
            logger.info(f"Node {node_id} has {free_gpu} free GPUs")
            (
                node_store_info,
                pinned_memory_pool,
                node_waiting_time,
            ) = store_info[node_id]
            if model_name not in node_store_info:
                logger.info(f"Model {model_name} not found in node {node_id}")
                # Note(Yao): Downloading from HuggingFace Hub is
                # slower than network bandwidth and difficult to estimate.
                # So we just consider local checkpoints for now.
                continue
            latency = self._get_model_loading_time(
                model_name,
                model_info[model_name],
                hardware_info[node_id],
                node_waiting_time,
                pinned_memory_pool,
            )
            if free_gpu >= num_gpus:
                scheduling_options.append(AllocationPlan(node_id, latency))
            elif self.enable_migration:
                gpu_shortage = num_gpus - free_gpu
                logger.info(
                    f"Node {node_id} does not have enough GPU, trying migration"
                )
                migration_plans = await self.get_migration_plans(
                    model_name,
                    gpu_shortage,
                    node_id,
                    copy.deepcopy(worker_nodes),
                    copy.deepcopy(model_info),
                    copy.deepcopy(store_info),
                    copy.deepcopy(hardware_info),
                )
                if migration_plans is not None:
                    scheduling_options.append(
                        AllocationPlan(
                            node_id=node_id,
                            latency=latency + migration_plans.total_latency,
                            migration_plans=migration_plans,
                        )
                    )
                    logger.info(
                        f"Migration plans for model {model_name}: {migration_plans}"
                    )
            else:
                logger.info(
                    f"Node {node_id} does not have enough GPU and migration is disabled"
                )
        return scheduling_options

    async def get_migration_plans(
        self,
        model_name: str,
        gpu_shortage: int,
        source_node_id: int,
        worker_nodes: Mapping,
        model_info: Mapping,
        store_info: Mapping,
        hardware_info: Mapping,
    ) -> Optional[MigrationPlans]:
        migratable_instances = []
        request_routers = {}
        async with self.metadata_lock:
            model_instance = copy.deepcopy(self.model_instance)
        if model_instance:
            logger.info(f"Checking migratable instances for model {model_name}")
            for target_model_name in model_instance:
                # Skip the instances that is already running the model
                if target_model_name == model_name:
                    continue
                for instance_id, node_id in model_instance[
                    target_model_name
                ].items():
                    if node_id == source_node_id:
                        logger.info(
                            f"Checking instance {instance_id} of model {target_model_name}"
                        )
                        if target_model_name not in request_routers:
                            request_routers[target_model_name] = ray.get_actor(
                                target_model_name, namespace="models"
                            )
                        logger.info(
                            f"Getting status for instance {instance_id} of model {target_model_name}"
                        )
                        instance_status = await request_routers[
                            target_model_name
                        ].get_instance_status.remote(instance_id)
                        if instance_status:
                            logger.info(
                                f"Instance {instance_id} status: {instance_status}"
                            )
                            alpha = self.model_scheduler_config.get(
                                target_model_name, {}
                            ).get("alpha", 0.01)
                            beta = self.model_scheduler_config.get(
                                target_model_name, {}
                            ).get("beta", 0.1)
                            num_current_tokens = (
                                instance_status.num_current_tokens
                            )
                            resuming_latency = alpha * num_current_tokens + beta
                            instance_status.resuming_latency = resuming_latency
                            migratable_instances.append(instance_status)

        logger.info(f"Migratable instances: {migratable_instances}")

        if not migratable_instances:
            return None

        try:
            numInstances = len(
                migratable_instances
            )  # Number of instances currently running
            gpu_shortage = int(gpu_shortage)
            logger.info(
                f"Number of instances: {numInstances}, Required GPUs: {gpu_shortage}"
            )
            # Initialize the DP table with MigrationPlans objects
            dp = [
                [MigrationPlans() for _ in range(gpu_shortage + 1)]
                for _ in range(numInstances + 1)
            ]
            dp[0][0].total_latency = 0
            dp[0][0].store_info = copy.deepcopy(store_info)
            logger.info(f"Number of instances: {numInstances}")

            # Iterate over all instances and GPU capacities
            for i in range(1, numInstances + 1):
                logger.info(f"Checking instance {i}")
                for j in range(gpu_shortage + 1):
                    logger.info(f"Checking GPU capacity {j}")
                    current_instance = migratable_instances[i - 1]
                    # GPU requirement for the current instance
                    n_gpus = current_instance.num_gpu
                    logger.info(f"GPU requirement for instance {i}: {n_gpus}")
                    # Check if the current instance can fit into the current GPU capacity
                    if j >= n_gpus:
                        # Calculate total_latency when including the current instance
                        logger.info(
                            f"Instance {i} can fit into GPU capacity {j}"
                        )
                        logger.info(
                            f"Store info: {dp[i - 1][j - n_gpus].store_info}"
                        )
                        target_node_id, loading_time = (
                            self._get_migration_target(
                                current_instance.model_name,
                                n_gpus,
                                source_node_id,
                                worker_nodes,
                                model_info,
                                copy.deepcopy(dp[i - 1][j - n_gpus].store_info),
                                hardware_info,
                            )
                        )
                        if not current_instance.resuming_latency:
                            logger.warn(
                                f"Resuming latency not found for instance {i}"
                            )
                            current_instance.resuming_latency = 0
                        migration_latency = (
                            current_instance.resuming_latency + loading_time
                        )
                        plan = MigrationPlan(
                            target_node_id=target_node_id,
                            source_instance=current_instance,
                        )
                        total_latency = (
                            migration_latency
                            + dp[i - 1][j - n_gpus].total_latency
                        )
                        logger.info(
                            f"Total latency for instance {i} with {j} GPUs: {total_latency}"
                        )
                        if total_latency < dp[i - 1][j].total_latency:
                            # Copy the previous plan and include the current instance
                            dp[i][j] = copy.deepcopy(dp[i - 1][j - n_gpus])
                            dp[i][j].append(plan, total_latency)
                            dp[i][j].store_info[target_node_id][2] += (
                                loading_time
                            )
                            logger.info(f"Store info: {dp[i][j].store_info}")
                        else:
                            # Copy the previous plan without including the current instance
                            dp[i][j] = copy.deepcopy(dp[i - 1][j])
                    else:
                        logger.info(
                            f"Instance {i} cannot fit into GPU capacity {j}"
                        )
                        # If the instance cannot fit, carry forward the previous plan
                        dp[i][j] = copy.deepcopy(dp[i - 1][j])

            # Return the optimal Migration plans
            minLatency = float("inf")
            optimal_plans = None
            for plans in dp[numInstances]:
                logger.info(f"Total latency: {plans.total_latency}")
                if (
                    plans.total_latency < minLatency
                    and plans.evictedGPUs >= gpu_shortage
                ):
                    minLatency = plans.total_latency
                    optimal_plans = plans
            logger.info(f"Migration plans: {optimal_plans}")
            return optimal_plans
        except Exception as e:
            logger.error(f"Failed to get migration plans: {e}")
            return None

    async def mark_resource(
        self,
        model_name: str,
        instance_id: str,
        node_id: int,
        resources: Optional[Mapping] = None,
    ) -> bool:
        num_gpus = 1 if resources is None else resources.get("num_gpus", 0)
        if (
            isinstance(num_gpus, bool)
            or not isinstance(num_gpus, int)
            or num_gpus <= 0
        ):
            return False
        async with self.metadata_lock:
            node = self.worker_nodes.get(node_id)
            if node is None:
                return False
            instances = self.model_instance.get(model_name, {})
            if instance_id in instances:
                return instances[instance_id] == node_id
            if node["free_gpu"] < num_gpus:
                return False
            self.model_instance.setdefault(model_name, {})[instance_id] = (
                node_id
            )
            node["free_gpu"] -= num_gpus
        return True

    async def set_model_scheduler_config(
        self, model_name: str, scheduler_config: Mapping
    ) -> bool:
        logger.info(f"Setting scheduler config for model {model_name}")
        async with self.metadata_lock:
            self.model_scheduler_config[model_name] = scheduler_config
        return True

    def _get_model_loading_time(
        self,
        model_name: str,
        model_size: int,
        hardware_info: Mapping,
        node_waiting_time: float,
        pinned_memory_pool: Mapping,
    ) -> float:
        latency = 0
        if model_name not in pinned_memory_pool:
            latency += (
                node_waiting_time + model_size / hardware_info["disk_bandwidth"]
            )
            logger.info(
                f"Loading model {model_name} will take {latency} seconds"
            )
        else:
            latency += model_size / hardware_info["pcie_bandwidth"]
            logger.info(
                f"Loading model {model_name} will take {latency} seconds"
            )
        return latency

    def _get_migration_target(
        self,
        model_name: str,
        num_gpu: int,
        source_node_id: int,
        worker_nodes: Mapping,
        model_info: Mapping,
        store_info: Mapping,
        hardware_info: Mapping,
    ) -> Optional[Tuple[int, float]]:
        candidate_plans = []
        for node_id, node_info in worker_nodes.items():
            if node_id == source_node_id:
                continue
            if node_info["free_gpu"] >= num_gpu:
                loading_time = self._get_model_loading_time(
                    model_name,
                    model_info[model_name],
                    hardware_info[node_id],
                    store_info[node_id][2],
                    store_info[node_id][1],
                )
                candidate_plans.append((node_id, loading_time))
        if not candidate_plans:
            return None
        candidate_plans.sort(key=lambda x: x[1])
        # return the node with the shortest loading time
        return candidate_plans[0]
