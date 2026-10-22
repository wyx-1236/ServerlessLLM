"""Approved Stage C qualification inside an owned Linux network namespace.

This executable never reserves GPUs. The external approved runner supplies a
fixed commit, deadline, exact GPU UUID, Store artifact, source snapshot and
output directory, and owns the entire cgroup, including Ray/EngineCore children.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def store_settings(storage):
    return dict(
        # The caller has already verified a private loopback-only netns.
        # Both backend (.1) and Ray worker discovery (.2) use this Store.
        host="0.0.0.0",
        port=8073,
        storage_path=str(storage),
        registration_required=False,
        mem_pool_size=2147483648,
        chunk_size=67108864,
        num_thread=2,
    )


async def qualify(config, output):
    import httpx
    import ray
    import uvicorn
    from ray.cluster_utils import Cluster

    from sllm.app_lib import create_app
    from sllm.controller import SllmController
    from sllm.vllm_prepare_worker import (
        _enable_loopback,
        check_network_namespace,
    )

    isolation = check_network_namespace(config["parent_network_namespace"])
    _enable_loopback()
    result = {"passed": False, "isolation": isolation, "checks": {}}
    write(output / "service-result.json", result)
    cluster = Cluster()
    store = None
    controller = None
    server = None
    http_task = None
    artifact = Path(config["artifact_path"])
    storage = artifact.parents[1]
    snapshot = config["source_snapshot"]
    os.environ["STORAGE_PATH"] = str(storage)
    ray_root = Path(config["ray_directory"])
    os.environ["RAY_TMPDIR"] = str(ray_root)
    (ray_root / "ray").mkdir(parents=True, exist_ok=True)
    # Both logical Ray nodes belong to this one bounded, single-GPU host job.
    store_code = (
        "import asyncio; from sllm_store.server import serve; "
        "asyncio.run(serve(**" + repr(store_settings(storage)) + "))"
    )
    try:
        with (output / "store.log").open("w") as log:
            store = subprocess.Popen(
                [sys.executable, "-c", store_code],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        import grpc

        from sllm_store.proto import storage_pb2, storage_pb2_grpc

        endpoints = ["127.0.0.1:8073", "127.0.0.2:8073"]
        for address in endpoints:
            ready = False
            with grpc.insecure_channel(address) as channel:
                stub = storage_pb2_grpc.StorageStub(channel)
                for _ in range(60):
                    if store.poll() is not None:
                        raise RuntimeError("Private Store exited")
                    try:
                        response = stub.GetServerConfig(
                            storage_pb2.GetServerConfigRequest(), timeout=0.5
                        )
                        ready = response.mem_pool_size == 2147483648
                        if ready:
                            break
                    except grpc.RpcError:
                        await asyncio.sleep(0.5)
            assert ready, f"Private Store readiness timeout: {address}"
        result["store_endpoints"] = endpoints
        result["checks"]["private_store"] = True
        write(output / "service-result.json", result)
        common = dict(
            include_dashboard=False,
            object_store_memory=80 * 1024**2,
            node_ip_address="127.0.0.2",
        )
        # Controller, StoreManager, scheduler and router each reserve one CPU
        # on the control node. Worker CPUs cannot satisfy control_node demand.
        cluster.add_node(
            num_cpus=4,
            num_gpus=0,
            resources={"control_node": 1},
            temp_dir=str(ray_root / "ray"),
            **common,
        )
        cluster.add_node(
            num_cpus=4,
            num_gpus=1,
            resources={"worker_node": 1, "worker_id_0": 1},
            **common,
        )
        ray.init(
            address=cluster.address,
            namespace="stage-c-" + output.name,
            _node_ip_address="127.0.0.2",
        )
        result["ray_nodes"] = [
            {"node_id": node["NodeID"], "resources": node["Resources"]}
            for node in ray.nodes()
        ]
        assert ray.cluster_resources().get("GPU") == 1
        controller = (
            ray.remote(SllmController)
            .options(
                name="controller", num_cpus=1, resources={"control_node": 0.1}
            )
            .remote({"enable_storage_aware": False, "enable_migration": False})
        )
        await asyncio.wait_for(controller.start.remote(), 90)
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(),
                host="127.0.0.1",
                port=8343,
                log_level="info",
                lifespan="off",
            )
        )
        http_task = asyncio.create_task(server.serve())
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started
        model = artifact.name
        model_config = dict(
            model=model,
            backend="vllm",
            num_gpus=1,
            backend_config=dict(
                pretrained_model_name_or_path=snapshot["path"],
                revision=snapshot["revision"],
                source_snapshot=snapshot,
                dtype="bfloat16",
                tensor_parallel_size=1,
                task="generate",
                enforce_eager=True,
                max_model_len=512,
                max_num_seqs=4,
                max_num_batched_tokens=512,
                kv_cache_memory_bytes=67108864,
                seed=0,
                model_loader_extra_config={"store_address": "127.0.0.1:8073"},
            ),
            auto_scaling_config=dict(
                metric="concurrency",
                target=4,
                min_instances=1,
                max_instances=1,
                keep_alive=60,
            ),
        )
        write(output / "model-config.json", model_config)
        request = dict(
            model=model,
            request_id="stage-c-single",
            prompt="The capital of France is",
            temperature=0,
            max_tokens=8,
        )
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8343", timeout=180
        ) as client:
            assert (await client.get("/health")).status_code == 200
            response = await client.post("/register", json=model_config)
            assert response.status_code == 200, response.text
            one = await client.post("/v1/chat/completions", json=request)
            assert one.status_code == 200, one.text
            first = one.json()
            assert (
                first["id"] == request["request_id"]
                and first["usage"]["completion_tokens"] > 0
            ), first
            result["single"] = first
            result["checks"]["http_controller_ray_store_vllm"] = True
            requests = [
                {**request, "request_id": "stage-c-concurrent-" + str(index)}
                for index in range(4)
            ]
            responses = await asyncio.gather(
                *(
                    client.post("/v1/chat/completions", json=body)
                    for body in requests
                )
            )
            concurrent = [item.json() for item in responses]
            assert all(item.status_code == 200 for item in responses)
            assert {item["id"] for item in concurrent} == {
                item["request_id"] for item in requests
            }
            assert all(
                item["choices"] == first["choices"]
                and item["usage"] == first["usage"]
                for item in concurrent
            )
            result["concurrent"] = concurrent
            result["checks"]["concurrent_unique_requests"] = True
            # Verify a backend error returns router counters and serving remains usable.
            invalid = await client.post(
                "/v1/chat/completions",
                json={
                    **request,
                    "request_id": "stage-c-invalid",
                    "max_tokens": -1,
                },
            )
            result["invalid_response"] = {
                "status": invalid.status_code,
                "body": invalid.text,
            }
            assert (
                invalid.status_code >= 400 or "error" in invalid.json()
            ), invalid.text
            retry = await client.post(
                "/v1/chat/completions",
                json={**request, "request_id": "stage-c-after-error"},
            )
            assert (
                retry.status_code == 200
                and retry.json()["usage"] == first["usage"]
            )
            result["checks"]["error_then_recovery"] = True
            names = ray.util.list_named_actors(all_namespaces=True)
            actors = [
                item for item in names if item["name"].startswith(model + "_")
            ]
            assert len(actors) == 1, actors
            backend = ray.get_actor(
                actors[0]["name"], namespace=actors[0]["namespace"]
            )
            long_request = {
                **request,
                "request_id": "stage-c-drain",
                "max_tokens": 384,
                "ignore_eos": True,
            }
            draining = asyncio.ensure_future(
                backend.generate.remote(long_request)
            )
            active = False
            for _ in range(200):
                if await backend.get_current_tokens.remote():
                    active = True
                    break
                if draining.done():
                    break
                await asyncio.sleep(0.01)
            assert active, "No active request observed before graceful stop"
            stop = asyncio.ensure_future(backend.stop.remote())
            await asyncio.sleep(0.05)
            rejected = await backend.generate.remote(
                {**request, "request_id": "stage-c-after-stop"}
            )
            assert "not running" in rejected.get("error", ""), rejected
            drained = await asyncio.wait_for(draining, 45)
            await asyncio.wait_for(stop, 45)
            assert drained["usage"]["completion_tokens"] == 384, drained
            assert await backend.get_current_tokens.remote() == []
            result["checks"]["real_active_graceful_stop"] = True
            removed = await client.post("/delete", json={"model": model})
            assert removed.status_code == 200, removed.text
            result["checks"]["http_delete_awaited_shutdown"] = True
            response = await client.post("/register", json=model_config)
            assert response.status_code == 200, response.text
            restarted = await client.post(
                "/v1/chat/completions",
                json={**request, "request_id": "stage-c-restarted"},
            )
            assert (
                restarted.status_code == 200
                and restarted.json()["choices"] == first["choices"]
            ), restarted.text
            result["restarted"] = restarted.json()
            result["checks"]["fresh_actor_restart"] = True
            names = ray.util.list_named_actors(all_namespaces=True)
            actors = [
                item for item in names if item["name"].startswith(model + "_")
            ]
            assert len(actors) == 1
            backend = ray.get_actor(
                actors[0]["name"], namespace=actors[0]["namespace"]
            )
            aborted = asyncio.ensure_future(
                backend.generate.remote(
                    {
                        **request,
                        "request_id": "stage-c-abort",
                        "max_tokens": 384,
                        "ignore_eos": True,
                    }
                )
            )
            active = False
            for _ in range(200):
                if await backend.get_current_tokens.remote():
                    active = True
                    break
                if aborted.done():
                    break
                await asyncio.sleep(0.01)
            assert active, "No active request observed before forced shutdown"
            await asyncio.wait_for(backend.shutdown.remote(), 45)
            await asyncio.gather(aborted, return_exceptions=True)
            assert await backend.get_current_tokens.remote() == []
            result["checks"]["real_active_forced_shutdown"] = True
            assert (
                await client.post("/delete", json={"model": model})
            ).status_code == 200
        await asyncio.wait_for(controller.shutdown.remote(), 45)
        result["checks"]["controller_shutdown"] = True
        ray.kill(controller, no_restart=True)
        controller = None
        result["runtime_checks_passed"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        if server is not None:
            server.should_exit = True
        if http_task is not None:
            await asyncio.gather(http_task, return_exceptions=True)
        if controller is not None:
            try:
                await asyncio.wait_for(controller.shutdown.remote(), 20)
            except BaseException:
                result["shutdown_error"] = traceback.format_exc()
            finally:
                ray.kill(controller, no_restart=True)
        ray.shutdown()
        cluster.shutdown()
        if store is not None and store.poll() is None:
            store.terminate()
            try:
                store.wait(timeout=10)
            except subprocess.TimeoutExpired:
                store.kill()
                store.wait(timeout=10)
        result["cleanup_completed"] = True
        result["passed"] = (
            result.get("runtime_checks_passed", False)
            and "shutdown_error" not in result
        )
        result["ended_epoch"] = time.time()
        write(output / "service-result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.approval).read_text())
    if (
        config.get("approved") is not True
        or config.get("action") != "soft lock GPU 1"
        or config.get("server") != "inspur-gpu-12"
    ):
        raise RuntimeError("Missing exact approved GPU scope")
    if (
        config.get("scope") != "stage-c-service"
        or time.time() >= config["deadline_epoch"] - 120
    ):
        raise RuntimeError("Invalid scope or cleanup reserve reached")
    if os.getenv("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise RuntimeError("GPU UUID mapping mismatch")
    output = Path(config["output"])
    output.mkdir(exist_ok=True)
    asyncio.run(qualify(config, output))


if __name__ == "__main__":
    main()
