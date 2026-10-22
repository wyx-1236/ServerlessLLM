"""Manual D1 qualification; importing is CPU-only and never reserves resources.

The external approved runner owns the deadline, cgroup, NVML monitoring and
resource release. ``passed`` describes runtime evidence, never GPU release or
transparent continuation. No export, downloading or production monkeypatching.
"""

import argparse
import ast
import asyncio
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

REQUIRED_CHECKS = (
    "private_store",
    "worker_mapping",
    "http_baseline",
    "scheduler_trigger",
    "active_source_exact_replay",
    "target_history",
    "atomic_publication",
    "source_cleanup",
    "fresh_http_equal",
    "target_prefix_cache_hit",
    "init_failure_rollback",
    "replay_failure_rollback",
    "controller_shutdown",
)


def validate_approval(config, environ, now, cgroup):
    indices = config.get("physical_gpu_indices")
    uuids = config.get("gpu_uuids")
    if (
        config.get("approved") is not True
        or config.get("server") != "inspur-gpu-12"
        or config.get("scope") != "stage-d1-migration"
        or not isinstance(indices, list)
        or len(indices) != 2
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != 2
        or config.get("action")
        != "soft lock GPU " + ",".join(map(str, indices))
        or not isinstance(uuids, list)
        or len(uuids) != 2
        or any(
            not isinstance(u, str)
            or not re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", u
            )
            for u in uuids
        )
        or len(set(uuids)) != 2
    ):
        raise ValueError("Missing exact approved two-GPU D1 scope")
    deadline = config.get("deadline_epoch")
    if (
        type(deadline) not in (float, int)
        or not now < deadline - 120
        or deadline > now + 1800
    ):
        raise ValueError("Invalid deadline or cleanup reserve reached")
    if environ.get("CUDA_VISIBLE_DEVICES") != ",".join(uuids):
        raise ValueError("Exact approved UUID order required")
    unit = config.get("unit", "")
    unit = unit if unit.endswith(".service") else unit + ".service"
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+\.service", unit)
        or not cgroup.splitlines()
        or not all(
            unit in PurePosixPath(line.split(":", 2)[-1]).parts
            for line in cgroup.splitlines()
        )
    ):
        raise ValueError("Runner is outside the approved service cgroup")
    output, ray_root = Path(config["output"]), Path(config["ray_directory"])
    if not output.is_absolute() or not ray_root.is_absolute():
        raise ValueError("Output and Ray root must be absolute")
    short_owned = (
        ray_root.parent == Path("/tmp")
        and ray_root.name.startswith("sd1-")
        and output.name == "gpu-stage-d1-" + ray_root.name[4:]
        and unit == "sllm-d1-migration-" + ray_root.name[4:] + ".service"
    )
    if (
        output.is_symlink()
        or ray_root.is_symlink()
        or not (
            ray_root.resolve().is_relative_to(output.resolve())
            and ray_root.resolve() != output.resolve()
            or short_owned
        )
    ):
        raise ValueError("Ray root must belong to this output/run")
    for key in ("artifact_path",):
        path = Path(config[key])
        if not path.is_absolute() or not path.is_dir() or path.is_symlink():
            raise ValueError("Missing immutable prepared artifact")
        if output.resolve().is_relative_to(path.resolve()):
            raise ValueError("Output must not modify artifact")
    snapshot = config["source_snapshot"]
    if (
        not Path(snapshot["path"]).is_absolute()
        or not Path(snapshot["path"]).is_dir()
        or not snapshot.get("revision")
    ):
        raise ValueError("Frozen source snapshot and revision required")
    if type(config.get("parent_network_namespace")) is not int:
        raise ValueError("Parent namespace inode required")
    verification_path = Path(config["native_verification_path"])
    verification = json.loads(verification_path.read_text())
    if (
        verification.get("passed") is not True
        or verification.get("wheel_sha256") != config["wheel_sha256"]
        or verification.get("deployment", {}).get("commit") != config["commit"]
        or verification.get("deployment", {}).get(
            "all_tracked_files_match_commit"
        )
        is not True
    ):
        raise ValueError(
            "Native wheel/deployment verification does not match frozen run"
        )
    if not (Path(config["artifact_path"]) / "artifact-manifest.json").is_file():
        raise ValueError(
            "Prepared artifact manifest missing; export is forbidden"
        )


def worker_environment(uuid, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    return {"CUDA_VISIBLE_DEVICES": uuid, "TMPDIR": str(directory)}


@contextlib.contextmanager
def parent_environment(values):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def trace_pairs(text, source, target):
    snapshots, pairs = {}, []
    for line in text.splitlines():
        for marker in (
            "Migration token snapshot: ",
            "Migration replay input: ",
        ):
            if marker not in line:
                continue
            try:
                record = ast.literal_eval(line.split(marker, 1)[1])
                if (
                    record["source_instance_id"] != source
                    or record["target_instance_id"] != target
                ):
                    continue
                tokens = record["tokens"]
                if (
                    not isinstance(tokens, list)
                    or not tokens
                    or not all(
                        isinstance(seq, list)
                        and len(seq) >= 64
                        and all(type(t) is int for t in seq)
                        for seq in tokens
                    )
                ):
                    continue
                iteration = record["iteration"]
                if type(iteration) is not int or iteration < 0:
                    continue
                if marker.startswith("Migration token"):
                    snapshots[iteration] = record
                elif snapshots.get(iteration) == record:
                    pairs.append(record)
            except (ValueError, SyntaxError, KeyError, TypeError):
                continue
    return pairs


def counter_file(directory, pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("Actual target backend PID required")
    path = Path(directory) / f"counter_{pid}.db"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Target PID has no owned multiprocess counters")
    return path


def counter_totals(rows, model):
    totals = {"queries": 0.0, "hits": 0.0}
    seen = set()
    for key, value, *_ in rows:
        _, sample, labels, _ = json.loads(key)
        if labels.get("model_name") != model:
            continue
        if labels.get("engine") != "0":
            raise ValueError("Unexpected target engine identity")
        for suffix in totals:
            if sample in (
                f"vllm:prefix_cache_{suffix}",
                f"vllm:prefix_cache_{suffix}_total",
            ):
                totals[suffix] += float(value)
                seen.add(suffix)
    if seen != set(totals):
        raise ValueError("Missing target/model prefix-cache counters")
    return totals


def cache_delta(before, after):
    delta = {key: after[key] - before[key] for key in ("queries", "hits")}
    if not all(0 < value < float("inf") for value in delta.values()):
        raise ValueError("No verified positive target prefix-cache reuse delta")
    return delta


def replace_backend_config(actor, overrides):
    previous = dict(actor.backend_config)
    actor.backend_config = {**previous, **overrides}
    return previous


def restore_backend_config(actor, previous):
    actor.backend_config = dict(previous)


@contextlib.asynccontextmanager
async def fault_configuration(router, overrides):
    previous = await router.__ray_call__.remote(
        replace_backend_config, overrides
    )
    try:
        yield
    finally:
        await router.__ray_call__.remote(restore_backend_config, previous)


def runtime_passed(result):
    return (
        not result.get("error")
        and result.get("cleanup_completed") is True
        and not result.get("cleanup_errors")
        and all(
            result.get("checks", {}).get(key) is True for key in REQUIRED_CHECKS
        )
    )


def actor_identity(actor):
    return {
        "pid": os.getpid(),
        "cgroup": Path("/proc/self/cgroup").read_text(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engine_model": actor.engine_args.model,
    }


def new_target_identity(
    rows, previous_ids, model, source, node, cgroup, proc_root=Path("/proc")
):
    """Read pinned Ray GCS metadata without blocking on target engine init."""
    candidates = [
        value
        for key, value in rows.items()
        if key not in previous_ids
        and value["Name"].startswith(model + "_")
        and value["Name"] != source
        and value.get("Pid", 0) > 0
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("Ambiguous qualification target actor")
    candidate = candidates[0]
    if candidate["Address"]["NodeID"] != node:
        raise ValueError("Target actor is on the wrong worker")
    process = Path(proc_root) / str(candidate["Pid"])
    try:
        actual_cgroup = (process / "cgroup").read_text()
    except FileNotFoundError:
        return None
    if actual_cgroup != cgroup:
        raise ValueError("Target actor is outside the owned cgroup")
    return {
        "pid": candidate["Pid"],
        "instance_id": candidate["Name"],
        "node_id": node,
        "cgroup": actual_cgroup,
    }


def router_state(actor):
    return {
        "ready": {
            key: {"node": value.node_id, "backend": value.backend_instance}
            for key, value in actor.ready_inference_instances.items()
        },
        "migration_record": dict(actor.migration_record),
        "request_count": actor.request_count,
        "pending": len(actor.pending_allocations),
    }


def scheduler_state(actor):
    return {
        "instances": {
            model: dict(items) for model, items in actor.model_instance.items()
        },
        "nodes": {
            node: dict(value) for node, value in actor.worker_nodes.items()
        },
    }


def install_probe(actor, name, source):
    actor.model_info[name] = 0
    actor.local_servers[source].disk_models[name] = ([], 0)
    return {
        "probe": name,
        "source_node": source,
        "weight": 0,
        "synthetic_metadata_only": True,
    }


def backend_history(actor):
    return [
        {
            "request_id": key,
            "prompt_token_ids": list(value.prompt_token_ids or []),
            "output_token_ids": list(value.outputs[0].token_ids),
        }
        for key, value in actor.request_history.items()
        if value is not None and value.outputs
    ]


def canonical_gpu_uuid(value):
    """Normalize Torch's bare UUID while rejecting malformed identifiers."""
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value
    ):
        raise ValueError("Invalid physical GPU UUID")
    return "GPU-" + value.removeprefix("GPU-").lower()


def worker_mapping():
    import ray
    import torch

    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def read_run_logs(root, offsets):
    text = []
    for path in root.rglob("worker-*.*"):
        if path.suffix not in (".out", ".err") or not path.is_file():
            continue
        inode, size = path.stat().st_ino, path.stat().st_size
        old_inode, old_size = offsets.get(str(path), (inode, 0))
        if (
            inode != old_inode
            or size < old_size
            or size - old_size > 32 * 1024**2
        ):
            raise RuntimeError(
                "Owned Ray logs replaced/truncated or evidence budget exceeded"
            )
        with path.open("rb") as stream:
            stream.seek(old_size)
            text.append(stream.read().decode("utf-8", errors="replace"))
    return "\n".join(text)


def gpu_memory(uuid):
    """Read-only NVML CLI sample; called only inside the approved run."""
    sample = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + uuid,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return int(sample.stdout.strip())


async def source_pump(backend, body, stopped, terminals):
    index = 0
    while not stopped.is_set():
        try:
            response = await backend.generate.remote(
                {**body, "request_id": body["request_id"] + str(index)}
            )
            terminals.append(
                {
                    "request_id": body["request_id"] + str(index),
                    "response": response,
                }
            )
        except BaseException as error:
            terminals.append(
                {
                    "request_id": body["request_id"] + str(index),
                    "terminal_exception": type(error).__name__,
                    "detail": str(error),
                }
            )
            break
        index += 1
        if "error" in response:
            break


async def wait_until(operation, predicate, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = await operation()
        if predicate(value):
            return value
        await asyncio.sleep(0.1)
    raise TimeoutError("Qualification observation deadline exceeded")


async def exercise_model(config, output, result, root, metrics):
    import httpx
    import ray
    from prometheus_client.mmap_dict import MmapedDict
    from qualification_process import cache_failure_evidence
    from qualify_service import write
    from transformers import AutoTokenizer

    from sllm.schedulers.storage_aware_scheduler import MigrationPlan

    pumps, pump_stop = [], asyncio.Event()

    async def exercise_faults():
        nonlocal pumps
        # Manual plans supplement the mandatory real scheduler trigger only.
        for fault, overrides in (
            ("init", {"kv_cache_memory_bytes": 1}),
            ("replay", {"max_model_len": 16}),
        ):
            before_state = await scheduler.__ray_call__.remote(scheduler_state)
            current = await routing()
            live_id, live = next(iter(current["ready"].items()))
            free_node = next(
                node
                for node, info in before_state["nodes"].items()
                if info["free_gpu"] == 1
            )
            free_uuid = config["gpu_uuids"][int(free_node)]
            memory_baseline = gpu_memory(free_uuid)
            async with fault_configuration(router, overrides):
                pumps = start_pumps(live["backend"], fault)
                migration = None
                failed_identity, failed_offsets = None, None
                fault_logs = {
                    str(path): (path.stat().st_ino, path.stat().st_size)
                    for path in root.rglob("worker-*.*")
                    if path.is_file()
                }
                previous_actor_ids = set(ray._private.state.actors())
                try:
                    await wait_until(
                        lambda: live["backend"].get_current_tokens.remote(),
                        lambda value: len(value) >= 2,
                    )
                    status = await router.get_instance_status.remote(live_id)
                    migration = asyncio.ensure_future(
                        router.execute_migration_plan.remote(
                            MigrationPlan(free_node, status)
                        )
                    )
                    observation_deadline = time.monotonic() + 180
                    while not migration.done():
                        if time.monotonic() >= observation_deadline:
                            raise TimeoutError(
                                "Fault migration did not settle within 180 seconds"
                            )
                        if failed_identity is None:
                            failed_identity = new_target_identity(
                                ray._private.state.actors(),
                                previous_actor_ids,
                                model,
                                live_id,
                                result["worker_mapping"][int(free_node)][
                                    "node_id"
                                ],
                                identity["cgroup"],
                            )
                            if failed_identity is not None:
                                failed_offsets = {
                                    name: value
                                    for name, value in fault_logs.items()
                                    if name.endswith(
                                        f'-{failed_identity["pid"]}.out'
                                    )
                                    or name.endswith(
                                        f'-{failed_identity["pid"]}.err'
                                    )
                                }
                        await asyncio.sleep(0.1)
                    try:
                        outcome = await migration
                    except Exception as error:
                        outcome = {"exception": str(error)}
                    assert outcome is None or isinstance(outcome, dict), outcome
                    assert (
                        failed_identity
                    ), "Failed target actor PID was never observed"
                    if fault == "init":
                        evidence = cache_failure_evidence(
                            "",  # Require this failed actor's new log marker.
                            root / "ray/session_latest/logs",
                            failed_offsets,
                            failed_identity["pid"],
                        )
                    else:
                        text = read_run_logs(root, fault_logs)
                        assert "KV cache replay failed" in str(
                            outcome
                        ) or "Migration replay failed" in str(outcome), outcome
                        assert re.search(
                            r"(?:maximum model length|max_model_len|maximum context length|context length).{0,80}\b16\b",
                            str(outcome),
                            re.IGNORECASE | re.DOTALL,
                        ), outcome
                        failed_pairs = trace_pairs(
                            text, live_id, failed_identity["instance_id"]
                        )
                        assert (
                            failed_pairs
                        ), "Missing actual failed-target replay input"
                        evidence = {
                            "exception": outcome,
                            "actual_trace": failed_pairs,
                        }

                    assert (await routing())["ready"].keys() == current[
                        "ready"
                    ].keys()
                    await wait_until(
                        lambda: scheduler.__ray_call__.remote(scheduler_state),
                        lambda state: state == before_state,
                    )

                    async def resource_boundary():
                        return {
                            "ray_free_gpu": ray.available_resources().get(
                                "GPU", 0
                            ),
                            "target_memory_mib": gpu_memory(free_uuid),
                        }

                    boundary = await wait_until(
                        resource_boundary,
                        lambda value: value["ray_free_gpu"] == 1
                        and value["target_memory_mib"] <= memory_baseline,
                    )
                    # No named actor other than the still-routed source survives.
                    assert not [
                        item
                        for item in ray.util.list_named_actors(
                            all_namespaces=True
                        )
                        if item["name"].startswith(model + "_")
                        and item["name"] != live_id
                    ]
                    recovery = await client.post(
                        "/v1/chat/completions",
                        json={**request, "request_id": "d1-after-" + fault},
                    )
                    assert (
                        recovery.status_code == 200
                        and recovery.json()["choices"] == reference["choices"]
                    )
                    result[fault + "_failure"] = {
                        "target": failed_identity,
                        "evidence": evidence,
                        "scheduler": before_state,
                        "recovery": recovery.json(),
                        "target_uuid": free_uuid,
                        "memory_baseline_mib": memory_baseline,
                        "resource_boundary": boundary,
                    }
                    result["checks"][fault + "_failure_rollback"] = True
                finally:
                    pump_stop.set()
                    if pumps:
                        await asyncio.wait_for(asyncio.gather(*pumps), 45)
                        pumps = []
                    if migration is not None and not migration.done():
                        migration.cancel()
                        await asyncio.gather(migration, return_exceptions=True)
                write(output / "migration-result.json", result)

    try:
        artifact, snapshot = (
            Path(config["artifact_path"]),
            config["source_snapshot"],
        )
        model = artifact.name
        assert model == "Qwen3-0.6B-stageB"
        model_config = dict(
            model=model,
            backend="vllm",
            num_gpus=1,
            placement_config={"local_disk": ["0", "1"]},
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
                kv_cache_memory_bytes=256 * 1024**2,
                enable_prefix_caching=True,
                seed=0,
                trace_debug=True,
                disable_log_stats=False,
                model_loader_extra_config={"store_address": "127.0.0.1:8073"},
            ),
            auto_scaling_config=dict(
                metric="concurrency",
                target=4,
                min_instances=1,
                max_instances=1,
                keep_alive=1800,
            ),
        )
        write(output / "model-config.json", model_config)
        request = dict(
            model=model,
            request_id="d1-reference",
            prompt="The capital of France is",
            temperature=0,
            max_tokens=8,
        )
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8343", timeout=180
        ) as client:
            response = await client.post("/register", json=model_config)
            assert response.status_code == 200, response.text
            response = await client.post("/v1/chat/completions", json=request)
            assert response.status_code == 200, response.text
            reference = response.json()
            assert reference["usage"]["completion_tokens"] == 8
            result["baseline_http"] = reference
            result["checks"]["http_baseline"] = True
            router = ray.get_actor(model, namespace="models")
            scheduler = ray.get_actor("model_loading_scheduler")
            manager = ray.get_actor("store_manager")

            # Functions passed to pinned Ray's __ray_call__ are synchronous;
            # no nested event-loop calls or replacement production methods.
            async def routing():
                return await router.__ray_call__.remote(router_state)

            initial = await routing()
            assert len(initial["ready"]) == 1
            source_id, source_info = next(iter(initial["ready"].items()))
            source = source_info["backend"]
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot["path"], local_files_only=True
            )
            tokens = tokenizer.encode(
                "Explain memory allocation and inference scheduling. " * 24
            )[:96]
            assert 64 <= len(tokens) <= 96
            body = dict(
                model=model,
                input_tokens=tokens,
                max_tokens=384,
                ignore_eos=True,
                temperature=0,
            )
            terminals = []

            def start_pumps(backend, label):
                pump_stop.clear()
                return [
                    asyncio.create_task(
                        source_pump(
                            backend,
                            {
                                **body,
                                "max_tokens": 384 - 32 * i,
                                "request_id": f"d1-{label}-{i}-",
                            },
                            pump_stop,
                            terminals,
                        )
                    )
                    for i in range(3)
                ]

            pumps = start_pumps(source, "active")
            await wait_until(
                lambda: source.get_current_tokens.remote(),
                lambda value: len(value) >= 2
                and all(len(seq) >= 64 for seq in value),
            )
            probe = "qualification-zero-weight-" + output.name
            result["allocation_probe"] = await manager.__ray_call__.remote(
                install_probe, probe, source_info["node"]
            )
            allocation = scheduler.allocate_resource.remote(
                probe, "qualification-probe", {"num_gpus": 1}
            )
            changed = await wait_until(
                routing,
                lambda state: source_id in state["migration_record"],
                180,
            )
            pump_stop.set()
            target_id = changed["migration_record"][source_id]
            assert (
                set(changed["ready"]) == {target_id}
                and source_id not in changed["ready"]
            )
            assert changed["request_count"] == 0 and changed["pending"] == 0
            target = changed["ready"][target_id]["backend"]
            assert await asyncio.wait_for(allocation, 60) == source_info["node"]
            await scheduler.deallocate_resource.remote(
                probe, "qualification-probe", {"num_gpus": 1}
            )
            await asyncio.wait_for(asyncio.gather(*pumps), 30)
            pumps = []
            result["source_request_terminals"] = terminals
            result["transparent_continuation"] = False
            result["checks"]["scheduler_trigger"] = True
            result["checks"]["atomic_publication"] = True
            logs = read_run_logs(root, {})
            pairs = trace_pairs(logs, source_id, target_id)
            assert (
                pairs
            ), "No actual active source snapshot paired with identical replay"
            result["migration_trace"] = pairs
            result["checks"]["active_source_exact_replay"] = True
            history = await target.__ray_call__.remote(backend_history)
            prompts = [
                item["prompt_token_ids"]
                for item in history
                if len(item["output_token_ids"]) == 1
            ]
            assert all(
                seq in prompts for pair in pairs for seq in pair["tokens"]
            ), history
            result["target_history"] = history
            result["checks"]["target_history"] = True
            try:
                await source.__ray_call__.remote(actor_identity)
            except ray.exceptions.RayActorError:
                pass
            else:
                raise AssertionError("Retired source actor remains alive")
            state = await scheduler.__ray_call__.remote(scheduler_state)
            assert state["instances"][model] == {
                target_id: changed["ready"][target_id]["node"]
            }
            assert state["nodes"][source_info["node"]]["free_gpu"] == 1
            result["source_cleanup"] = state
            result["checks"]["source_cleanup"] = True
            response = await client.post(
                "/v1/chat/completions",
                json={**request, "request_id": "d1-after"},
            )
            after = response.json()
            assert (
                response.status_code == 200
                and after["choices"] == reference["choices"]
                and after["usage"] == reference["usage"]
            )
            result["fresh_http"] = after
            result["checks"]["fresh_http_equal"] = True
            identity = await target.__ray_call__.remote(actor_identity)
            assert identity["cgroup"] == Path("/proc/self/cgroup").read_text()

            def counters():
                return counter_totals(
                    MmapedDict.read_all_values_from_file(
                        str(counter_file(metrics, identity["pid"]))
                    ),
                    identity["engine_model"],
                )

            # vLLM log_stats flushes periodically; require observed samples.
            await asyncio.sleep(10)
            before = counters()
            prefix = pairs[0]["tokens"][0]
            hit_request = await target.generate.remote(
                {"input_tokens": prefix, "max_tokens": 1, "temperature": 0}
            )
            assert "error" not in hit_request, hit_request

            async def cache_observation():
                observed = counters()
                try:
                    cache_delta(before, observed)
                except ValueError:
                    return None
                return observed

            after_counters = await wait_until(
                cache_observation, lambda value: value is not None, 30
            )
            result["target_prefix_cache"] = {
                "pid": identity["pid"],
                "before": before,
                "after": after_counters,
                "delta": cache_delta(before, after_counters),
            }
            result["checks"]["target_prefix_cache_hit"] = True
            await exercise_faults()
    finally:
        pump_stop.set()
        for task in pumps:
            task.cancel()
        if pumps:
            await asyncio.wait_for(
                asyncio.gather(*pumps, return_exceptions=True), 25
            )


async def qualify(config, output, result):
    # Set the multiprocess directory before importing vLLM/Prometheus anywhere.
    metrics = output / "prometheus"
    metrics.mkdir()
    os.environ.update(
        PROMETHEUS_MULTIPROC_DIR=str(metrics),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        STORAGE_PATH=str(Path(config["artifact_path"]).parents[1]),
        RAY_TMPDIR=config["ray_directory"],
    )
    import grpc
    import ray
    import uvicorn
    from qualify_service import store_settings, write
    from ray.cluster_utils import Cluster

    from sllm.app_lib import create_app
    from sllm.controller import SllmController
    from sllm.vllm_prepare_worker import (
        _enable_loopback,
        check_network_namespace,
    )
    from sllm_store.proto import storage_pb2, storage_pb2_grpc

    cluster, store, controller, server, http_task = None, None, None, None, None
    result["cleanup_errors"] = []
    try:
        result["isolation"] = check_network_namespace(
            config["parent_network_namespace"]
        )
        _enable_loopback()
        root = Path(config["ray_directory"])
        root.mkdir(parents=True, exist_ok=False)
        storage = Path(config["artifact_path"]).parents[1]
        code = (
            "import asyncio; from sllm_store.server import serve; asyncio.run(serve(**"
            + repr(store_settings(storage))
            + "))"
        )
        with (output / "store.log").open("w") as log:
            store = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        for address in ("127.0.0.1:8073", "127.0.0.2:8073"):
            with grpc.insecure_channel(address) as channel:
                stub = storage_pb2_grpc.StorageStub(channel)

                async def store_ready():
                    if store.poll() is not None:
                        raise RuntimeError("Private native Store exited")
                    try:
                        response = stub.GetServerConfig(
                            storage_pb2.GetServerConfigRequest(), timeout=0.5
                        )
                        return response.mem_pool_size == 2147483648
                    except grpc.RpcError:
                        return False

                await wait_until(store_ready, bool, 30)
        result["checks"]["private_store"] = True
        common = dict(
            include_dashboard=False,
            object_store_memory=80 * 1024**2,
            node_ip_address="127.0.0.2",
        )
        cluster = Cluster()
        cluster.add_node(
            num_cpus=4,
            num_gpus=0,
            resources={"control_node": 1},
            temp_dir=str(root / "ray"),
            **common,
        )
        environments = []
        for index, uuid in enumerate(config["gpu_uuids"]):
            env = worker_environment(uuid, root / f"worker-{index}")
            environments.append(env)
            with parent_environment(env):
                cluster.add_node(
                    num_cpus=2,
                    num_gpus=1,
                    resources={"worker_node": 1, f"worker_id_{index}": 1},
                    env_vars=env,
                    **common,
                )
        ray.init(
            address=cluster.address,
            namespace="stage-d1-" + output.name,
            _node_ip_address="127.0.0.2",
        )
        result["ray_nodes"] = [
            {"node_id": node["NodeID"], "resources": node["Resources"]}
            for node in ray.nodes()
        ]
        assert (
            ray.cluster_resources()["CPU"] == 8
            and ray.cluster_resources()["GPU"] == 2
        )
        mappings = []
        for index, uuid in enumerate(config["gpu_uuids"]):
            observed = (
                await ray.remote(max_calls=1)(worker_mapping)
                .options(
                    num_cpus=1,
                    num_gpus=1,
                    resources={f"worker_id_{index}": 0.1},
                )
                .remote()
            )
            assert observed["visible_devices"] == uuid and canonical_gpu_uuid(
                observed["cuda_uuid"]
            ) == canonical_gpu_uuid(uuid), observed
            assert observed["tmpdir"] == environments[index]["TMPDIR"], observed
            observed.update(
                logical_worker=index,
                physical_gpu_index=config["physical_gpu_indices"][index],
            )
            mappings.append(observed)
        assert len({item["node_id"] for item in mappings}) == 2
        result["worker_mapping"] = mappings
        result["checks"]["worker_mapping"] = True
        controller = (
            ray.remote(SllmController)
            .options(
                name="controller", num_cpus=1, resources={"control_node": 0.1}
            )
            .remote({"enable_storage_aware": True, "enable_migration": True})
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

        async def http_started():
            return server.started

        await wait_until(http_started, bool, 10)
        await exercise_model(config, output, result, root, metrics)
        await asyncio.wait_for(controller.shutdown.remote(), 45)
        result["checks"]["controller_shutdown"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:

        async def cleanup(name, operation):
            try:
                value = operation()
                if hasattr(value, "__await__"):
                    await asyncio.wait_for(value, 25)
            except BaseException:
                result["cleanup_errors"].append(
                    {"step": name, "error": traceback.format_exc()}
                )

        if server is not None:
            server.should_exit = True
        if http_task is not None:
            await cleanup("http", lambda: http_task)
        if controller is not None:
            if not result["checks"].get("controller_shutdown"):
                await cleanup(
                    "controller_shutdown", lambda: controller.shutdown.remote()
                )
            await cleanup(
                "controller_kill", lambda: ray.kill(controller, no_restart=True)
            )
        await cleanup("ray_driver", ray.shutdown)
        if cluster is not None:
            await cleanup("ray_cluster", cluster.shutdown)
        if store is not None and store.poll() is None:

            def stop_store():
                store.terminate()
                try:
                    store.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    store.kill()
                    store.wait(timeout=10)

            await cleanup("store", stop_store)
        result["cleanup_completed"] = not result["cleanup_errors"]
        result["passed"] = runtime_passed(result)
        result["status"] = "passed" if result["passed"] else "failed"
        result["ended_epoch"] = time.time()
        write(output / "migration-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", required=True)
    config = json.loads(Path(parser.parse_args().approval).read_text())
    validate_approval(
        config, os.environ, time.time(), Path("/proc/self/cgroup").read_text()
    )
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "passed": False,
        "status": "running",
        "started_epoch": time.time(),
        "checks": {},
        "approval": config,
        "external_release_verified": False,
        "limitations": [
            "single-host two logical workers",
            "source abort/cancel is terminal",
            "no transparent vLLM continuation",
            "external runner owns release proof",
        ],
    }
    result_file = output / "migration-result.json"
    with result_file.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {"status": "running", "passed": False, "result": str(result_file)}
        ),
        flush=True,
    )

    async def bounded():
        try:
            await asyncio.wait_for(
                qualify(config, output, result),
                config["deadline_epoch"] - time.time() - 120,
            )
        except BaseException:
            result.update(
                passed=False, status="failed", error=traceback.format_exc()
            )
            result_file.write_text(json.dumps(result, indent=2))

    asyncio.run(bounded())
    print(
        json.dumps({"status": result["status"], "passed": result["passed"]}),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
