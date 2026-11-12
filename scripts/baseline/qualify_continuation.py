"""Explicitly approved same-HTTP-request migration qualification.

Importing this module is CPU-only. This executable consumes an exact approval;
it never reserves GPUs. The external runner owns final process/GPU release.
"""

import argparse
import asyncio
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

SCENARIOS = ("greedy", "seeded-multi")
REQUIRED_CHECKS = (
    "private_store",
    "worker_mapping",
    "http_baseline",
    "scheduler_trigger",
    "same_http_continuation",
    "final_snapshot_restored",
    "full_tokens_equal",
    "request_ledger_settled",
    "atomic_publication",
    "source_cleanup",
    "target_prefix_cache_hit",
    "controller_shutdown",
)


def validate_approval(config, environ, now, cgroup):
    from qualify_migration import validate_approval as validate

    validate(
        config,
        environ,
        now,
        cgroup,
        scope="stage-d2-continuation",
        run_prefix="d2",
    )


def runtime_passed(result):
    cases = result.get("cases")
    return bool(
        not result.get("error")
        and result.get("cleanup_completed") is True
        and not result.get("cleanup_errors")
        and result.get("transparent_continuation") is True
        and all(
            result.get("checks", {}).get(key) is True for key in REQUIRED_CHECKS
        )
        and isinstance(cases, list)
        and cases
        and {case.get("scenario") for case in cases} == set(SCENARIOS)
        and all(
            any(
                case.get("scenario") == scenario
                and case.get("validation", {}).get("source_had_output") is True
                for case in cases
            )
            for scenario in SCENARIOS
        )
        and all(
            case.get("validated") is True
            and not case.get("error")
            and not case.get("aborted")
            and case.get("validation", {}).get("full_tokens_equal") is True
            and case.get("validation", {}).get("resumed_choice_indices")
            for case in cases
        )
    )


def _logprobs(rows):
    if rows is None:
        return None
    return [
        None
        if row is None
        else {
            str(token): {
                "logprob": value.logprob,
                "rank": value.rank,
                "decoded_token": value.decoded_token,
            }
            for token, value in row.items()
        }
        for row in rows
    ]


def backend_history(actor):
    """Read every choice without modifying engine or request state."""
    return [
        {
            "request_id": request_id,
            "prompt_token_ids": list(output.prompt_token_ids or []),
            "choices": [
                {
                    "index": choice.index,
                    "token_ids": list(choice.token_ids),
                    "text": choice.text,
                    "logprobs": _logprobs(choice.logprobs),
                    "cumulative_logprob": choice.cumulative_logprob,
                    "finish_reason": choice.finish_reason,
                    "stop_reason": choice.stop_reason,
                }
                for choice in output.outputs
            ],
        }
        for request_id, output in actor.request_history.items()
        if output is not None
    ]


def backend_active(actor):
    return {
        key: (
            []
            if record["output"] is None
            else [len(choice.token_ids) for choice in record["output"].outputs]
        )
        for key, record in actor.active_requests.items()
    }


def router_state(actor):
    return {
        "ready": {
            key: {
                "node": value.node_id,
                "backend": value.backend_instance,
                "concurrency": value.concurrency,
            }
            for key, value in actor.ready_inference_instances.items()
        },
        "migration_record": dict(actor.migration_record),
        "request_count": actor.request_count,
        "pending": len(actor.pending_allocations),
        "allocated": len(actor.allocated_instances),
        "tracked": len(actor.migration_requests),
    }


async def send_once(client, request, attempts):
    request_id = request["request_id"]
    if request_id in attempts:
        raise ValueError("Qualification must not resend a request ID")
    attempt = attempts[request_id] = {
        "request": copy.deepcopy(request),
        "started_epoch": time.time(),
    }
    try:
        response = await client.post("/v1/chat/completions", json=request)
        attempt["http_status"] = response.status_code
        attempt["response"] = response.json()
        if response.status_code != 200 or "error" in attempt["response"]:
            raise RuntimeError(f"HTTP request failed: {attempt['response']}")
        return attempt["response"]
    except BaseException as error:
        attempt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt["ended_epoch"] = time.time()


async def http_pump(client, body, label, stopped, attempts):
    index = 0
    while not stopped.is_set():
        await send_once(
            client, {**body, "request_id": f"{label}-{index}"}, attempts
        )
        index += 1


async def exercise_model(config, output, result, root, metrics):
    import httpx
    import qualify_migration as base
    import ray
    from continuation_evidence import find_handoff, parse_events, validate_case
    from prometheus_client.mmap_dict import MmapedDict
    from qualify_service import write
    from transformers import AutoTokenizer

    # Helpers from the reused script must not depend on that script directory
    # being on every Ray actor's module search path.
    ray.cloudpickle.register_pickle_by_value(base)
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    artifact = Path(config["artifact_path"])
    snapshot = config["source_snapshot"]
    model = artifact.name
    if model != "Qwen3-0.6B-stageB":
        raise ValueError(
            "This bounded qualification requires the Stage B artifact"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot["path"], local_files_only=True
    )
    tokens = tokenizer.encode(
        "Explain memory allocation and inference scheduling. " * 24
    )[:96]
    if not 64 <= len(tokens) <= 96:
        raise ValueError(
            "Qualification prompt is outside the bounded token range"
        )
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
    result["cases"], result["scenarios"] = [], []
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8343", timeout=240
    ) as client:
        registered = await client.post("/register", json=model_config)
        if registered.status_code != 200:
            raise RuntimeError(registered.text)
        # The baseline HTTP request also causes initial instance creation.
        baseline = await send_once(
            client,
            {
                "model": model,
                "request_id": "d2-baseline",
                "input_tokens": tokens,
                "temperature": 0,
                "max_tokens": 8,
                "ignore_eos": True,
            },
            {},
        )
        if baseline.get("usage", {}).get("completion_tokens") != 8:
            raise RuntimeError("Baseline HTTP generation failed")
        result["checks"]["http_baseline"] = True
        router = ray.get_actor(model, namespace="models")
        scheduler = ray.get_actor("model_loading_scheduler")
        manager = ray.get_actor("store_manager")
        for scenario in SCENARIOS:
            await exercise_scenario(
                scenario,
                config,
                output,
                result,
                root,
                metrics,
                model,
                tokens,
                client,
                router,
                scheduler,
                manager,
                base,
                MmapedDict,
                find_handoff,
                parse_events,
                validate_case,
            )
        result["transparent_continuation"] = True


async def exercise_scenario(
    scenario,
    config,
    output,
    result,
    root,
    metrics,
    model,
    tokens,
    client,
    router,
    scheduler,
    manager,
    base,
    counter_reader,
    find_handoff,
    parse_events,
    validate_case,
):
    import ray
    from qualify_service import write

    async def routing():
        return await router.__ray_call__.remote(router_state)

    initial = await routing()
    if len(initial["ready"]) != 1:
        raise RuntimeError("Expected one source instance")
    source_id, source_info = next(iter(initial["ready"].items()))
    source = source_info["backend"]
    source_identity = await source.__ray_call__.remote(base.actor_identity)
    if source_identity["cgroup"] != Path("/proc/self/cgroup").read_text():
        raise RuntimeError("Source is not in the owned cgroup")
    params = dict(temperature=0, n=1)
    if scenario == "seeded-multi":
        params = dict(
            temperature=0.7,
            seed=17,
            n=2,
            min_tokens=32,
            presence_penalty=0.2,
            frequency_penalty=0.1,
        )
    body = dict(
        model=model,
        input_tokens=tokens,
        max_tokens=384,
        ignore_eos=True,
        logprobs=5,
        **params,
    )
    reference_id = f"d2-{scenario}-reference"
    reference = await send_once(
        client, {**body, "request_id": reference_id}, {}
    )
    reference_history = next(
        item
        for item in await source.__ray_call__.remote(backend_history)
        if item["request_id"] == reference_id
    )
    attempts, observations = {}, {}
    stopped = asyncio.Event()
    pumps = [
        asyncio.create_task(
            http_pump(client, body, f"d2-{scenario}-{index}", stopped, attempts)
        )
        for index in range(3)
    ]
    allocation = None
    probe_released = False
    probe = f"qualification-continuation-{scenario}-{output.name}"
    scenario_evidence = {
        "scenario": scenario,
        "attempts": attempts,
        "source": source_identity,
        "source_instance": source_id,
        "reference": reference,
        "reference_history": reference_history,
        "active_observations": observations,
    }
    result["scenarios"].append(scenario_evidence)

    async def observe():
        for task in pumps:
            if task.done():
                task.result()
                if not stopped.is_set():
                    raise RuntimeError("HTTP pump stopped before migration")
        active = await source.__ray_call__.remote(backend_active)
        for request_id, lengths in active.items():
            attempt = attempts.get(request_id)
            if attempt and "ended_epoch" not in attempt:
                observations[request_id] = {
                    "observed_epoch": time.time(),
                    "lengths": lengths,
                }
        return active

    async def changed():
        state = await routing()
        if source_id not in state["migration_record"]:
            try:
                await observe()
            except ray.exceptions.RayActorError:
                state = await routing()
                if source_id not in state["migration_record"]:
                    raise
        if (
            allocation is not None
            and allocation.done()
            and allocation.result() is None
        ):
            raise RuntimeError("Migration allocation failed")
        return state

    try:
        write(output / "continuation-result.json", result)
        await base.wait_until(
            observe,
            lambda active: any(any(lengths) for lengths in active.values()),
            60,
        )
        scenario_evidence[
            "allocation_probe"
        ] = await manager.__ray_call__.remote(
            base.install_probe, probe, source_info["node"]
        )
        allocation = asyncio.ensure_future(
            scheduler.allocate_resource.remote(
                probe, f"continuation-{scenario}", {"num_gpus": 1}
            )
        )
        state = await base.wait_until(
            changed, lambda value: source_id in value["migration_record"], 180
        )
        stopped.set()
        target_id = state["migration_record"][source_id]
        if set(state["ready"]) != {target_id}:
            raise RuntimeError(
                "Source/target route publication is inconsistent"
            )
        target = state["ready"][target_id]["backend"]
        allocated_node = await asyncio.wait_for(asyncio.shield(allocation), 60)
        if allocated_node != source_info["node"]:
            raise RuntimeError(
                "Storage-aware allocation did not reclaim the source"
            )
        await scheduler.deallocate_resource.remote(
            probe, f"continuation-{scenario}", {"num_gpus": 1}
        )
        probe_released = True
        await asyncio.wait_for(asyncio.gather(*pumps), 180)
        target_identity = await target.__ray_call__.remote(base.actor_identity)
        if target_identity["cgroup"] != source_identity["cgroup"]:
            raise RuntimeError("Target is not in the same owned cgroup")
        logs = base.read_run_logs(root, {})
        events = parse_events(logs)
        frozen = [
            event
            for event in events
            if event["event"] == "frozen"
            and event["pid"] == source_identity["pid"]
        ]
        if len({event["epoch"] for event in frozen}) != 1:
            raise RuntimeError("Missing or ambiguous final source freeze")
        epoch = frozen[0]["epoch"]
        source_batch, target_batch, resumed = find_handoff(
            events, source_identity["pid"], target_identity["pid"], epoch
        )
        source_snapshots = {
            item["request_id"]: item for item in source_batch["requests"]
        }
        target_snapshots = {
            item["request_id"]: item for item in target_batch["requests"]
        }
        history = {
            item["request_id"]: item
            for item in await target.__ray_call__.remote(backend_history)
        }
        # Preserve all comparison inputs before assertions or actor cleanup.
        # A rejected candidate must be as diagnosable as a successful one.
        scenario_evidence.update(
            epoch=epoch,
            frozen=source_batch,
            prepared=target_batch,
            resumed=resumed,
            target=target_identity,
            target_instance=target_id,
            final_history=history,
        )
        write(output / "continuation-result.json", result)
        resumed_ids = {event["request_id"] for event in resumed}
        if not source_snapshots or not set(source_snapshots) <= resumed_ids:
            raise RuntimeError(
                "Final frozen requests were not activated at target"
            )
        validated = []
        for request_id, frozen_request in source_snapshots.items():
            attempt = attempts.get(request_id)
            if attempt is None or request_id not in observations:
                raise RuntimeError(
                    "Frozen request lacks an observed active HTTP caller"
                )
            if attempt.get("error") or attempt.get("http_status") != 200:
                raise RuntimeError(
                    "Original HTTP request did not complete successfully"
                )
            try:
                report = validate_case(
                    attempt["request"],
                    attempt["response"],
                    frozen_request,
                    target_snapshots[request_id],
                    history[request_id],
                    reference,
                    reference_history,
                    require_generated_prefix=False,
                )
            except Exception as exc:
                scenario_evidence["validation_error"] = {
                    "request_id": request_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                write(output / "continuation-result.json", result)
                raise
            validated.append(
                {
                    "scenario": scenario,
                    "request_id": request_id,
                    "validated": True,
                    "validation": report,
                }
            )
        result["cases"].extend(validated)
        settled = await base.wait_until(
            routing,
            lambda value: value["request_count"]
            == value["pending"]
            == value["allocated"]
            == value["tracked"]
            == 0
            and all(
                item["concurrency"] == 0 for item in value["ready"].values()
            ),
        )
        scenario_evidence["settled_router"] = {
            key: value for key, value in settled.items() if key != "ready"
        }
        try:
            await source.__ray_call__.remote(base.actor_identity)
        except ray.exceptions.RayActorError:
            pass
        else:
            raise RuntimeError("Source actor survived completed handoff")
        resources = await scheduler.__ray_call__.remote(base.scheduler_state)
        if (
            resources["instances"][model]
            != {target_id: state["ready"][target_id]["node"]}
            or resources["nodes"][source_info["node"]]["free_gpu"] != 1
        ):
            raise RuntimeError(
                "Source reservation was not released exactly once"
            )
        scenario_evidence["source_cleanup"] = resources

        # Measure a target-owned cache delta using a frozen original prefix.
        def counters():
            return base.counter_totals(
                counter_reader.read_all_values_from_file(
                    str(base.counter_file(metrics, target_identity["pid"]))
                ),
                target_identity["engine_model"],
            )

        await asyncio.sleep(10)
        before = counters()
        first = next(iter(source_snapshots.values()))
        prefix = first["prompt_token_ids"] + first["choices"][0]["token_ids"]
        hit = await target.generate.remote(
            {"input_tokens": prefix, "max_tokens": 1, "temperature": 0}
        )
        if "error" in hit:
            raise RuntimeError(f"Target cache probe failed: {hit}")

        async def observed_cache():
            after = counters()
            try:
                return {
                    "before": before,
                    "after": after,
                    "delta": base.cache_delta(before, after),
                }
            except ValueError:
                return None

        scenario_evidence["target_cache"] = await base.wait_until(
            observed_cache, bool, 30
        )
        for check in REQUIRED_CHECKS:
            if check not in {
                "private_store",
                "worker_mapping",
                "http_baseline",
                "controller_shutdown",
            }:
                result["checks"][check] = True
    finally:
        stopped.set()
        for task in pumps:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        if allocation is not None:
            # Cancelling the local ObjectRef future does not establish that
            # its remote allocation is gone. Observe late ownership first.
            allocated = await asyncio.wait_for(asyncio.shield(allocation), 60)
            if allocated is not None and not probe_released:
                await scheduler.deallocate_resource.remote(
                    probe, f"continuation-{scenario}", {"num_gpus": 1}
                )
        write(output / "continuation-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", required=True)
    config = json.loads(Path(parser.parse_args().approval).read_text())
    validate_approval(
        config, os.environ, time.time(), Path("/proc/self/cgroup").read_text()
    )
    from qualify_migration import qualify

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "passed": False,
        "status": "running",
        "started_epoch": time.time(),
        "checks": {},
        "cases": [],
        "approval": config,
        "transparent_continuation": False,
        "external_release_verified": False,
        "limitations": [
            "single-host Qwen3-0.6B BF16 TP1 eager V2",
            "greedy and explicitly seeded two-choice text generation only",
            "no EOS or cross-boundary stop-string GPU qualification",
            "rollback/cancellation GPU cases require separate qualification",
            "external runner owns GPU/process release proof",
        ],
    }
    path = output / "continuation-result.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    async def bounded():
        try:
            await asyncio.wait_for(
                qualify(
                    config,
                    output,
                    result,
                    exercise=exercise_model,
                    pass_check=runtime_passed,
                    result_name=path.name,
                    namespace="stage-d2-continuation",
                ),
                config["deadline_epoch"] - time.time() - 120,
            )
        except BaseException:
            result.update(
                passed=False, status="failed", error=traceback.format_exc()
            )
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    asyncio.run(bounded())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "result": str(path),
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
