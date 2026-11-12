"""CPU-only boundaries for stopping approval, selection and shared flow."""

import asyncio
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_continuation_flow import flow_factory as flow_factory
from test_continuation_qualification import (
    approval as approval,
)
from test_continuation_qualification import (
    qualification as qualification,
)
from test_continuation_qualification import (
    successful_result,
)


@pytest.fixture
def stopping(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "scripts/baseline")
    )
    return importlib.import_module("qualify_stopping")


def test_approval_scopes_are_isolated(stopping, qualification, approval):
    config = {
        **approval,
        "scope": "stage-d2-stopping",
        "numerical_profile": "batch-invariant-v1",
    }
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(config["gpu_uuids"]),
        "VLLM_BATCH_INVARIANT": "1",
    }
    cgroup = "0::/system.slice/" + config["unit"] + ".service\n"
    stopping.validate_approval(config, env, 1000, cgroup)
    for field, value in [
        ("scope", "stage-d2-continuation"),
        ("scope", "stage-d1-migration"),
        ("numerical_profile", "default"),
    ]:
        with pytest.raises(ValueError):
            stopping.validate_approval(
                {**config, field: value}, env, 1000, cgroup
            )
    with pytest.raises(ValueError):
        qualification.validate_approval(config, env, 1000, cgroup)


def test_old_result_cannot_pass_stopping(stopping, qualification):
    assert not stopping.runtime_passed(successful_result(qualification))


def configure_stopping_flow(flow, tmp_path, mode="normal"):
    from test_continuation_profile import worker_report

    class Tokenizer:
        eos_token_id = 2

        def decode(self, tokens, **kwargs):
            return "".join(chr(t) for t in tokens if t != 2)

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            return [10, 11]

    tokenizer = Tokenizer()

    def history(request, *, frozen=False):
        eos = not request.get("ignore_eos", True)
        stop = request.get("stop")
        size = (
            (12 if eos else 100)
            if frozen
            else (64 if eos else 320 if stop else 384)
        )
        if frozen and stop and mode == "outside-boundary":
            size = 16
        tokens = list(range(1000, 1000 + size))
        if not frozen and eos and mode != "probe-exhausted":
            tokens[-1] = 2
        raw = tokenizer.decode(tokens)
        if frozen and stop:
            raw = raw[: max(0, len(raw) - len(stop[0]) + 1)]
        elif stop and not request.get("include_stop_str_in_output"):
            raw = raw[:32]
        item = flow.choice(
            0, size, None if frozen else "stop" if eos or stop else "length"
        )
        item.update(
            token_ids=tokens,
            text=raw,
            stop_reason=stop[0] if stop and not frozen else None,
        )
        item["logprobs"] = [
            {t: dict(logprob=-0.1, rank=1, decoded_token=str(t))}
            for t in tokens
        ]
        return dict(
            request_id=request["request_id"],
            prompt_token_ids=request["input_tokens"],
            choices=[item],
        )

    flow.history = history
    snapshots = flow.snapshots

    def frozen_snapshots():
        items = snapshots()
        for item in items:
            if not item["sampling_params"].get("ignore_eos"):
                item["sampling_params"].update(
                    _eos_token_id=2, stop_token_ids=[]
                )
        return items

    flow.snapshots = frozen_snapshots
    source = worker_report()
    source["cgroup"] = "0::/cpu-continuation-test\n"
    target = {
        **source,
        "gpu_uuid": "GPU-22222222-2222-2222-2222-222222222222",
        "pid": 202,
    }
    flow.numerical_reports = {"source": [source], "target": [target]}
    config = dict(
        scope="stage-d2-stopping",
        numerical_profile="batch-invariant-v1",
        gpu_uuids=[source["gpu_uuid"], target["gpu_uuid"]],
        source_snapshot={"path": str(tmp_path)},
    )
    return config, tokenizer


@pytest.mark.asyncio
async def test_all_stopping_scenarios_and_summary_reject_corruption(
    stopping, flow_factory, tmp_path
):
    result = {
        "checks": {},
        "cases": [],
        "scenarios": [],
        "suite": stopping.SCOPE,
    }
    for scenario in stopping.SCENARIOS:
        flow = flow_factory()
        config, tokenizer = configure_stopping_flow(flow, tmp_path)
        result["approval"] = config
        await asyncio.wait_for(
            flow.run(
                scenario, result, config, suite=stopping, tokenizer=tokenizer
            ),
            5,
        )
        assert flow.pumps_drained.is_set() and not flow.reservations
        assert len(set(flow.posts)) == len(flow.posts)
    result.update(
        cleanup_completed=True, cleanup_errors=[], transparent_continuation=True
    )
    result["checks"] = dict.fromkeys(
        stopping.continuation.REQUIRED_CHECKS, True
    )
    assert stopping.runtime_passed(result)
    assert len(result["numerical_observations"]) == 6
    assert len(result["cases"]) == 9
    for mutation in [
        "duplicate-case",
        "missing-scenario",
        "unrelated-id",
        "lost-prefix",
        "wrong-eos-source",
        "lost-worker",
    ]:
        bad = copy.deepcopy(result)
        if mutation == "duplicate-case":
            bad["cases"].append(bad["cases"][0])
        if mutation == "missing-scenario":
            bad["scenarios"].pop()
        if mutation == "unrelated-id":
            bad["scenarios"][0]["stopping_validation"][
                "qualified_request_ids"
            ] = ["foreign"]
        if mutation == "lost-prefix":
            bad["scenarios"][1]["stopping"]["decoded_prefixes"] = {}
        if mutation == "wrong-eos-source":
            bad["scenarios"][0]["stopping"]["eos_source"][
                "tokenizer_eos_token_id"
            ] = 8
        if mutation == "lost-worker":
            bad["numerical_observations"].pop()
        assert not stopping.runtime_passed(bad), mutation


@pytest.mark.asyncio
async def test_eos_probe_exhaustion_does_not_start_migration(
    stopping, flow_factory, tmp_path
):
    flow = flow_factory()
    config, tokenizer = configure_stopping_flow(
        flow, tmp_path, "probe-exhausted"
    )
    result = {"checks": {}, "cases": [], "scenarios": []}
    with pytest.raises(ValueError, match="No qualified EOS"):
        await flow.run(
            "eos", result, config, suite=stopping, tokenizer=tokenizer
        )
    assert len(flow.posts) == 3
    assert not flow.requests and not flow.allocation_started.is_set()
    saved = json.loads((flow.output / stopping.RESULT_NAME).read_text())
    assert len(saved["scenarios"][0]["stopping"]["probes"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [object(), [True], [[10, 11]], [-1]])
async def test_invalid_prompt_tokens_keep_failure_evidence(
    stopping, flow_factory, tmp_path, tokens
):
    flow = flow_factory()
    config, tokenizer = configure_stopping_flow(flow, tmp_path)
    tokenizer.apply_chat_template = lambda *args, **kwargs: tokens
    result = {"checks": {}, "cases": [], "scenarios": []}
    with pytest.raises(ValueError, match="flat list of non-negative integer"):
        await flow.run(
            "eos", result, config, suite=stopping, tokenizer=tokenizer
        )
    saved = json.loads((flow.output / stopping.RESULT_NAME).read_text())
    scenario = saved["scenarios"][0]
    assert "flat list" in scenario["preparation_error"]
    assert "prompt_token_ids" not in scenario["stopping"]["probes"][0]
    assert not flow.posts and not flow.allocation_started.is_set()


@pytest.mark.asyncio
async def test_stopping_boundary_failure_keeps_evidence_and_releases(
    stopping, flow_factory, tmp_path
):
    flow = flow_factory()
    config, tokenizer = configure_stopping_flow(
        flow, tmp_path, "outside-boundary"
    )
    result = {"checks": {}, "cases": [], "scenarios": []}
    with pytest.raises(ValueError, match="No request crossed"):
        await asyncio.wait_for(
            flow.run(
                "stop-string-exclude",
                result,
                config,
                suite=stopping,
                tokenizer=tokenizer,
            ),
            5,
        )
    assert (
        flow.pumps_drained.is_set()
        and len(flow.deallocated) == 1
        and not flow.reservations
    )
    saved = json.loads((flow.output / stopping.RESULT_NAME).read_text())
    assert (
        saved["scenarios"][0]["frozen"]
        and saved["scenarios"][0]["final_history"]
    )
    assert saved["scenarios"][0]["stopping_error"] and not saved["cases"]


@pytest.mark.asyncio
async def test_cancellation_waits_for_late_owned_allocation(
    stopping, flow_factory, tmp_path
):
    flow = flow_factory()
    config, tokenizer = configure_stopping_flow(flow, tmp_path)

    async def delayed(probe, instance, resources):
        flow.allocation_started.set()
        await flow.late_allocation.wait()
        flow.reservations[probe] = instance
        return "0"

    flow.scheduler.allocate_resource.remote = delayed
    result = {"checks": {}, "cases": [], "scenarios": []}
    task = asyncio.create_task(
        flow.run("eos", result, config, suite=stopping, tokenizer=tokenizer)
    )
    try:
        await asyncio.wait_for(flow.allocation_started.wait(), 2)
        task.cancel()
        await asyncio.wait_for(flow.pumps_drained.wait(), 2)
        assert not task.done()
        flow.late_allocation.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert len(flow.deallocated) == 1 and not flow.reservations
        assert (
            flow.router.request_count == 0
            and not flow.router.allocated_instances
        )
        assert (flow.output / stopping.RESULT_NAME).is_file()
    finally:
        flow.late_allocation.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_new_entry_imports_without_model_dependencies(tmp_path):
    import subprocess
    import sys

    scripts = Path(__file__).parents[2] / "scripts/baseline"
    program = """
import builtins,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','vllm','transformers','ray'}:
        raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
sys.path.insert(0,sys.argv[1])
import qualify_stopping,stopping_evidence
assert qualify_stopping.SCOPE=='stage-d2-stopping'
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program, str(scripts)],
        check=True,
        cwd=tmp_path,
        timeout=20,
    )


@pytest.mark.asyncio
async def test_shared_flow_calls_explicit_suite_and_owns_its_result(
    flow_factory,
):
    flow = flow_factory()
    calls = []

    async def prepare(
        scenario,
        model,
        tokens,
        tokenizer,
        config,
        source,
        client,
        evidence,
        save,
    ):
        calls.append("prepare")
        evidence["preparation_seen"] = True
        save()
        body = dict(
            model=model,
            input_tokens=tokens,
            temperature=0,
            n=1,
            max_tokens=384,
            ignore_eos=True,
            logprobs=5,
        )
        reference = await flow.module.send_once(
            client, {**body, "request_id": "custom-reference"}, {}
        )
        history = flow.module.backend_history(source)[0]
        return body, reference, history

    def validate(evidence, tokenizer):
        calls.append("validate")
        assert evidence["frozen"] and evidence["final_history"]
        return {"qualified_request_ids": list(evidence["attempts"])}

    suite = SimpleNamespace(
        RESULT_NAME="stopping-result.json",
        prepare_reference=prepare,
        validate_scenario=validate,
    )
    result = {"checks": {}, "cases": [], "scenarios": []}
    await asyncio.wait_for(flow.run("eos", result, suite=suite), 5)
    assert calls == ["prepare", "validate"]
    assert (flow.output / suite.RESULT_NAME).is_file()
    assert not (flow.output / "continuation-result.json").exists()
    assert result["scenarios"][0]["stopping_validation"][
        "qualified_request_ids"
    ]
    assert flow.pumps_drained.is_set() and not flow.reservations


@pytest.mark.asyncio
async def test_failed_preparation_is_saved_before_traffic(flow_factory):
    flow = flow_factory()

    async def fail(*args):
        raise ValueError("No qualified EOS probe")

    suite = SimpleNamespace(
        RESULT_NAME="stopping-result.json", prepare_reference=fail
    )
    result = {"checks": {}, "cases": [], "scenarios": []}
    with pytest.raises(ValueError, match="No qualified EOS"):
        await flow.run("eos", result, suite=suite)
    saved = json.loads((flow.output / suite.RESULT_NAME).read_text())
    assert "No qualified EOS" in saved["scenarios"][0]["preparation_error"]
    assert not flow.requests and not flow.reservations
