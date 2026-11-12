"""CPU guards for separately approved transparent-continuation qualification."""

import ast
import asyncio
import builtins
import copy
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_migration_qualification import approval as _approval
from test_migration_qualification import qualification as replay_qualification

approval = _approval

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "baseline"
SCRIPT = SCRIPTS / "qualify_continuation.py"


def load_qualification(monkeypatch):
    assert (
        SCRIPT.is_file()
    ), "The separately approved continuation entry point is missing"
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "continuation_qualification_probe", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qualification(monkeypatch):
    return load_qualification(monkeypatch)


@pytest.fixture
def continuation_approval(approval):
    return {**approval, "scope": "stage-d2-continuation", "unit": "d2-test"}


def validate(module, approval):
    return module.validate_approval(
        approval,
        {"CUDA_VISIBLE_DEVICES": ",".join(approval["gpu_uuids"])},
        1000,
        "0::/system.slice/"
        + approval["unit"].removesuffix(".service")
        + ".service\n",
    )


def successful_result(module):
    return {
        "checks": dict.fromkeys(module.REQUIRED_CHECKS, True),
        "cleanup_completed": True,
        "cleanup_errors": [],
        "transparent_continuation": True,
        "cases": [
            {
                "request_id": scenario,
                "scenario": scenario,
                "validated": True,
                "validation": {
                    "request_id": scenario,
                    "choice_count": count,
                    "prompt_tokens": 2,
                    "completion_tokens": 3 * count,
                    "frozen_output_tokens": [2] * count,
                    "resumed_choice_indices": list(range(count)),
                    "full_tokens_equal": True,
                    "source_had_output": True,
                },
            }
            for scenario, count in (("greedy", 1), ("seeded-multi", 2))
        ],
    }


def test_imports_are_stdlib_only_including_every_import_alias():
    assert SCRIPT.is_file(), "Continuation qualification script must exist"
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert all(
        name.split(".")[0] in sys.stdlib_module_names for name in imports
    )


def test_import_and_old_approval_rejection_never_load_gpu_libraries(
    monkeypatch,
    approval,
):
    original = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "vllm", "ray", "sllm", "sllm_store"}:
            attempted.append(name)
            raise AssertionError("Heavy import before approval: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load_qualification(monkeypatch)
    with pytest.raises(ValueError):
        validate(module, approval)
    assert attempted == []


def test_new_scope_is_accepted_without_mutating_approval(
    qualification,
    continuation_approval,
):
    before = copy.deepcopy(continuation_approval)
    validate(qualification, continuation_approval)
    assert continuation_approval == before


def test_numerical_profile_requires_explicit_environment_before_resource_start(
    qualification,
    continuation_approval,
    monkeypatch,
    tmp_path,
):
    config = {
        **continuation_approval,
        "numerical_profile": "batch-invariant-v1",
    }
    calls = prepare_cli(monkeypatch, tmp_path, config)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    with pytest.raises(ValueError, match="startup environment"):
        qualification.main()
    assert not calls and not Path(config["output"]).exists()


@pytest.mark.parametrize("name", ["default", "batch-invariant-v1"])
def test_main_passes_only_selected_profile_to_private_workers(
    qualification,
    continuation_approval,
    monkeypatch,
    tmp_path,
    name,
):
    import qualify_migration

    config = {**continuation_approval, "numerical_profile": name}
    prepare_cli(monkeypatch, tmp_path, config)
    monkeypatch.setenv(
        "VLLM_BATCH_INVARIANT", "1" if name == "batch-invariant-v1" else "0"
    )
    received = []

    async def inspect(*args, **kwargs):
        received.append(kwargs["worker_env"])

    monkeypatch.setattr(qualify_migration, "qualify", inspect)
    # No real Ray/engine construction, and this intentionally cannot pass.
    assert qualification.main() == 1
    assert received == (
        [{"VLLM_BATCH_INVARIANT": "1"}]
        if name == "batch-invariant-v1"
        else [{}]
    )


def test_numerical_success_requires_all_four_valid_worker_observations(
    qualification,
):
    from test_continuation_profile import worker_report

    result = successful_result(qualification)
    uuids = [
        worker_report()["gpu_uuid"],
        "GPU-22222222-2222-2222-2222-222222222222",
    ]
    result["approval"] = {
        "numerical_profile": "batch-invariant-v1",
        "gpu_uuids": uuids,
    }
    assert not qualification.runtime_passed(result)
    report = worker_report()
    result["numerical_observations"] = [
        {
            "scenario": scenario,
            "role": role,
            "node": str(index),
            "validated": True,
            "expected_uuid": uuids[index],
            "expected_cgroup": report["cgroup"],
            "workers": [{**copy.deepcopy(report), "gpu_uuid": uuids[index]}],
        }
        for scenario in qualification.SCENARIOS
        for index, role in enumerate(("source", "target"))
    ]
    assert qualification.runtime_passed(result)
    for index in range(4):
        invalid = copy.deepcopy(result)
        invalid["numerical_observations"].pop(index)
        assert not qualification.runtime_passed(invalid)
        invalid = copy.deepcopy(result)
        invalid["numerical_observations"][index]["workers"][0][
            "batch_invariant_initialized"
        ] = False
        assert not qualification.runtime_passed(invalid)


@pytest.mark.asyncio
async def test_numerical_inspection_uses_pinned_named_worker_rpc(qualification):
    from sllm.backends.vllm_qualification import NumericalQualificationMixin

    calls = []

    async def collective_rpc(method, **kwargs):
        calls.append((method, kwargs))
        return [{"worker": "observed"}]

    actor = SimpleNamespace(
        engine=SimpleNamespace(collective_rpc=collective_rpc)
    )
    assert await NumericalQualificationMixin.get_numerical_runtime(actor) == [
        {"worker": "observed"}
    ]
    assert calls == [("sllm_numerical_runtime", {"timeout": 30})]


def test_old_and_new_scopes_cannot_authorize_each_other(
    qualification,
    approval,
    continuation_approval,
):
    validate(replay_qualification, approval)
    with pytest.raises(ValueError):
        validate(qualification, approval)
    with pytest.raises(ValueError):
        validate(replay_qualification, continuation_approval)


@pytest.mark.parametrize("approved", [False, None, 0, 1, "true"])
def test_new_scope_cannot_create_approval(
    qualification,
    continuation_approval,
    approved,
):
    config = {**continuation_approval, "approved": approved}
    before = copy.deepcopy(config)
    with pytest.raises(ValueError):
        validate(qualification, config)
    assert config == before


def test_missing_approval_is_not_implicitly_granted(
    qualification,
    continuation_approval,
):
    config = dict(continuation_approval)
    config.pop("approved")
    with pytest.raises(ValueError):
        validate(qualification, config)
    assert "approved" not in config


def test_short_ray_root_requires_d2_run_stamp(
    qualification,
    continuation_approval,
):
    config = {
        **continuation_approval,
        "output": "/tmp/evidence/gpu-stage-d2-20261111-120000",
        "ray_directory": "/tmp/sd2-20261111-120000",
        "unit": "sllm-d2-migration-20261111-120000",
    }
    validate(qualification, config)
    for field, value in (
        ("ray_directory", "/tmp/sd1-20261111-120000"),
        ("output", "/tmp/evidence/gpu-stage-d1-20261111-120000"),
        ("unit", "sllm-d1-migration-20261111-120000"),
    ):
        with pytest.raises(ValueError):
            validate(qualification, {**config, field: value})


def prepare_cli(monkeypatch, tmp_path, config):
    import qualify_migration

    config = {**config, "deadline_epoch": time.time() + 600}
    approval_file = tmp_path / "approval.json"
    approval_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--approval", str(approval_file)]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(config["gpu_uuids"]))
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::/system.slice/" + config["unit"] + ".service\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    calls = []

    async def forbidden_execution(*args, **kwargs):
        calls.append(True)
        raise AssertionError(
            "A rejected CLI invocation must not start resources"
        )

    monkeypatch.setattr(qualify_migration, "qualify", forbidden_execution)
    return calls


def test_main_rejects_old_approval_before_starting_or_writing_output(
    qualification,
    approval,
    monkeypatch,
    tmp_path,
):
    calls = prepare_cli(monkeypatch, tmp_path, approval)
    with pytest.raises(ValueError):
        qualification.main()
    assert calls == []
    assert not Path(approval["output"]).exists()


def test_main_preserves_existing_result_before_resource_start(
    qualification,
    continuation_approval,
    monkeypatch,
    tmp_path,
):
    calls = prepare_cli(monkeypatch, tmp_path, continuation_approval)
    directory = Path(continuation_approval["output"])
    directory.mkdir()
    result = directory / "continuation-result.json"
    result.write_text("previous evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        qualification.main()
    assert calls == []
    assert result.read_text(encoding="utf-8") == "previous evidence"


def test_pass_requires_every_check_and_complete_cleanup(qualification):
    result = successful_result(qualification)
    assert qualification.runtime_passed(result)
    for key in qualification.REQUIRED_CHECKS:
        missing = copy.deepcopy(result)
        missing["checks"].pop(key)
        assert not qualification.runtime_passed(missing)
        for value in (False, 1, "true"):
            invalid = copy.deepcopy(result)
            invalid["checks"][key] = value
            assert not qualification.runtime_passed(invalid)
    for field, value in (
        ("cleanup_completed", False),
        ("cleanup_completed", 1),
        ("cleanup_errors", ["source remained alive"]),
        ("error", "request ended with abort"),
        ("transparent_continuation", False),
        ("transparent_continuation", 1),
    ):
        assert not qualification.runtime_passed({**result, field: value})


@pytest.mark.parametrize("cases", [None, [], [{}], [{"validated": False}]])
def test_empty_or_unvalidated_cases_cannot_pass(qualification, cases):
    result = successful_result(qualification)
    result["cases"] = cases
    assert not qualification.runtime_passed(result)


def test_every_case_needs_exact_full_token_validation(qualification):
    result = successful_result(qualification)
    for field, value in (
        ("validated", False),
        ("validated", 1),
        ("validation", {}),
    ):
        invalid = copy.deepcopy(result)
        invalid["cases"].append(
            {**copy.deepcopy(result["cases"][0]), field: value}
        )
        assert not qualification.runtime_passed(invalid)
    for value in (False, 1, "true"):
        invalid = copy.deepcopy(result)
        invalid["cases"][0]["validation"]["full_tokens_equal"] = value
        assert not qualification.runtime_passed(invalid)
    invalid = copy.deepcopy(result)
    invalid["cases"][0]["validation"]["resumed_choice_indices"] = []
    assert not qualification.runtime_passed(invalid)


@pytest.mark.parametrize(
    "field,value", [("error", "failed"), ("aborted", True)]
)
def test_case_failure_cannot_be_hidden_by_validated_flag(
    qualification, field, value
):
    result = successful_result(qualification)
    result["cases"][0][field] = value
    assert not qualification.runtime_passed(result)


@pytest.mark.parametrize("scenario", ["greedy", "seeded-multi"])
def test_both_generation_scenarios_are_required(qualification, scenario):
    result = successful_result(qualification)
    result["cases"] = [
        item for item in result["cases"] if item["scenario"] == scenario
    ]
    assert not qualification.runtime_passed(result)


@pytest.mark.parametrize("scenario", ["greedy", "seeded-multi"])
@pytest.mark.parametrize("had_output", [False, None, 1])
def test_each_scenario_requires_confirmed_source_generation(
    qualification,
    scenario,
    had_output,
):
    result = successful_result(qualification)
    for case in result["cases"]:
        if case["scenario"] == scenario:
            case["validation"]["source_had_output"] = had_output
    assert not qualification.runtime_passed(result)


def test_all_zero_frozen_prefixes_do_not_qualify_real_continuation(
    qualification,
):
    result = successful_result(qualification)
    for case in result["cases"]:
        report = case["validation"]
        report["frozen_output_tokens"] = [0] * report["choice_count"]
        report["source_had_output"] = False
    assert not qualification.runtime_passed(result)


def test_only_finished_choice_prefix_does_not_qualify_active_continuation(
    qualification,
):
    result = successful_result(qualification)
    report = result["cases"][1]["validation"]
    report.update(
        frozen_output_tokens=[5, 0],
        resumed_choice_indices=[1],
        source_had_output=False,
    )
    assert not qualification.runtime_passed(result)


def test_zero_prefix_additional_case_does_not_hide_valid_active_case(
    qualification,
):
    result = successful_result(qualification)
    extra = copy.deepcopy(result["cases"][0])
    extra["request_id"] = "late"
    extra["validation"].update(
        request_id="late",
        frozen_output_tokens=[0],
        source_had_output=False,
    )
    result["cases"].append(extra)
    assert qualification.runtime_passed(result)


def test_legacy_replay_result_does_not_claim_continuation(qualification):
    legacy = {
        "checks": dict.fromkeys(replay_qualification.REQUIRED_CHECKS, True),
        "cleanup_completed": True,
        "cleanup_errors": [],
        "source_request_terminals": [{"finish_reason": "abort"}],
        "transparent_continuation": False,
    }
    assert not qualification.runtime_passed(legacy)


class Client:
    def __init__(
        self, *, status=200, error=None, payload=None, decode_error=False
    ):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.status = status
        self.error = error
        self.payload = payload
        self.decode_error = decode_error

    async def post(self, path, *, json):
        self.calls.append((path, copy.deepcopy(json)))
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        payload = self.payload or {
            "id": json["request_id"],
            "choices": [{"finish_reason": "stop"}],
        }

        def response_json():
            if self.decode_error:
                raise ValueError("Invalid response JSON")
            return payload

        def raise_for_status():
            if self.status != 200:
                raise RuntimeError("HTTP failed")

        return SimpleNamespace(
            status_code=self.status,
            json=response_json,
            text=repr(payload),
            raise_for_status=raise_for_status,
        )


async def joined(operation):
    return await asyncio.wait_for(operation, 2)


@pytest.mark.asyncio
async def test_send_once_registers_before_http_and_preserves_original_request(
    qualification,
):
    client, attempts = Client(), {}
    request = {
        "request_id": "original",
        "model": "model",
        "options": {"seed": 7},
    }
    before = copy.deepcopy(request)
    task = asyncio.create_task(
        qualification.send_once(client, request, attempts)
    )
    try:
        await joined(client.started.wait())
        assert attempts["original"]["request"] == before
        assert attempts["original"]["started_epoch"] > 0
        request["options"]["seed"] = 99
        assert attempts["original"]["request"] == before
        with pytest.raises(ValueError):
            await qualification.send_once(client, before, attempts)
        assert len(client.calls) == 1
        client.release.set()
        result = await joined(task)
        assert result["id"] == "original"
        record = attempts["original"]
        assert record["http_status"] == 200
        assert record["response"] == result
        assert record["ended_epoch"] >= record["started_epoch"]
        with pytest.raises(ValueError):
            await qualification.send_once(client, before, attempts)
        assert len(client.calls) == 1
    finally:
        client.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["transport", "status", "cancel", "api-error", "decode"]
)
async def test_failure_or_cancel_never_erases_attempt_or_retries(
    qualification, failure
):
    client = Client(
        status=503 if failure == "status" else 200,
        error=OSError("transport failed") if failure == "transport" else None,
        payload={"error": "engine failure"} if failure == "api-error" else None,
        decode_error=failure == "decode",
    )
    attempts = {}
    request = {"request_id": "failed", "model": "model"}
    task = asyncio.create_task(
        qualification.send_once(client, request, attempts)
    )
    try:
        await joined(client.started.wait())
        if failure == "cancel":
            task.cancel()
        else:
            client.release.set()
        outcome = (await joined(asyncio.gather(task, return_exceptions=True)))[
            0
        ]
        assert isinstance(outcome, BaseException)
        assert attempts["failed"]["error"]
        assert (
            attempts["failed"]["ended_epoch"]
            >= attempts["failed"]["started_epoch"]
        )
        with pytest.raises(ValueError):
            await qualification.send_once(client, request, attempts)
        assert len(client.calls) == 1
    finally:
        client.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_pump_stop_keeps_current_http_call_and_sends_no_next_request(
    qualification,
):
    client, attempts, stopped = Client(), {}, asyncio.Event()
    body = {"model": "model", "max_tokens": 3}
    before = copy.deepcopy(body)
    task = asyncio.create_task(
        qualification.http_pump(client, body, "pump", stopped, attempts)
    )
    try:
        await joined(client.started.wait())
        stopped.set()
        await asyncio.sleep(0)
        assert not task.done()
        client.release.set()
        await joined(task)
        assert len(client.calls) == len(attempts) == 1
        request_id = next(iter(attempts))
        assert request_id.startswith("pump-")
        assert client.calls[0][1]["request_id"] == request_id
        assert attempts[request_id]["response"]["id"] == request_id
        assert body == before
        stopped.clear()
        with pytest.raises(ValueError):
            await qualification.http_pump(
                client, body, "pump", stopped, attempts
            )
        assert len(client.calls) == 1
    finally:
        stopped.set()
        client.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_already_stopped_pump_sends_nothing(qualification):
    client, attempts, stopped = Client(), {}, asyncio.Event()
    stopped.set()
    await joined(
        qualification.http_pump(
            client, {"model": "model"}, "pump", stopped, attempts
        )
    )
    assert client.calls == []
    assert attempts == {}


def test_backend_history_includes_every_choice_without_exposing_mutable_state(
    qualification,
):
    choices = [
        SimpleNamespace(
            index=index,
            token_ids=[20 + index],
            text=f"choice {index}",
            logprobs=[
                {
                    20 + index: SimpleNamespace(
                        logprob=-0.5, rank=1, decoded_token=str(20 + index)
                    )
                }
            ],
            cumulative_logprob=-0.5,
            finish_reason="stop",
            stop_reason="done",
        )
        for index in range(2)
    ]
    actor = SimpleNamespace(
        request_history={
            "finished": SimpleNamespace(
                prompt_token_ids=[10, 11], outputs=choices
            ),
            "no-output": None,
        }
    )
    history = qualification.backend_history(actor)
    record = next(item for item in history if item["request_id"] == "finished")
    assert record["prompt_token_ids"] == [10, 11]
    assert [choice["index"] for choice in record["choices"]] == [0, 1]
    assert [choice["token_ids"] for choice in record["choices"]] == [[20], [21]]
    assert all(
        choice["cumulative_logprob"] == -0.5 for choice in record["choices"]
    )
    assert [choice["text"] for choice in record["choices"]] == [
        "choice 0",
        "choice 1",
    ]
    record["prompt_token_ids"].clear()
    record["choices"][0]["token_ids"].append(999)
    assert record["choices"][0]["logprobs"] == [
        {"20": {"logprob": -0.5, "rank": 1, "decoded_token": "20"}}
    ]
    record["choices"][0]["logprobs"][0]["20"]["logprob"] = -99
    assert actor.request_history["finished"].prompt_token_ids == [10, 11]
    assert choices[0].token_ids == [20]
    assert choices[0].logprobs[0][20].logprob == -0.5


def test_migration_trace_is_disabled_without_serializing_payload(caplog):
    from sllm.backends.vllm_migration import VllmMigration

    backend = VllmMigration.__new__(VllmMigration)
    backend.trace_debug = False
    with caplog.at_level(logging.INFO, logger="ray"):
        backend._trace_migration("frozen", "epoch", requests=object())
    assert "SLLM migration evidence:" not in caplog.text


def test_migration_trace_records_exact_pid_epoch_and_json_payload(
    caplog, monkeypatch
):
    from sllm.backends.vllm_migration import VllmMigration

    backend = VllmMigration.__new__(VllmMigration)
    backend.trace_debug = True
    monkeypatch.setattr(logging.getLogger("ray"), "propagate", True)
    requests = [
        {"request_id": "original", "choices": [{"token_ids": [20, 21]}]}
    ]
    before = copy.deepcopy(requests)
    with caplog.at_level(logging.INFO, logger="ray"):
        backend._trace_migration("frozen", "epoch", requests=requests)
    marker = "SLLM migration evidence: "
    messages = [
        record.getMessage()
        for record in caplog.records
        if marker in record.getMessage()
    ]
    assert len(messages) == 1
    assert json.loads(messages[0].split(marker, 1)[1]) == {
        "schema_version": 1,
        "event": "frozen",
        "pid": os.getpid(),
        "epoch": "epoch",
        "requests": before,
    }
    assert requests == before
