"""Reject weak or mismatched evidence of same-request continuation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / "scripts/baseline/continuation_evidence.py"
SPEC = importlib.util.spec_from_file_location("continuation_evidence", SOURCE)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def choice(index, tokens, text, reason=None):
    return {
        "index": index,
        "token_ids": tokens,
        "text": text,
        "finish_reason": reason,
        "stop_reason": tokens[-1] if reason == "stop" and tokens else None,
        "logprobs": [
            {token: {"logprob": -0.1, "rank": 1, "decoded_token": str(token)}}
            for token in tokens
        ],
        "cumulative_logprob": -0.1 * len(tokens),
    }


def response(history):
    completion = sum(len(item["token_ids"]) for item in history["choices"])
    return {
        "id": history["request_id"],
        "model": "model",
        "object": "chat.completion",
        "choices": [
            {
                "index": item["index"],
                "message": {"role": "assistant", "content": item["text"]},
                "finish_reason": item["finish_reason"],
                "logprobs": json.loads(json.dumps(item["logprobs"])),
            }
            for item in history["choices"]
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": completion,
            "total_tokens": completion + 2,
        },
    }


@pytest.fixture
def case():
    request = {
        "request_id": "inflight",
        "model": "model",
        "input_tokens": [10, 11],
        "max_tokens": 4,
        "n": 2,
        "temperature": 0,
        "logprobs": 1,
    }
    frozen = {
        "schema_version": 1,
        "request_id": "inflight",
        "model": "model",
        "identity": {"artifact": {"revision": "fixed"}, "tokenizer": "abc"},
        "prompt": None,
        "prompt_token_ids": [10, 11],
        "sampling_params": {
            "max_tokens": 4,
            "n": 2,
            "temperature": 0.0,
            "logprobs": 1,
            "_eos_token_id": 32,
        },
        "prompt_logprobs": None,
        "choices": [choice(0, [20, 21], "AB"), choice(1, [30], "C")],
    }
    history = {
        "request_id": "inflight",
        "prompt_token_ids": [10, 11],
        "choices": [
            choice(0, [20, 21, 22, 23], "ABCD", "length"),
            choice(1, [30, 31, 32], "CDE", "stop"),
        ],
    }
    reference = {**copy.deepcopy(history), "request_id": "reference"}
    return dict(
        request=request,
        response=response(history),
        source_snapshot=frozen,
        target_snapshot=copy.deepcopy(frozen),
        final_history=history,
        reference_response=response(reference),
        reference_history=reference,
    )


def test_complete_tokens_choices_logprobs_and_usage_match(case):
    before = copy.deepcopy(case)
    result = evidence.validate_case(**case)
    assert result["request_id"] == "inflight"
    assert result["choice_count"] == 2
    assert result["completion_tokens"] == 7
    assert result["frozen_output_tokens"] == [2, 1]
    assert result["resumed_choice_indices"] == [0, 1]
    assert result["full_tokens_equal"] is True
    assert case == before


def test_one_finished_choice_is_not_resampled(case):
    case["source_snapshot"]["choices"][1] = copy.deepcopy(
        case["final_history"]["choices"][1]
    )
    case["target_snapshot"] = copy.deepcopy(case["source_snapshot"])
    result = evidence.validate_case(**case)
    assert result["resumed_choice_indices"] == [0]


def test_zero_output_admission_is_checked_but_not_active_resume_evidence(case):
    for item in case["source_snapshot"]["choices"]:
        item.update(choice(item["index"], [], ""))
    case["target_snapshot"] = copy.deepcopy(case["source_snapshot"])
    result = evidence.validate_case(**case, require_generated_prefix=False)
    assert result["source_had_output"] is False
    assert result["resumed_choice_indices"] == [0, 1]
    assert result["completion_tokens"] == 7


def test_zero_prefix_still_requires_every_unfinished_choice_to_progress(case):
    for item in case["source_snapshot"]["choices"]:
        item.update(choice(item["index"], [], ""))
    case["target_snapshot"] = copy.deepcopy(case["source_snapshot"])
    for key in ("final_history", "reference_history"):
        case[key]["choices"][0] = choice(0, [], "", "stop")
    case["response"] = response(case["final_history"])
    case["reference_response"] = response(case["reference_history"])
    with pytest.raises(ValueError):
        evidence.validate_case(**case, require_generated_prefix=False)


def test_mixed_prefixes_require_one_actual_active_continuation(case):
    case["source_snapshot"]["choices"][1] = choice(1, [], "")
    case["target_snapshot"] = copy.deepcopy(case["source_snapshot"])
    result = evidence.validate_case(**case)
    assert result["source_had_output"] is True
    assert result["resumed_choice_indices"] == [0, 1]


def test_single_choice_and_omitted_n_use_original_default(case):
    case["request"].pop("n")
    for key in ("source_snapshot", "target_snapshot"):
        case[key]["sampling_params"]["n"] = 1
        case[key]["choices"] = case[key]["choices"][:1]
    for key in ("final_history", "reference_history"):
        case[key]["choices"] = case[key]["choices"][:1]
    case["response"] = response(case["final_history"])
    case["reference_response"] = response(case["reference_history"])
    assert evidence.validate_case(**case)["choice_count"] == 1


@pytest.mark.parametrize("which", ["response", "reference_response"])
@pytest.mark.parametrize("key", ["error", "preempted", "_sllm_migration"])
def test_internal_or_error_envelopes_are_not_success(case, which, key):
    case[which][key] = "failure"
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


@pytest.mark.parametrize(
    "which,key,value",
    [
        ("response", "id", "new-http-request"),
        ("response", "model", "other"),
        ("source_snapshot", "request_id", "another"),
        ("source_snapshot", "model", "other"),
        ("source_snapshot", "schema_version", 2),
        ("source_snapshot", "schema_version", True),
        ("target_snapshot", "identity", {"tokenizer": "wrong"}),
        ("final_history", "request_id", "warmup"),
        ("reference_history", "request_id", "wrong-reference"),
        ("final_history", "prompt_token_ids", [10, 11, 20]),
    ],
)
def test_identity_and_snapshot_mismatch_is_rejected(case, which, key, value):
    case[which][key] = value
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


@pytest.mark.parametrize("which", ["source_snapshot", "target_snapshot"])
@pytest.mark.parametrize(
    "field,value", [("max_tokens", 2), ("n", 1), ("temperature", 1)]
)
def test_original_effective_parameters_are_preserved(case, which, field, value):
    case[which]["sampling_params"][field] = value
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


@pytest.mark.parametrize(
    "which",
    [
        "source_snapshot",
        "target_snapshot",
        "final_history",
        "reference_history",
        "response",
        "reference_response",
    ],
)
def test_no_missing_choice(case, which):
    case[which]["choices"].pop()
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_prefix",
        "duplicated_prefix",
        "text_only_match",
        "no_progress",
        "abort",
        "wrong_logprob",
        "wrong_stop",
        "excess_budget",
        "bad_index",
        "bool_token",
        "missing_logprobs",
    ],
)
def test_final_token_and_finish_contracts(case, mutation):
    final = case["final_history"]["choices"][0]
    if mutation == "missing_prefix":
        final["token_ids"] = [22, 23]
    elif mutation == "duplicated_prefix":
        final["token_ids"] = [20, 21, 20, 21, 22, 23]
    elif mutation == "text_only_match":
        final["token_ids"][-1] = 999
    elif mutation == "no_progress":
        final.update(choice(0, [20, 21], "AB", "stop"))
    elif mutation == "abort":
        final["finish_reason"] = "abort"
    elif mutation == "wrong_logprob":
        final["logprobs"][0][20]["logprob"] = -5
    elif mutation == "wrong_stop":
        final["stop_reason"] = 20
    elif mutation == "excess_budget":
        final["token_ids"].append(24)
    elif mutation == "bad_index":
        final["index"] = 1
    elif mutation == "bool_token":
        final["token_ids"][0] = True
    elif mutation == "missing_logprobs":
        final["logprobs"] = None
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


@pytest.mark.parametrize("which", ["response", "reference_response"])
@pytest.mark.parametrize(
    "field", ["prompt_tokens", "completion_tokens", "total_tokens"]
)
def test_usage_must_count_original_prompt_once(case, which, field):
    case[which]["usage"][field] += 1
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


@pytest.mark.parametrize(
    "kind", ["already_finished", "no_output", "budget_spent"]
)
def test_freeze_must_have_active_generated_output(case, kind):
    for item in case["source_snapshot"]["choices"]:
        if kind == "already_finished":
            item["finish_reason"] = "stop"
        elif kind == "no_output":
            item.update(choice(item["index"], [], ""))
        else:
            item.update(choice(item["index"], [20, 21, 22, 23], "ABCD"))
    case["target_snapshot"] = copy.deepcopy(case["source_snapshot"])
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


def test_unseeded_random_sampling_does_not_claim_exact_equivalence(case):
    case["request"]["temperature"] = 1.0
    for key in ("source_snapshot", "target_snapshot"):
        case[key]["sampling_params"]["temperature"] = 1.0
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


def test_seeded_sampling_can_be_compared_when_the_actual_results_match(case):
    case["request"].update(temperature=1.0, seed=42)
    for key in ("source_snapshot", "target_snapshot"):
        case[key]["sampling_params"].update(temperature=1.0, seed=42)
    assert evidence.validate_case(**case)["full_tokens_equal"]


@pytest.mark.parametrize("field", ["text", "logprobs", "prompt_logprobs"])
def test_frozen_committed_values_cannot_be_replaced_on_the_target(case, field):
    if field == "text":
        case["source_snapshot"]["choices"][0]["text"] = "wrong"
    elif field == "logprobs":
        case["source_snapshot"]["choices"][0]["logprobs"][0][20]["logprob"] = -9
    else:
        for key in ("source_snapshot", "target_snapshot"):
            case[key].pop("prompt_logprobs")
    case["target_snapshot"] = copy.deepcopy(case["source_snapshot"])
    with pytest.raises(ValueError):
        evidence.validate_case(**case)


@pytest.fixture
def events(case):
    return [
        dict(
            schema_version=1,
            event="frozen",
            pid=101,
            epoch="handoff",
            requests=[case["source_snapshot"]],
        ),
        dict(
            schema_version=1,
            event="prepared",
            pid=202,
            epoch="handoff",
            requests=[case["target_snapshot"]],
        ),
        dict(
            schema_version=1,
            event="resume",
            pid=202,
            epoch="handoff",
            request_id="inflight",
        ),
    ]


def test_log_parser_and_exact_handoff_allow_identical_duplicate_logs(events):
    text = "unrelated log\n" + "\n".join(
        "(actor pid=101) INFO SLLM migration evidence: " + json.dumps(event)
        for event in events + events
    )
    parsed = evidence.parse_events(text)
    frozen, prepared, resumes = evidence.find_handoff(
        parsed, 101, 202, "handoff"
    )
    assert frozen["pid"] == 101 and prepared["pid"] == 202
    assert resumes == [events[2]]


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        "null",
        '{"schema_version":2}',
        '{"schema_version":true}',
        '{"schema_version":1,"schema_version":2}',
    ],
)
def test_matching_evidence_marker_never_silently_drops_bad_json(payload):
    with pytest.raises(ValueError):
        evidence.parse_events("SLLM migration evidence: " + payload)


@pytest.mark.parametrize(
    "source_pid,target_pid,epoch",
    [
        (999, 202, "handoff"),
        (101, 999, "handoff"),
        (101, 202, "wrong"),
        (101, 101, "handoff"),
    ],
)
def test_handoff_cannot_use_other_actor_or_epoch(
    events, source_pid, target_pid, epoch
):
    with pytest.raises(ValueError):
        evidence.find_handoff(events, source_pid, target_pid, epoch)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_frozen",
        "missing_prepared",
        "missing_resume",
        "conflict",
        "wrong_request",
    ],
)
def test_handoff_requires_complete_nonconflicting_observed_events(
    events, mutation
):
    if mutation == "missing_frozen":
        events.pop(0)
    elif mutation == "missing_prepared":
        events.pop(1)
    elif mutation == "missing_resume":
        events.pop(2)
    elif mutation == "conflict":
        changed = copy.deepcopy(events[0])
        changed["requests"][0]["sampling_params"]["max_tokens"] = 2
        events.append(changed)
    else:
        events[2]["request_id"] = "fresh-http"
    with pytest.raises(ValueError):
        evidence.find_handoff(events, 101, 202, "handoff")
