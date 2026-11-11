"""Pure-data checks for the approved same-HTTP-request migration probe."""

import copy
import json

MARKER = "SLLM migration evidence: "
TERMINAL_REASONS = {"length", "stop"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    """Normalize JSON's string dictionary keys, including token logprobs."""
    if isinstance(value, dict):
        result = {str(key): _canonical(item) for key, item in value.items()}
        _require(len(result) == len(value), "Ambiguous JSON dictionary keys")
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate evidence JSON field: {key}")
        result[key] = value
    return result


def _requests(record):
    requests = record.get("requests")
    _require(isinstance(requests, list), "Missing batch snapshots")
    result = {}
    for item in requests:
        _require(isinstance(item, dict), "Invalid batch snapshot")
        request_id = item.get("request_id")
        _require(
            isinstance(request_id, str)
            and request_id
            and request_id not in result,
            "Missing or duplicate batch request ID",
        )
        result[request_id] = item
    return result


def _event(record):
    _require(isinstance(record, dict), "Evidence event must be an object")
    _require(
        type(record.get("schema_version")) is int
        and record["schema_version"] == 1,
        "Unsupported evidence schema",
    )
    _require(
        type(record.get("pid")) is int and record["pid"] > 0,
        "Evidence requires an actual actor PID",
    )
    _require(
        isinstance(record.get("epoch"), str) and record["epoch"],
        "Evidence requires a migration epoch",
    )
    kind = record.get("event")
    _require(kind in {"frozen", "prepared", "resume"}, "Unknown evidence event")
    if kind == "resume":
        _require(
            isinstance(record.get("request_id"), str) and record["request_id"],
            "Resume event requires its original request ID",
        )
    else:
        _requests(record)


def parse_events(text):
    """Read exact marked JSON events; corrupted marked evidence is an error."""
    result = []
    for line in text.splitlines():
        if MARKER not in line:
            continue
        record = json.loads(
            line.split(MARKER, 1)[1], object_pairs_hook=_json_object
        )
        _event(record)
        result.append(record)
    return result


def find_handoff(events, source_pid, target_pid, epoch):
    """Bind a complete observed handoff to two actual actors and one epoch."""
    _require(
        type(source_pid) is int
        and type(target_pid) is int
        and source_pid > 0
        and target_pid > 0
        and source_pid != target_pid,
        "Handoff requires two distinct actual actor PIDs",
    )
    _require(isinstance(epoch, str) and epoch, "Missing handoff epoch")
    batches, resumes = {}, {}
    for event in events:
        _event(event)
        kind = event["event"]
        expected_pid = source_pid if kind == "frozen" else target_pid
        if event["epoch"] != epoch or event["pid"] != expected_pid:
            continue
        key = event["request_id"] if kind == "resume" else kind
        destination = resumes if kind == "resume" else batches
        if key in destination:
            _require(
                _canonical(destination[key]) == _canonical(event),
                "Conflicting duplicate migration evidence",
            )
        else:
            destination[key] = event
    _require(
        set(batches) == {"frozen", "prepared"}, "Incomplete handoff batches"
    )
    frozen = _requests(batches["frozen"])
    prepared = _requests(batches["prepared"])
    _require(
        frozen and set(frozen) == set(prepared), "Mismatched handoff requests"
    )
    _require(
        _canonical(frozen) == _canonical(prepared),
        "Target preparation changed frozen snapshots",
    )
    _require(
        resumes and set(resumes) <= set(frozen), "Missing or unrelated resume"
    )
    return copy.deepcopy(
        (
            batches["frozen"],
            batches["prepared"],
            [resumes[key] for key in sorted(resumes)],
        )
    )


def _tokens(tokens, label):
    _require(
        isinstance(tokens, list)
        and all(type(token) is int and token >= 0 for token in tokens),
        f"Invalid {label} token IDs",
    )
    return tokens


def _choices(container, count, label):
    _require(isinstance(container, dict), f"Missing {label}")
    choices = container.get("choices")
    _require(
        isinstance(choices, list) and len(choices) == count,
        f"Missing {label} choices",
    )
    _require(
        all(
            isinstance(item, dict) and type(item.get("index")) is int
            for item in choices
        )
        and [item["index"] for item in choices] == list(range(count)),
        f"Invalid {label} choice indices",
    )
    return choices


def _history_choice(item, params, *, frozen=False):
    required = {
        "token_ids",
        "text",
        "finish_reason",
        "stop_reason",
        "logprobs",
        "cumulative_logprob",
    }
    _require(required <= item.keys(), "Incomplete choice history")
    tokens = _tokens(item["token_ids"], "output")
    _require(isinstance(item["text"], str), "Invalid output text")
    _require(
        len(tokens) <= params["max_tokens"], "Output exceeds original budget"
    )
    reason = item["finish_reason"]
    _require(
        reason in TERMINAL_REASONS or (frozen and reason is None),
        "Aborted or unfinished choice is not successful completion",
    )
    if reason == "length":
        _require(
            len(tokens) == params["max_tokens"], "Length ended before budget"
        )
    rows = item["logprobs"]
    requested = params.get("logprobs") is not None or params.get(
        "logprob_token_ids"
    )
    _require(
        rows is not None or not requested or not tokens,
        "Missing requested output logprobs",
    )
    if rows is not None:
        _require(
            isinstance(rows, list) and len(rows) == len(tokens),
            "Logprobs do not cover the complete output history",
        )
        for token, row in zip(tokens, rows):
            _require(isinstance(row, dict), "Invalid logprob row")
            normalized = _canonical(row)
            _require(str(token) in normalized, "Generated token has no logprob")
            for value in normalized.values():
                _require(
                    isinstance(value, dict)
                    and {"logprob", "rank", "decoded_token"} <= value.keys(),
                    "Incomplete token logprob",
                )


def _response(response, history, model, count, request_id):
    _require(isinstance(response, dict), "Missing HTTP response body")
    _require(
        not {"error", "preempted", "_sllm_migration"} & response.keys(),
        "An error or internal envelope is not a completed HTTP response",
    )
    _require(response.get("id") == request_id, "HTTP request identity changed")
    _require(response.get("model") == model, "HTTP model identity changed")
    _require(
        response.get("object") == "chat.completion", "Wrong response schema"
    )
    choices = _choices(response, count, "HTTP response")
    for actual, recorded in zip(choices, history["choices"]):
        _require(
            actual.get("message")
            == {"role": "assistant", "content": recorded["text"]}
            and actual.get("finish_reason") == recorded["finish_reason"]
            and "logprobs" in actual
            and _canonical(actual["logprobs"])
            == _canonical(recorded["logprobs"]),
            "HTTP choices disagree with actual backend history",
        )
    prompt = len(history["prompt_token_ids"])
    completion = sum(len(item["token_ids"]) for item in history["choices"])
    expected = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    usage = response.get("usage")
    _require(
        isinstance(usage, dict)
        and usage == expected
        and all(type(usage[key]) is int for key in expected),
        "Usage must count original prompt once and every choice output",
    )


def validate_case(
    request,
    response,
    source_snapshot,
    target_snapshot,
    final_history,
    reference_response,
    reference_history,
    *,
    require_generated_prefix=True,
):
    """Verify a migrated HTTP response against frozen state and full control."""
    request_id, model = request.get("request_id"), request.get("model")
    _require(
        isinstance(request_id, str) and request_id, "Missing HTTP request ID"
    )
    _require(isinstance(model, str) and model, "Missing HTTP model")
    prompt = _tokens(request.get("input_tokens"), "original prompt")
    _require(prompt, "Qualification requires a nonempty exact prompt")
    limit, count = request.get("max_tokens"), request.get("n", 1)
    _require(
        type(limit) is int and limit > 0,
        "An explicit output budget is required",
    )
    _require(type(count) is int and count > 0, "Invalid original choice count")
    _require(
        request.get("temperature", 1.0) == 0
        or type(request.get("seed")) is int,
        "Unseeded random sampling cannot establish exact token equivalence",
    )
    for snapshot in (source_snapshot, target_snapshot):
        _require(isinstance(snapshot, dict), "Missing migration snapshot")
        _require(
            {"prompt_logprobs", "sampling_params", "choices"}
            <= snapshot.keys(),
            "Incomplete frozen request snapshot",
        )
        _require(
            type(snapshot.get("schema_version")) is int
            and snapshot["schema_version"] == 1,
            "Wrong snapshot schema",
        )
        _require(
            snapshot.get("request_id") == request_id
            and snapshot.get("model") == model
            and _tokens(snapshot.get("prompt_token_ids"), "snapshot prompt")
            == prompt,
            "Snapshot no longer describes the original HTTP request",
        )
        _require(
            isinstance(snapshot.get("identity"), dict) and snapshot["identity"],
            "Missing model/tokenizer identity",
        )
    _require(
        _canonical(source_snapshot) == _canonical(target_snapshot),
        "Target snapshot differs from the final frozen state",
    )
    params = source_snapshot.get("sampling_params")
    _require(isinstance(params, dict), "Missing effective sampling parameters")
    _require(
        type(params.get("max_tokens")) is int
        and params["max_tokens"] == limit
        and type(params.get("n")) is int
        and params["n"] == count,
        "Migration changed the original budget or choice count",
    )
    for key, value in request.items():
        if key in {"request_id", "model", "input_tokens", "prompt", "messages"}:
            continue
        actual = params.get(key)
        if key == "stop" and isinstance(value, str):
            value = [value]
        _require(
            _canonical(actual) == _canonical(value),
            f"Migration changed effective parameter {key}",
        )
    frozen = _choices(source_snapshot, count, "frozen")
    final = _choices(final_history, count, "final history")
    reference = _choices(reference_history, count, "reference history")
    _require(
        final_history.get("request_id") == request_id, "Unrelated final history"
    )
    reference_id = reference_history.get("request_id")
    _require(
        isinstance(reference_id, str)
        and reference_id
        and reference_id != request_id,
        "Independent reference request identity required",
    )
    for history in (final_history, reference_history):
        _require(
            _tokens(history.get("prompt_token_ids"), "history prompt")
            == prompt,
            "History changed the original prompt boundary",
        )
    resumed = []
    source_had_output = False
    for initial, completed, control in zip(frozen, final, reference):
        _history_choice(initial, params, frozen=True)
        _history_choice(completed, params)
        _history_choice(control, params)
        old = initial["token_ids"]
        _require(
            completed["token_ids"][: len(old)] == old,
            "Final output lost or changed the frozen prefix",
        )
        _require(
            completed["text"].startswith(initial["text"]),
            "Final output changed committed visible text",
        )
        if initial["logprobs"] is not None:
            _require(
                completed["logprobs"] is not None
                and _canonical(completed["logprobs"][: len(old)])
                == _canonical(initial["logprobs"]),
                "Final output changed committed logprobs",
            )
        if initial["finish_reason"] is not None:
            _require(
                _canonical(initial) == _canonical(completed),
                "An already completed choice was resampled",
            )
        elif len(old) < limit:
            _require(
                len(completed["token_ids"]) > len(old),
                "Unfinished choice never continued on the target",
            )
            resumed.append(initial["index"])
            source_had_output = source_had_output or bool(old)
        _require(
            _canonical(completed) == _canonical(control),
            "Full output tokens/text/logprobs differ from the control",
        )
    _require(resumed, "No unfinished choice continued on the target")
    if require_generated_prefix:
        _require(
            source_had_output,
            "No active generated choice was frozen and continued",
        )
    _response(response, final_history, model, count, request_id)
    _response(reference_response, reference_history, model, count, reference_id)
    return {
        "request_id": request_id,
        "choice_count": count,
        "prompt_tokens": len(prompt),
        "completion_tokens": sum(len(item["token_ids"]) for item in final),
        "frozen_output_tokens": [len(item["token_ids"]) for item in frozen],
        "resumed_choice_indices": resumed,
        "source_had_output": source_had_output,
        "full_tokens_equal": True,
    }
