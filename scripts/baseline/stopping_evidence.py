"""CPU-only construction and rejection rules for bounded stopping evidence."""

SCENARIOS = ("eos", "stop-string-exclude", "stop-string-include")
EOS_PROMPTS = (
    "List the integers from 1 to 20, separated by commas, then finish.",
    "Write a short paragraph of about 60 words explaining how a computer stores data.",
    "Give exactly five short numbered tips for organizing files on a computer.",
)
DECODE_OPTIONS = dict(
    skip_special_tokens=True, spaces_between_special_tokens=True
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameters(scenario):
    require(scenario in SCENARIOS, "Unknown stopping scenario")
    return dict(
        temperature=0,
        n=1,
        min_tokens=32 if scenario == "eos" else 0,
        max_tokens=384,
        logprobs=5,
        ignore_eos=scenario != "eos",
        include_stop_str_in_output=scenario == "stop-string-include",
        **DECODE_OPTIONS,
    )


def eos_identity(primary, generation):
    values = generation.get("eos_token_id", primary)
    values = [values] if type(values) is int else values
    require(
        type(primary) is int and primary >= 0, "Missing primary EOS identity"
    )
    require(
        isinstance(values, list)
        and values
        and all(type(v) is int and v >= 0 for v in values),
        "Invalid generation EOS identity",
    )
    return {"primary": primary, "ids": sorted(set([primary, *values]))}


def one(history):
    choices = history.get("choices")
    require(
        isinstance(choices, list)
        and len(choices) == 1
        and choices[0].get("index") == 0,
        "Expected exactly one choice",
    )
    item = choices[0]
    require(
        isinstance(item.get("token_ids"), list)
        and all(type(v) is int and v >= 0 for v in item["token_ids"]),
        "Invalid output tokens",
    )
    require(isinstance(item.get("text"), str), "Missing output text")
    return item


def stop_spec(scenario, probe, decode):
    require(scenario in SCENARIOS[1:], "Not a stop-string scenario")
    item = one(probe)
    tokens, text = item["token_ids"], item["text"]
    require(
        len(tokens) == 384 and item["finish_reason"] == "length",
        "Stop probe must exhaust the original 384-token budget",
    )
    first, last = decode(tokens[:32]), decode(tokens[:320])
    require(
        text.startswith(first)
        and text.startswith(last)
        and 0 < len(first) < len(last) < len(text),
        "Unreliable decoded token boundaries",
    )
    stop = last[len(first) :]
    start = text.find(stop)
    require(start >= 0, "Selected stop has no match in probe")
    return dict(
        scenario=scenario,
        stop=stop,
        start=start,
        end=start + len(stop),
        selection_start=len(first),
        selection_end=len(last),
        probe_text=text,
        probe_tokens=list(tokens),
        decode_options=dict(DECODE_OPTIONS),
    )


def validate_terminal(history, spec):
    item = one(history)
    tokens = item["token_ids"]
    require(
        item.get("finish_reason") == "stop" and 0 < len(tokens) < 384,
        "Expected natural stop before budget exhaustion",
    )
    if spec["scenario"] == "eos":
        eos = spec["eos"]
        require(
            eos == eos_identity(eos["primary"], {"eos_token_id": eos["ids"]}),
            "Invalid EOS evidence",
        )
        require(
            32 <= len(tokens)
            and tokens[-1] in eos["ids"]
            and not set(tokens[:-1]).intersection(eos["ids"]),
            "Missing terminal EOS or EOS already in prefix",
        )
        expected = None if tokens[-1] == eos["primary"] else tokens[-1]
        require(
            item.get("stop_reason") == expected,
            "Terminal reason does not identify the EOS",
        )
    else:
        require(spec["scenario"] in SCENARIOS[1:], "Unknown terminal scenario")
        start, end, stop, text = (
            spec[k] for k in ("start", "end", "stop", "probe_text")
        )
        require(
            type(start) is int
            and type(end) is int
            and 0 <= start < end < len(text)
            and text[start:end] == stop
            and text.find(stop) == start,
            "Invalid stop interval or first match",
        )
        require(
            spec["decode_options"] == DECODE_OPTIONS,
            "Changed stop decoding options",
        )
        require(
            tokens == spec["probe_tokens"][: len(tokens)],
            "Stop control changed probe tokens",
        )
        boundary = end if spec["scenario"] == "stop-string-include" else start
        require(
            item["text"] == text[:boundary] and item.get("stop_reason") == stop,
            "Stop text or reason differs from selected boundary",
        )
    return item


def boundary(frozen, final, spec, raw_prefix=None):
    initial, completed = one(frozen), validate_terminal(final, spec)
    require(
        frozen["request_id"] == final["request_id"],
        "Unrelated stopping history",
    )
    params = frozen["sampling_params"]
    require(
        all(
            params.get(k) == v for k, v in parameters(spec["scenario"]).items()
        ),
        "Stopping request parameters changed",
    )
    if spec["scenario"] == "eos":
        eos = spec["eos"]
        require(
            not params.get("stop")
            and params.get("_eos_token_id") == eos["primary"]
            and set(params.get("stop_token_ids") or [])
            == set(eos["ids"]) - {eos["primary"]},
            "Frozen EOS configuration differs from model identity",
        )
    else:
        require(
            params.get("stop") == [spec["stop"]],
            "Frozen stop differs from selected string",
        )
        require(
            isinstance(raw_prefix, str)
            and spec["probe_text"].startswith(raw_prefix),
            "Decoded frozen prefix is unrelated to probe",
        )
    old, new = initial["token_ids"], completed["token_ids"]
    require(new[: len(old)] == old, "Target lost frozen tokens")
    qualified = initial.get("finish_reason") is None and 0 < len(old) < len(new)
    record = dict(
        request_id=frozen["request_id"],
        choice_index=0,
        frozen_tokens=len(old),
        final_tokens=len(new),
    )
    if spec["scenario"] == "eos":
        qualified = qualified and not set(old).intersection(spec["eos"]["ids"])
        record["terminal_eos"] = new[-1]
    else:
        qualified = qualified and spec["start"] < len(raw_prefix) < spec["end"]
        record.update(
            frozen_text_length=len(raw_prefix),
            stop_start=spec["start"],
            stop_end=spec["end"],
        )
    record["qualified"] = bool(qualified)
    return record


def audit_reference(evidence):
    from continuation_evidence import _canonical, _response

    state = evidence["stopping"]
    spec = state["spec"]
    require(spec["scenario"] == evidence["scenario"], "Wrong stopping spec")
    probes = state["probes"]
    require(
        0 < len(probes) <= (3 if spec["scenario"] == "eos" else 1),
        "Invalid probe count",
    )
    selected = [
        p
        for p in probes
        if p.get("request", {}).get("request_id") == state["selected_probe"]
    ]
    require(
        len(selected) == 1
        and not selected[0].get("error")
        and not selected[0].get("rejection"),
        "Missing or duplicate selected probe",
    )
    probe, control = selected[0], state["control"]
    require(
        not control.get("error")
        and probe["request"]["request_id"] != control["request"]["request_id"],
        "Control is not independent",
    )
    require(
        _canonical(control["history"])
        == _canonical(evidence["reference_history"])
        and _canonical(control["response"])
        == _canonical(evidence["reference"]),
        "Unrelated reference evidence",
    )
    body = {k: v for k, v in control["request"].items() if k != "request_id"}
    require(
        all(body.get(k) == v for k, v in parameters(spec["scenario"]).items()),
        "Control parameters differ from scenario",
    )
    require(
        isinstance(body["input_tokens"], list)
        and 0 < len(body["input_tokens"]) <= 128,
        "Control prompt outside bounds",
    )
    for receipt in (probe, control):
        _response(
            receipt["response"],
            receipt["history"],
            body["model"],
            1,
            receipt["request"]["request_id"],
        )
        require(
            receipt["history"]["prompt_token_ids"]
            == receipt["request"]["input_tokens"],
            "Control prompt history mismatch",
        )
    probe_body = {
        k: v for k, v in probe["request"].items() if k != "request_id"
    }
    if spec["scenario"] == "eos":
        origin = state["eos_source"]
        require(
            spec["eos"]
            == eos_identity(
                origin["tokenizer_eos_token_id"], origin["generation_config"]
            ),
            "EOS spec differs from model identity",
        )
        require(
            probe_body == body
            and not body.get("stop")
            and not body.get("stop_token_ids"),
            "EOS control changed parameters",
        )
        validate_terminal(probe["history"], spec)
        require(
            _canonical(probe["history"]["choices"])
            == _canonical(control["history"]["choices"]),
            "Independent EOS controls differ",
        )
    else:
        raw = one(probe["history"])
        require(
            len(raw["token_ids"]) == 384
            and raw["finish_reason"] == "length"
            and raw["token_ids"] == spec["probe_tokens"]
            and raw["text"] == spec["probe_text"],
            "Stop spec differs from probe",
        )
        require(
            body.get("stop") == [spec["stop"]]
            and probe_body == {k: v for k, v in body.items() if k != "stop"},
            "Stop control changed parameters",
        )
    validate_terminal(evidence["reference_history"], spec)
    return body


def audit_scenario(evidence):
    """Recheck raw saved comparisons and qualified request ownership."""
    from continuation_evidence import compare_reference_prefixes, validate_case

    body = audit_reference(evidence)
    spec = evidence["stopping"]["spec"]
    require(
        compare_reference_prefixes(
            evidence["source_prefix_history"], evidence["reference_history"]
        )["comparable"],
        "Source prefix differs before migration",
    )
    frozen = evidence["frozen"]["requests"]
    prepared = evidence["prepared"]["requests"]
    ids = [item["request_id"] for item in frozen]
    require(
        ids and len(ids) == len(set(ids)),
        "Missing or duplicate frozen requests",
    )
    require(
        len(prepared) == len(ids)
        and {item["request_id"] for item in prepared} == set(ids),
        "Missing or duplicate prepared requests",
    )
    targets = {item["request_id"]: item for item in prepared}
    records = []
    for initial in frozen:
        rid = initial["request_id"]
        attempt, final = (
            evidence["attempts"][rid],
            evidence["final_history"][rid],
        )
        require(
            {k: v for k, v in attempt["request"].items() if k != "request_id"}
            == body,
            "Migrated request differs from control",
        )
        require(
            attempt.get("http_status") == 200 and not attempt.get("error"),
            "Original HTTP caller failed",
        )
        validate_case(
            attempt["request"],
            attempt["response"],
            initial,
            targets[rid],
            final,
            evidence["reference"],
            evidence["reference_history"],
            require_generated_prefix=False,
        )
        decoded = evidence["stopping"].get("decoded_prefixes", {}).get(rid)
        if spec["scenario"] != "eos":
            require(
                decoded is not None
                and decoded["token_ids"] == one(initial)["token_ids"],
                "Missing decoded frozen prefix",
            )
        records.append(
            boundary(
                initial,
                final,
                spec,
                None if decoded is None else decoded["text"],
            )
        )
    qualified = [
        record["request_id"] for record in records if record["qualified"]
    ]
    require(qualified, "No request crossed the stopping boundary")
    return {"qualified_request_ids": qualified, "boundaries": records}
