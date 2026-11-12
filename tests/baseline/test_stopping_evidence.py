"""CPU rejection cases for the separately approved stopping suite."""

import copy
import importlib
from pathlib import Path

import pytest
from test_continuation_evidence import choice


@pytest.fixture
def stopping(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "scripts/baseline")
    )
    return importlib.import_module("stopping_evidence")


def history(tokens, text, stop_reason=None):
    item = choice(0, tokens, text, "stop")
    item["stop_reason"] = stop_reason
    return {"request_id": "request", "choices": [item]}


def test_eos_identity_and_terminal(stopping):
    identity = stopping.eos_identity(2, {"eos_token_id": [2, 3]})
    assert identity == {"primary": 2, "ids": [2, 3]}
    spec = {"scenario": "eos", "eos": identity}
    stopping.validate_terminal(history([9] * 31 + [2], "text"), spec)
    stopping.validate_terminal(history([9] * 31 + [3], "text", 3), spec)


@pytest.mark.parametrize(
    "tokens,reason",
    [
        ([9] * 32, None),
        ([9] * 31 + [2], 9),
        ([2], None),
        ([9] * 383 + [2], None),
        ([2] + [9] * 30 + [2], None),
    ],
)
def test_false_eos_is_rejected(stopping, tokens, reason):
    spec = {"scenario": "eos", "eos": stopping.eos_identity(2, {})}
    with pytest.raises(ValueError):
        stopping.validate_terminal(history(tokens, "text", reason), spec)


@pytest.mark.parametrize("value", [True, -1, [], [2, True], "2"])
def test_invalid_eos_identity(stopping, value):
    with pytest.raises(ValueError):
        stopping.eos_identity(2, {"eos_token_id": value})


def test_stop_selection_and_crossing(stopping):
    tokens = list(range(1000, 1384))
    decode = lambda ids: "".join(chr(token) for token in ids)
    probe = history(tokens, decode(tokens))
    probe["choices"][0]["finish_reason"] = "length"
    spec = stopping.stop_spec("stop-string-exclude", probe, decode)
    assert (spec["start"], spec["end"]) == (32, 320)
    final = history(tokens[:320], decode(tokens[:32]), spec["stop"])
    stopping.validate_terminal(final, spec)
    params = stopping.parameters(spec["scenario"])
    params["stop"] = [spec["stop"]]
    frozen = {
        "request_id": "request",
        "sampling_params": params,
        "choices": [choice(0, tokens[:150], "")],
    }
    record = stopping.boundary(frozen, final, spec, decode(tokens[:150]))
    assert record["qualified"] and record["frozen_text_length"] == 150
    for count in [0, 32, 320]:
        frozen["choices"][0] = choice(0, tokens[:count], "")
        assert not stopping.boundary(
            frozen, final, spec, decode(tokens[:count])
        )["qualified"]
    with pytest.raises(ValueError):
        stopping.boundary(frozen, final, spec, "unrelated")


@pytest.mark.parametrize(
    "mutate",
    ["length", "wrong-text", "wrong-stop", "empty", "duplicate-choice"],
)
def test_invalid_stop_control(stopping, mutate):
    tokens = list(range(1000, 1384))
    decode = lambda ids: "".join(chr(token) for token in ids)
    probe = history(tokens, decode(tokens))
    probe["choices"][0]["finish_reason"] = "length"
    spec = stopping.stop_spec("stop-string-include", probe, decode)
    final = history(tokens[:320], decode(tokens[:320]), spec["stop"])
    stopping.validate_terminal(final, spec)
    if mutate == "length":
        final["choices"][0]["finish_reason"] = "length"
    if mutate == "wrong-text":
        final["choices"][0]["text"] = "wrong"
    if mutate == "wrong-stop":
        final["choices"][0]["stop_reason"] = "wrong"
    if mutate == "empty":
        final["choices"][0]["token_ids"] = []
    if mutate == "duplicate-choice":
        final["choices"] *= 2
    with pytest.raises(ValueError):
        stopping.validate_terminal(final, spec)


def test_eos_frozen_state_must_match_configuration(stopping):
    spec = {
        "scenario": "eos",
        "eos": stopping.eos_identity(2, {"eos_token_id": [2, 3]}),
    }
    params = {
        **stopping.parameters("eos"),
        "_eos_token_id": 2,
        "stop_token_ids": [3],
    }
    frozen = {
        "request_id": "request",
        "sampling_params": params,
        "choices": [choice(0, [9] * 12, "prefix")],
    }
    final = history([9] * 31 + [2], "prefix text")
    assert stopping.boundary(frozen, final, spec)["qualified"]
    for change in [
        {"stop_token_ids": [7]},
        {"_eos_token_id": 7},
        {"ignore_eos": True},
        {"stop": ["text"]},
        {"max_tokens": 100},
    ]:
        invalid = copy.deepcopy(frozen)
        invalid["sampling_params"].update(change)
        with pytest.raises(ValueError):
            stopping.boundary(invalid, final, spec)


def test_repeated_text_uses_actual_first_match(stopping):
    probe = history([65] * 384, "A" * 384)
    probe["choices"][0]["finish_reason"] = "length"
    spec = stopping.stop_spec(
        "stop-string-exclude", probe, lambda ids: "A" * len(ids)
    )
    assert (spec["selection_start"], spec["selection_end"]) == (32, 320)
    assert (spec["start"], spec["end"]) == (0, 288)
    final = history([65] * 288, "", "A" * 288)
    stopping.validate_terminal(final, spec)
