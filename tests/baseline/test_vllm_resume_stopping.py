"""Compare resumed stopping with pinned upstream CPU state machines."""

import copy
from types import SimpleNamespace

import pytest
from test_vllm_resume_adapter import (
    _metadata,
    _source_module,
    _state,
)
from test_vllm_resume_adapter import (
    pinned as pinned,
)
from test_vllm_resume_adapter import (
    scheduler_types as scheduler_types,
)

from sllm.backends import vllm_resume_adapter as adapter

STOP_CASES = [
    ("ABENDtail", "END", 0, 12, 5, "AB", "stop"),
    ("ENDXENDtail", "END", 3, 12, 7, "ENDX", "stop"),
    ("AB结束tail", "结束", 0, 12, 4, "AB", "stop"),
    ("ABABA!", "ABA", 0, 12, 3, "", "stop"),
    ("ABENDtail", "END", 0, 3, 3, "ABE", "length"),
]


@pytest.mark.parametrize("case", STOP_CASES)
@pytest.mark.parametrize("include_stop", [False, True])
@pytest.mark.parametrize("flat", [False, True])
def test_stop_boundaries_match_uninterrupted_output(
    pinned, case, include_stop, flat
):
    text, stop, minimum, limit, count, visible, reason = case
    params = pinned.sampling.SamplingParams(
        stop=[stop],
        min_tokens=minimum,
        max_tokens=limit,
        include_stop_str_in_output=include_stop,
        logprobs=1,
        flat_logprobs=flat,
    )
    tokens = list(map(ord, text))

    def advance(state, token):
        matched = state.detokenizer.update([token], False)
        state.logprobs_processor.logprobs.append(
            {token: pinned.logprobs.Logprob(-0.5, 1, chr(token))}
        )
        state.logprobs_processor.cumulative_logprob -= 0.5
        return matched

    def output(state, finish=None, stopped=None):
        return pinned.output.RequestOutput(
            "public",
            "P",
            [80],
            None,
            [state._new_completion_output([], finish, stopped)],
            finish is not None,
        )

    control = _state(pinned, params)
    for index, token in enumerate(tokens[:count]):
        matched = advance(control, token)
        assert matched == (
            stop if reason == "stop" and index == count - 1 else None
        )
    expected = output(control, reason, stop if reason == "stop" else None)
    expected_choice = expected.outputs[0]
    assert expected_choice.text == visible + (
        stop if include_stop and reason == "stop" else ""
    )
    assert expected_choice.token_ids == tokens[:count]

    # Every possible unfinished split, plus a chain that migrates at every token.
    schedules = [(cut,) for cut in range(count)] + [tuple(range(count))]
    for cuts in schedules:
        state = _state(pinned, params)
        for index, token in enumerate(tokens[:count]):
            if index in cuts:
                flushed = output(state, "abort")
                normalized = adapter.normalize_abort_output(
                    flushed, _metadata(params)
                )
                snapshot = adapter.snapshot_request(
                    _metadata(params),
                    normalized,
                    model="m",
                    identity={"revision": "a"},
                )
                state = _state(
                    pinned,
                    adapter._decode_sampling(snapshot["sampling_params"]),
                )
                adapter._restore_frontend(
                    state, snapshot, snapshot["choices"][0]
                )
            matched = advance(state, token)
            assert matched == (
                stop if reason == "stop" and index == count - 1 else None
            )
        actual = output(state, reason, stop if reason == "stop" else None)
        actual_snapshot = adapter.snapshot_request(
            _metadata(params), actual, model="m", identity={"revision": "a"}
        )
        expected_snapshot = adapter.snapshot_request(
            _metadata(params), expected, model="m", identity={"revision": "a"}
        )
        assert actual_snapshot == expected_snapshot
        assert actual.outputs[0].cumulative_logprob == -0.5 * count
        assert adapter.child_request_ids(actual_snapshot, "next") == []


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "options,continuation,maximum,status,stop_reason",
    [
        ({}, [2], 32, "FINISHED_STOPPED", None),
        ({"stop_token_ids": [9]}, [9], 32, "FINISHED_STOPPED", 9),
        ({"ignore_eos": True}, [2, 7, 8], 32, "FINISHED_LENGTH_CAPPED", None),
        (
            {"ignore_eos": True, "stop_token_ids": [9]},
            [9],
            32,
            "FINISHED_STOPPED",
            9,
        ),
        ({}, [7, 8, 9], 32, "FINISHED_LENGTH_CAPPED", None),
        ({}, [7], 4, "FINISHED_LENGTH_CAPPED", None),
    ],
)
def test_restored_core_uses_original_eos_stop_and_length_limits(
    pinned,
    scheduler_types,
    monkeypatch,
    async_scheduling,
    options,
    continuation,
    maximum,
    status,
    stop_reason,
):
    request_module, module, _ = scheduler_types
    checks = _source_module(
        monkeypatch,
        pinned.root,
        "vllm/v1/core/sched/utils.py",
        "_pinned_stop",
        ["check_stop"],
        {"RequestStatus": request_module.RequestStatus},
    )
    params = pinned.sampling.SamplingParams(
        max_tokens=5, min_tokens=2, **options
    )
    params.update_from_generation_config({}, eos_token_id=2)
    history = [5, 6]
    control = request_module.Request(
        "control", [80], copy.deepcopy(params), None
    )
    control.append_output_token_ids(history)
    restored_params = adapter._decode_sampling(adapter._encode_sampling(params))
    restored_params.extra_args = {
        adapter.RESUME_KEY: {"schema_version": 1, "output_token_ids": history}
    }
    restored = request_module.Request("restored", [80], restored_params, None)
    scheduler = module.ResumeScheduler(
        SimpleNamespace(
            use_v2_model_runner=True,
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        )
    )
    scheduler.add_request(restored)
    assert restored.max_tokens == control.max_tokens == 5
    assert restored.sampling_params.min_tokens == 2
    for index, token in enumerate(continuation):
        for request in (control, restored):
            request.append_output_token_ids(token)
        stopped = checks.check_stop(control, maximum)
        assert checks.check_stop(restored, maximum) == stopped
        assert stopped == (index == len(continuation) - 1)
    assert (
        restored.status
        == control.status
        == getattr(request_module.RequestStatus, status)
    )
    assert restored.stop_reason == control.stop_reason == stop_reason
    assert restored.output_token_ids == control.output_token_ids
    assert restored.num_prompt_tokens == control.num_prompt_tokens == 1
