"""Debug history access must not change migration or shutdown ownership."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_legacy_contract import legacy  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_debug", [False, True])
async def test_completed_tokens_are_available_only_with_debug_enabled(
    legacy, trace_debug
):
    backend = legacy({"trace_debug": trace_debug})
    backend.engine.release.set()
    for index in range(3):
        await backend.generate({"request_id": f"done-{index}", "prompt": "hi"})
    expected = (
        {f"done-{index}": [10, 11, 20] for index in range(3)}
        if trace_debug
        else {}
    )
    assert await backend.get_debug_tokens() == expected
    assert await backend.get_current_tokens() == []
    assert await backend.request_trace.request_count() == 0
    assert await backend.request_trace.return_all_request_ids() == []
    assert await backend.request_trace.return_all_results() == []


@pytest.mark.asyncio
async def test_history_and_active_snapshots_remain_independent(legacy):
    backend = legacy({"trace_debug": True})
    engine = backend.engine
    engine.release.set()
    await backend.generate({"request_id": "done"})
    engine.release.clear()
    engine.started.clear()
    request = asyncio.create_task(
        backend.generate({"request_id": "active", "input_tokens": [1, 2]})
    )
    await engine.started.wait()
    try:
        assert await backend.get_debug_tokens() == {"done": [10, 11, 20]}
        assert await backend.get_current_tokens() == [[1, 2]]
        assert await backend.request_trace.return_all_request_ids() == [
            "active"
        ]
        history = await backend.get_debug_tokens()
        history["done"].append(999)
        history["invented"] = []
        active = await backend.get_current_tokens()
        active[0].append(999)
        assert await backend.get_debug_tokens() == {"done": [10, 11, 20]}
        assert await backend.get_current_tokens() == [[1, 2]]
    finally:
        engine.release.set()
        await request
    assert await backend.get_debug_tokens() == {
        "done": [10, 11, 20],
        "active": [10, 11, 20],
    }
    assert await backend.get_current_tokens() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("close", ["stop", "shutdown"])
async def test_retained_history_does_not_block_close_or_get_aborted(
    legacy, close
):
    backend = legacy({"trace_debug": True})
    engine = backend.engine
    engine.release.set()
    engine.abort = AsyncMock()
    await backend.generate({"request_id": "done"})
    await asyncio.wait_for(getattr(backend, close)(), timeout=1)
    assert engine.close_calls == 1
    engine.abort.assert_not_called()
    assert await backend.get_debug_tokens() == {"done": [10, 11, 20]}
    assert await backend.get_current_tokens() == []


@pytest.mark.asyncio
async def test_cancel_before_output_has_empty_debug_tokens(legacy):
    backend = legacy({"trace_debug": True})
    request = asyncio.create_task(
        backend.generate({"request_id": "cancelled", "input_tokens": [1, 2]})
    )
    await backend.engine.started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert await backend.get_debug_tokens() == {"cancelled": []}
    assert await backend.get_current_tokens() == []
    await asyncio.wait_for(backend.stop(), timeout=1)


@pytest.mark.asyncio
async def test_failure_preserves_last_observed_tokens_for_diagnosis(legacy):
    backend = legacy({"trace_debug": True})
    backend.engine.release.set()
    backend.engine.fail_after_output = True
    assert "error" in await backend.generate({"request_id": "failed"})
    assert await backend.get_debug_tokens() == {"failed": [10, 11, 20]}
    assert await backend.get_current_tokens() == []


@pytest.mark.asyncio
async def test_history_uses_prompt_and_first_choice_without_exposing_output(
    legacy,
):
    backend = legacy({"trace_debug": True})
    backend.request_history["multiple"] = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[
            SimpleNamespace(token_ids=[3, 4]),
            SimpleNamespace(token_ids=[5, 6]),
        ],
    )
    backend.request_history["pooling"] = SimpleNamespace(
        prompt_token_ids=[7, 8], outputs=SimpleNamespace(embedding=[0.1, 0.2])
    )
    backend.request_history["no-choice"] = SimpleNamespace(
        prompt_token_ids=None, outputs=[]
    )
    history = await backend.get_debug_tokens()
    assert history == {
        "multiple": [1, 2, 3, 4],
        "pooling": [7, 8],
        "no-choice": [],
    }
    history["multiple"].clear()
    assert backend.request_history["multiple"].outputs[0].token_ids == [3, 4]


@pytest.mark.asyncio
async def test_active_output_is_not_published_as_history_until_finished(legacy):
    backend = legacy({"trace_debug": True})
    engine = backend.engine
    engine.release.set()
    engine.pause_after_output = True
    request = asyncio.create_task(backend.generate({"request_id": "active"}))
    await engine.started.wait()
    try:
        assert await backend.get_current_tokens() == [[10, 11, 20]]
        assert await backend.get_debug_tokens() == {}
    finally:
        engine.finish.set()
        await request
    assert await backend.get_debug_tokens() == {"active": [10, 11, 20]}
