"""CPU lifecycle contracts, including admission before any engine output."""

import asyncio

import pytest
from test_legacy_contract import legacy  # noqa: F401


@pytest.mark.asyncio
async def test_stop_waits_for_admission_before_first_output(legacy):
    backend = legacy()
    engine = backend.engine
    request = asyncio.create_task(backend.generate({"input_tokens": [1, 2]}))
    await engine.started.wait()
    assert await backend.get_current_tokens() == [[1, 2]]
    stop = asyncio.create_task(backend.stop())
    await asyncio.sleep(0)
    assert not stop.done()
    assert await backend.generate({}) == {"error": "Engine is not running"}
    engine.release.set()
    await request
    await asyncio.wait_for(stop, 1)
    assert engine.close_calls == 1


@pytest.mark.asyncio
async def test_input_untouched_and_duplicate_id_rejected(legacy):
    backend = legacy()
    data = {"model": "logical", "prompt": "hello", "request_id": "same"}
    before = dict(data)
    request = asyncio.create_task(backend.generate(data))
    await backend.engine.started.wait()
    assert "Duplicate" in (await backend.generate(data))["error"]
    assert data == before
    assert len(backend.engine.calls) == 1
    backend.engine.release.set()
    await request
    assert await backend.request_trace.request_count() == 0


@pytest.mark.asyncio
async def test_cancellation_aborts_and_unregisters(legacy):
    backend = legacy()
    request = asyncio.create_task(backend.generate({"request_id": "cancel"}))
    await backend.engine.started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert backend.engine.release.is_set()
    assert await backend.get_current_tokens() == []
    await asyncio.wait_for(backend.stop(), 1)


@pytest.mark.asyncio
async def test_shutdown_escalates_graceful_stop_and_joins(legacy):
    backend = legacy()
    engine = backend.engine
    request = asyncio.create_task(backend.generate({}))
    await engine.started.wait()
    stop = asyncio.create_task(backend.stop())
    await asyncio.sleep(0)
    assert not stop.done()
    await asyncio.wait_for(
        asyncio.gather(backend.shutdown(), backend.shutdown(), stop), 1
    )
    await asyncio.gather(request, return_exceptions=True)
    assert engine.close_calls == 1
    assert await backend.request_trace.request_count() == 0
    await backend.shutdown()
    assert engine.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_close_waiter_does_not_cancel_close(legacy):
    backend = legacy()
    engine = backend.engine
    request = asyncio.create_task(backend.generate({}))
    await engine.started.wait()
    stop = asyncio.create_task(backend.stop())
    await asyncio.sleep(0)
    stop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop
    engine.release.set()
    await request
    await asyncio.wait_for(backend.stop(), 1)
    assert engine.close_calls == 1


@pytest.mark.asyncio
async def test_failed_initialization_never_marks_running(legacy, monkeypatch):
    backend = legacy()
    backend.status = type(backend.status).UNINITIALIZED
    backend.engine = None
    from sllm.backends import vllm_compat

    def fail(args):
        raise RuntimeError("init failed")

    monkeypatch.setattr(vllm_compat, "create", fail)
    with pytest.raises(RuntimeError, match="init failed"):
        await backend.init_backend()
    assert backend.status.name == "UNINITIALIZED"
    assert backend.engine is None


@pytest.mark.asyncio
async def test_pooling_generation_model_reports_unsupported(legacy):
    assert "pooling" in (await legacy().encode({"input": ["hello"]}))["error"]


@pytest.mark.asyncio
async def test_abort_failure_preserves_error_and_cleans_admission(legacy):
    backend = legacy()
    backend.engine.release.set()
    backend.engine.fail_after_output = True

    async def failed_abort(request_id):
        raise RuntimeError("abort failed")

    backend.engine.abort = failed_abort
    result = await backend.generate({})
    assert result == {"error": "engine failure"}
    assert await backend.get_current_tokens() == []
    await backend.stop()


@pytest.mark.asyncio
async def test_encode_unique_ids_active_cleanup_and_response(legacy):
    import types

    backend = legacy({"task": "embed"})
    ids = []

    async def encode(inputs, params, request_id):
        ids.append(request_id)
        assert await backend.request_trace.request_count() > 0
        yield types.SimpleNamespace(
            prompt_token_ids=[1, 2],
            outputs=types.SimpleNamespace(embedding=[0.25, 0.75]),
        )

    backend.engine.encode = encode
    results = await asyncio.gather(
        backend.encode({"input": ["a", "b"]}), backend.encode({"input": "c"})
    )
    assert len(ids) == len(set(ids)) == 3
    assert results[0]["data"][0]["embedding"] == [0.25, 0.75]
    assert results[0]["usage"]["query_tokens"] == 4
    assert await backend.get_current_tokens() == []
    await backend.stop()


@pytest.mark.asyncio
async def test_encode_converts_real_v031_pooling_output_and_tokens(
    legacy, monkeypatch
):
    import sys
    import types

    backend = legacy({"task": "embed"})
    module = sys.modules["vllm"]
    converted = []
    seen = []

    def from_base(cls, output):
        converted.append(output)
        return types.SimpleNamespace(
            prompt_token_ids=output.prompt_token_ids,
            outputs=types.SimpleNamespace(
                embedding=output.outputs.data.tolist()
            ),
        )

    monkeypatch.setattr(
        module.EmbeddingRequestOutput,
        "from_base",
        classmethod(from_base),
        raising=False,
    )

    async def encode(inputs, params, request_id):
        seen.append(inputs)
        yield types.SimpleNamespace(
            prompt_token_ids=[1, 2],
            outputs=types.SimpleNamespace(
                data=types.SimpleNamespace(tolist=lambda: [0.1, 0.2])
            ),
        )

    backend.engine.encode = encode
    result = await backend.encode({"input": [1, 2]})
    assert seen == [{"prompt_token_ids": [1, 2]}]
    assert len(converted) == 1
    assert result["data"][0]["embedding"] == [0.1, 0.2]
    assert await backend.get_current_tokens() == []
