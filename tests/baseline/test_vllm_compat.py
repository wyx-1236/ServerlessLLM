"""Pinned engine adapter ABI and Store identity projection, without CUDA."""

import json
import sys
import types
from dataclasses import make_dataclass

import pytest
from test_legacy_contract import legacy  # noqa: F401


def test_partial_construction_explicitly_closes_engine_core(monkeypatch):
    from sllm.backends import vllm_compat

    closed = []

    class Engine:
        @classmethod
        def from_engine_args(cls, args):
            return cls(args=args)

        def __init__(self, *, args):
            self.engine_core = types.SimpleNamespace(
                shutdown=lambda: closed.append(args)
            )
            raise RuntimeError("child startup failed")

        def shutdown(self):
            self.engine_core.shutdown()

    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(AsyncLLMEngine=Engine)
    )
    with pytest.raises(RuntimeError, match="child startup failed"):
        vllm_compat.create("args")
    assert closed == ["args"]


@pytest.mark.asyncio
async def test_abort_is_awaited_and_shutdown_is_synchronous():
    from sllm.backends import vllm_compat

    calls = []

    class Engine:
        async def abort(self, request_id, internal=False):
            calls.append((request_id, internal))

        def shutdown(self, timeout=None):
            calls.append(("shutdown", timeout))

    engine = Engine()
    await vllm_compat.abort(engine, "public-id")
    await vllm_compat.close(engine)
    assert calls == [("public-id", False), ("shutdown", None)]


@pytest.fixture
def store_projection(monkeypatch, tmp_path):
    from sllm.backends import vllm_compat

    fields = json.loads(
        (
            __import__("pathlib").Path(__file__).parent
            / "fixtures/vllm-0.31.0-engine-fields.json"
        ).read_text()
    )
    Args = make_dataclass("Args", [(name, object, None) for name in fields])
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(__version__="0.31.0", AsyncEngineArgs=Args),
    )
    identity = {
        "engine_version": "0.31.0",
        "world_size": 1,
        "dtype": "torch.bfloat16",
    }
    path = tmp_path / "vllm" / "logical"
    path.mkdir(parents=True)
    (path / "artifact-manifest.json").write_text(
        json.dumps({"identity": identity})
    )
    verified = []
    monkeypatch.setitem(
        sys.modules,
        "sllm_store.artifact",
        types.SimpleNamespace(
            validate_artifact=lambda path, identity: verified.append(
                (path, identity)
            )
        ),
    )
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))

    def project(**kwargs):
        return vllm_compat.engine_args(
            "logical",
            {
                "model_loader_extra_config": {
                    "store_address": "127.0.0.1:8073"
                },
                **kwargs,
            },
        )

    return project, path, identity, verified


def test_store_projection_validates_and_passes_exact_identity(store_projection):
    project, path, identity, verified = store_projection
    args = project(torch_dtype="bfloat16")
    assert verified == [(path.resolve(), identity)]
    assert args.load_format == "serverless_llm"
    assert args.model == str(path.resolve())
    assert args.dtype == "bfloat16"
    assert args.model_loader_extra_config == {
        "artifact_path": str(path.resolve()),
        "identity": identity,
        "store_address": "127.0.0.1:8073",
        "store_root": str(path.parent.parent.resolve()),
    }


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"tensor_parallel_size": 2}, "TP"),
        ({"dtype": "float16"}, "dtype"),
        ({"model_loader_extra_config": {}}, "store_address"),
        (
            {"model_loader_extra_config": {"store_address": "0.0.0.0:8073"}},
            "store_address",
        ),
        (
            {"model_loader_extra_config": {"store_address": "127.0.0.1:0"}},
            "port",
        ),
        (
            {"model_loader_extra_config": {"candidate_manifest": {}}},
            "extra config",
        ),
    ],
)
def test_store_projection_rejects_conflicting_or_unsafe_config(
    store_projection, kwargs, reason
):
    project, *_ = store_projection
    with pytest.raises(ValueError, match=reason):
        project(**kwargs)


def test_wrong_engine_version_rejected_before_construction(legacy, monkeypatch):
    legacy()
    import vllm

    from sllm.backends import vllm_compat

    monkeypatch.setattr(vllm, "__version__", "0.31.1")
    with pytest.raises(ValueError, match="version"):
        vllm_compat.engine_args("logical", {})
