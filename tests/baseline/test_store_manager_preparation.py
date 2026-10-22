"""CPU-only StoreManager integration contracts; no Ray or native Store runtime."""

import asyncio
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def store_module(monkeypatch):
    ray = types.ModuleType("ray")
    modules = {
        "ray": ray,
        "sllm.hardware_info_collector": types.SimpleNamespace(
            collect_all_info=None
        ),
        "sllm.logger": types.SimpleNamespace(init_logger=logging.getLogger),
        "sllm.model_downloader": types.SimpleNamespace(
            VllmModelDownloader=object,
            download_lora_adapter=None,
            download_transformers_model=None,
        ),
        "sllm.utils": types.SimpleNamespace(
            get_worker_nodes=lambda: {"node": {"address": "127.0.0.1"}},
        ),
        "sllm_store.client": types.SimpleNamespace(SllmStoreClient=object),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "sllm/store_manager.py"
    spec = importlib.util.spec_from_file_location("store_manager_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_downloader_actor_is_killed_on_every_exit(store_module, outcome):
    calls = []
    killed = []
    started = asyncio.Event()

    async def run(*args, **kwargs):
        calls.append((args, kwargs))
        started.set()
        if outcome == "failure":
            raise ValueError("prepare failed")
        if outcome == "cancel":
            await asyncio.Event().wait()
        return {"manifest": "verified"}

    actor = types.SimpleNamespace(
        download_vllm_model=types.SimpleNamespace(remote=run),
    )
    factory = types.SimpleNamespace(remote=lambda: actor)
    factory.options = lambda **kwargs: factory
    store_module.ray.remote = lambda cls: factory
    store_module.ray.kill = lambda actor, **kwargs: killed.append(
        (actor, kwargs)
    )
    manager = store_module.StoreManager()
    snapshot = {"digest": "frozen"}
    task = asyncio.create_task(
        manager.download_vllm_model(
            "logical",
            "source",
            "node",
            1,
            2,
            "bfloat16",
            revision="commit",
            source_snapshot=snapshot,
        )
    )
    if outcome == "success":
        assert await task == {"manifest": "verified"}
    elif outcome == "failure":
        with pytest.raises(ValueError, match="prepare failed"):
            await task
    else:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert calls == [
        (
            ("logical", "source", "bfloat16", 2),
            {"revision": "commit", "source_snapshot": snapshot},
        )
    ]
    assert killed == [(actor, {"no_restart": True})]


def local_store(module, responses):
    store = module.SllmLocalStore.__new__(module.SllmLocalStore)
    store.lock = asyncio.Lock()
    store.disk_models = {}
    calls = []
    responses = iter(responses)

    def register(path):
        calls.append(path)
        return next(responses)

    store.client = types.SimpleNamespace(register_model=register)
    return store, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("tp", [0, -1, True, 1.5, "2", None])
async def test_local_registration_rejects_invalid_tp(store_module, tp):
    store, calls = local_store(store_module, [10, 20])
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        await store.register_model(
            "logical", "vllm", {"tensor_parallel_size": tp}
        )
    assert calls == []
    assert store.disk_models == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [-1, 0, True, None, 1.5, "10"])
async def test_invalid_rank_size_does_not_publish_metadata(store_module, size):
    store, calls = local_store(store_module, [10, size])
    with pytest.raises(ValueError, match="rank_1"):
        await store.register_model(
            "logical", "vllm", {"tensor_parallel_size": 2}
        )
    assert len(calls) == 2
    assert store.disk_models == {}


@pytest.mark.asyncio
async def test_local_registration_publishes_all_ranks_after_validation(
    store_module,
):
    store, calls = local_store(store_module, [10, 20])
    assert (
        await store.register_model(
            "logical", "vllm", {"tensor_parallel_size": 2}
        )
        == 30
    )
    assert calls == [
        os.path.join("vllm", "logical", f"rank_{rank}") for rank in range(2)
    ]
    assert store.disk_models == {"logical": (calls, 30)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype_config,expected",
    [
        ({"dtype": "bfloat16"}, "bfloat16"),
        ({"torch_dtype": "float32"}, "float32"),
        ({"dtype": "float16", "torch_dtype": "float16"}, "float16"),
        ({}, "float16"),
    ],
)
async def test_register_prepares_before_registering_and_forwards_source(
    store_module,
    dtype_config,
    expected,
):
    manager = store_module.StoreManager()
    events = []

    async def prepare(*args, **kwargs):
        assert manager.model_info == {}
        events.append(("prepare", args, kwargs))

    async def register(*args):
        assert events[0][0] == "prepare"
        events.append(("register", args))
        return 30

    manager.download_vllm_model = prepare
    manager.local_servers["node"] = types.SimpleNamespace(
        register_model=register
    )
    snapshot = {"digest": "frozen"}
    config = {
        "pretrained_model_name_or_path": "actual/source",
        "revision": "commit",
        "source_snapshot": snapshot,
        "tensor_parallel_size": 2,
        **dtype_config,
    }
    await manager.register(
        {"model": "logical", "backend": "vllm", "backend_config": config}
    )
    assert events[0] == (
        "prepare",
        ("logical", "actual/source", "node", 1, 2, expected),
        {"revision": "commit", "source_snapshot": snapshot},
    )
    assert manager.model_info == {"logical": 30}
    assert manager.model_storage_info == {"logical": {"node": True}}


@pytest.mark.asyncio
async def test_conflicting_dtype_aliases_fail_before_preparation(store_module):
    manager = store_module.StoreManager()
    manager.local_servers["node"] = object()
    with pytest.raises(ValueError, match="dtype"):
        await manager.register(
            {
                "model": "logical",
                "backend": "vllm",
                "backend_config": {
                    "dtype": "bfloat16",
                    "torch_dtype": "float16",
                },
            }
        )
    assert manager.model_info == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["prepare", "register"])
async def test_failed_registration_can_retry(store_module, failure_at):
    manager = store_module.StoreManager()
    attempts = []

    async def prepare(*args, **kwargs):
        attempts.append("prepare")
        if failure_at == "prepare" and attempts.count("prepare") == 1:
            raise ValueError("prepare failed")

    async def register(*args):
        attempts.append("register")
        if failure_at == "register" and attempts.count("register") == 1:
            raise ValueError("registration failed")
        return 30

    manager.download_vllm_model = prepare
    manager.local_servers["node"] = types.SimpleNamespace(
        register_model=register
    )
    config = {"model": "logical", "backend": "vllm"}
    with pytest.raises(ValueError):
        await manager.register(config)
    assert manager.model_info == {}
    assert not manager.model_storage_info.get("logical")
    await manager.register(config)
    assert manager.model_info == {"logical": 30}
    assert manager.model_storage_info == {"logical": {"node": True}}


@pytest.mark.asyncio
async def test_vllm_duplicate_registration_returns_existing_size(store_module):
    store, calls = local_store(store_module, [10])
    assert await store.register_model("logical", "vllm", {}) == 10
    assert await store.register_model("logical", "vllm", {}) == 10
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_invalid_tp_is_rejected_for_existing_model(store_module):
    store, calls = local_store(store_module, [10])
    await store.register_model("logical", "vllm", {})
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        await store.register_model(
            "logical", "vllm", {"tensor_parallel_size": 0}
        )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_multinode_retry_preserves_registered_size(store_module):
    manager = store_module.StoreManager()
    store_module.get_worker_nodes = lambda: {
        "first": {"address": "first"},
        "second": {"address": "second"},
    }
    first, calls = local_store(store_module, [10])
    attempts = []

    async def second_register(*args):
        attempts.append("register")
        if len(attempts) == 1:
            raise ValueError("registration failed")
        return 10

    async def prepare(*args, **kwargs):
        return {"manifest": "verified"}

    manager.local_servers = {
        "first": first,
        "second": types.SimpleNamespace(register_model=second_register),
    }
    manager.download_vllm_model = prepare
    config = {
        "model": "logical",
        "backend": "vllm",
        "placement_config": {
            "local_disk": ["first", "second"],
        },
    }
    with pytest.raises(ValueError, match="registration failed"):
        await manager.register(config)
    assert manager.model_info == {}
    assert manager.model_storage_info == {"logical": {"first": True}}
    await manager.register(config)
    assert len(calls) == 1
    assert manager.model_info == {"logical": 10}
    assert manager.model_storage_info == {
        "logical": {"first": True, "second": True}
    }
