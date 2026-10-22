"""Native calls are strict doubles: no CUDA module/service is imported."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def store_module(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = object
    native = types.ModuleType("sllm_store._C")
    for name in (
        "allocate_cuda_memory",
        "get_cuda_memory_handles",
        "get_device_uuid_map",
        "restore_tensors",
        "save_tensors",
    ):
        setattr(
            native, name, lambda *args: pytest.fail("unexpected native call")
        )
    client = types.ModuleType("sllm_store.client")
    client.SllmStoreClient = object
    maps = types.ModuleType("sllm_store.device_map_utils")
    maps._expand_tensor_name = lambda *_: {}
    utils = types.ModuleType("sllm_store.utils")
    utils.calculate_device_memory = lambda *_: {}
    utils.calculate_tensor_device_offsets = lambda *_: ({}, {})
    for name, value in {
        "torch": torch,
        "sllm_store._C": native,
        "sllm_store.client": client,
        "sllm_store.device_map_utils": maps,
        "sllm_store.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    sys.path.insert(0, str(Path(__file__).parents[2] / "sllm_store"))
    try:
        spec = importlib.util.spec_from_file_location(
            "store_load_contract_probe",
            Path(__file__).parents[2] / "sllm_store/sllm_store/torch.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_confirm_failure_does_not_return_unconfirmed_tensors(
    store_module, monkeypatch
):
    module = store_module
    monkeypatch.setattr(
        module,
        "load_dict_non_blocking",
        lambda *a, **kw: ("replica", {"raw": object()}),
    )
    client = types.SimpleNamespace(confirm_model_loaded=lambda *_: False)
    monkeypatch.setattr(module, "SllmStoreClient", lambda *_: client)
    with pytest.raises(RuntimeError, match="confirm"):
        module.load_dict("model/rank_0", {"": 0})


def test_private_store_address_is_forwarded_for_load_and_confirm(
    store_module, monkeypatch
):
    module = store_module
    observed = []
    expected = {"raw": object()}

    def load(*args, **kwargs):
        observed.append(kwargs["server_address"])
        return "replica", expected

    def client(address):
        observed.append(address)
        return types.SimpleNamespace(confirm_model_loaded=lambda *_: None)

    monkeypatch.setattr(module, "load_dict_non_blocking", load)
    monkeypatch.setattr(module, "SllmStoreClient", client)
    result = module.load_dict(
        "model/rank_0", {"": 0}, server_address="127.0.0.1:19071"
    )
    assert result is expected
    assert observed == ["127.0.0.1:19071", "127.0.0.1:19071"]
