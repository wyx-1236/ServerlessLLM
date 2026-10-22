"""CPU contracts; vLLM doubles follow frozen v0.31.0 public loader APIs.

Sources: vllm/model_executor/model_loader/{__init__,base_loader,utils}.py,
vllm/model_executor/utils.py in https://github.com/vllm-project/vllm/tree/v0.31.0.
No vLLM/native Store import, model download or CUDA initialization is used.
"""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sllm_store"))


def test_entrypoint_import_is_lightweight():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sllm_store.vllm_plugin; import sys; "
            "assert not any(x in sys.modules for x in "
            "['torch', 'vllm', 'sllm_store._C', 'sllm_store.vllm_loader'])",
        ],
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(
                Path(__file__).resolve().parents[2] / "sllm_store"
            ),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registration_version_and_reentrancy(monkeypatch):
    plugin = importlib.import_module("sllm_store.vllm_plugin")
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.30.0")
    with pytest.raises(RuntimeError, match="0.31.0"):
        plugin.register()
    calls = []
    fake = types.ModuleType("vllm.model_executor.model_loader")
    fake.register_model_loader = lambda name: lambda cls: calls.append(
        (name, cls)
    )
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    loader = types.ModuleType("sllm_store.vllm_loader")
    loader.ServerlessLLMModelLoader = type("ServerlessLLMModelLoader", (), {})
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setattr(plugin.metadata, "version", lambda _: "0.31.0")
    monkeypatch.setattr(plugin, "_registered", False)
    plugin.register()
    plugin.register()
    assert calls == [("serverless_llm", loader.ServerlessLLMModelLoader)]


def test_candidate_validation_keeps_file_and_index_checks(tmp_path):
    from sllm_store.artifact import (
        ArtifactTransaction,
        validate_candidate_artifact,
    )

    identity = dict(
        model="test",
        revision="sha",
        engine="vllm",
        engine_version="0.31.0",
        dtype="bfloat16",
        writer="native-test",
        world_size=1,
    )
    with ArtifactTransaction(tmp_path / "out") as tx:
        rank = tx.path / "rank_0"
        rank.mkdir()
        (rank / "tensor.data_0").write_bytes(bytes(8))
        (rank / "tensor_index.json").write_text(
            json.dumps({"storage_0": [0, 8, [8], [1], "torch.uint8"]})
        )

        def verify(path, candidate):
            assert (
                validate_candidate_artifact(path, identity, candidate)
                == candidate
            )
            (rank / "tensor.data_0").write_bytes(bytes([1]) * 8)
            with pytest.raises(ValueError, match="inventory|digest"):
                validate_candidate_artifact(path, identity, candidate)
            (rank / "tensor.data_0").write_bytes(bytes(8))
            return {"passed": True, "method": "unit-only"}

        tx.commit(identity, verify)


@pytest.fixture
def loader_api(monkeypatch):
    """Only upstream API boundaries are doubled; tensors use actual CPU Torch."""
    torch = pytest.importorskip("torch")
    fake = types.ModuleType("vllm.model_executor.model_loader.base_loader")

    class Base:
        def __init__(self, load_config):
            self.load_config = load_config

        def create_model(self, vllm_config, model_config, prefix=""):
            return vllm_config.factory()

    fake.BaseModelLoader = Base
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    utils.process_weights_after_loading = lambda *args: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    from contextlib import contextmanager

    executor = types.ModuleType("vllm.model_executor.utils")

    @contextmanager
    def processed():
        yield

    executor.weights_already_processed = processed
    monkeypatch.setitem(sys.modules, executor.__name__, executor)
    dtype = types.ModuleType("vllm.utils.torch_utils")

    @contextmanager
    def default_dtype(value):
        old = torch.get_default_dtype()
        torch.set_default_dtype(value)
        try:
            yield
        finally:
            torch.set_default_dtype(old)

    dtype.set_default_torch_dtype = default_dtype
    monkeypatch.setitem(sys.modules, dtype.__name__, dtype)
    sys.modules.pop("sllm_store.vllm_loader", None)
    api = importlib.import_module("sllm_store.vllm_loader")
    monkeypatch.setattr(api, "require_version", lambda: None)
    monkeypatch.setattr(api, "tensor_parallel_identity", lambda: (0, 1))
    yield api
    sys.modules.pop("sllm_store.vllm_loader", None)


def identity():
    from sllm_store.vllm_plugin import LAYOUT_ID, PLUGIN_ID, WRITER_ID

    return dict(
        model="frozen/model",
        revision="rev",
        engine="vllm",
        engine_version="0.31.0",
        dtype="torch.float32",
        world_size=1,
        writer=WRITER_ID,
        plugin=PLUGIN_ID,
        layout=LAYOUT_ID,
        weight_phase="execution-ready",
    )


def test_loader_strict_config(loader_api, tmp_path):
    api = loader_api
    for extra in [
        {},
        {"artifact_path": "relative", "identity": identity()},
        {
            "artifact_path": str(tmp_path),
            "identity": identity(),
            "surprise": True,
        },
    ]:
        with pytest.raises(ValueError):
            api.ServerlessLLMModelLoader(
                types.SimpleNamespace(
                    model_loader_extra_config=extra, device=None
                )
            )
    bad = identity()
    bad["weight_phase"] = "checkpoint"
    with pytest.raises(ValueError, match="identity"):
        api.ServerlessLLMModelLoader(
            types.SimpleNamespace(
                model_loader_extra_config={
                    "artifact_path": str(tmp_path),
                    "identity": bad,
                },
                device=None,
            )
        )


@pytest.mark.parametrize("candidate", [False, True])
def test_loader_restores_persistent_state_and_keeps_postprocessing(
    loader_api, monkeypatch, tmp_path, candidate
):
    import torch

    from sllm_store.artifact import ArtifactTransaction
    from sllm_store.tensor_layout import capture_state

    api = loader_api

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(4.0))
            self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))
            self.register_buffer("scratch", torch.tensor(1.0), persistent=False)

    source = Model()
    storages, layout = capture_state(source)
    from sllm_store.vllm_plugin import LAYOUT_ID, PLUGIN_ID, WRITER_ID

    layout.update(
        rank=0,
        world_size=1,
        plugin=PLUGIN_ID,
        layout=LAYOUT_ID,
        writer=WRITER_ID,
        weight_phase="execution-ready",
    )
    target = tmp_path / "model"
    # Disk transport fixture; no native module is imported.
    with ArtifactTransaction(target) as tx:
        directory = tx.path / "rank_0"
        directory.mkdir()
        data = bytearray()
        index = {}
        for key, value in storages.items():
            while len(data) % 8:
                data.append(0)
            index[key] = [
                len(data),
                value.numel(),
                [value.numel()],
                [1],
                "torch.uint8",
            ]
            data.extend(bytes(value.tolist()))
        (directory / "tensor.data_0").write_bytes(data)
        (directory / "tensor_index.json").write_text(json.dumps(index))
        (directory / "tensor-layout.json").write_text(json.dumps(layout))
        tx.commit(
            identity(), lambda *_: {"passed": True, "method": "test-only"}
        )
    calls = []
    native = types.ModuleType("sllm_store.torch")

    def load_dict(model_path, device_map, storage_path):
        calls.append((model_path, device_map, storage_path))
        return storages

    native.load_dict = load_dict
    monkeypatch.setitem(sys.modules, native.__name__, native)
    processing = sys.modules["vllm.model_executor.model_loader.utils"]
    processed = []
    monkeypatch.setattr(
        processing,
        "process_weights_after_loading",
        lambda model, config, device: processed.append(device),
    )
    extra = {
        "artifact_path": str(target),
        "identity": identity(),
        "store_address": "127.0.0.1:18073",
    }
    if candidate:
        extra["candidate_manifest"] = json.loads(
            (target / "artifact-manifest.json").read_text()
        )
        (target / "artifact-commit.json").unlink()
        (target / "artifact-manifest.json").unlink()
    load = types.SimpleNamespace(model_loader_extra_config=extra, device=None)
    model_config = types.SimpleNamespace(
        model=str(target),
        dtype=torch.float32,
        quantization=None,
        hf_config=types.SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
    )
    config = types.SimpleNamespace(
        load_config=load,
        device_config=types.SimpleNamespace(device="cpu"),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        factory=Model,
    )
    loader = api.ServerlessLLMModelLoader(load)

    def cpu_transport(directory, device):
        calls.append((str(directory), str(device)))
        return storages

    monkeypatch.setattr(loader, "_load_storage", cpu_transport)
    restored = loader.load_model(config, model_config)
    assert torch.equal(restored.weight, source.weight)
    assert restored.counter.dtype == torch.int64
    assert restored._sllm_storage_owners is storages
    assert not restored.training
    assert len(processed) == 1
    assert calls == [(str(target / "rank_0"), "cpu")]
    # Exported execution state cannot be silently processed a second time.
    monkeypatch.setattr(
        processing,
        "process_weights_after_loading",
        lambda model, config, device: model.counter.fill_(99),
    )
    with pytest.raises(ValueError, match="changed execution"):
        loader.load_model(config, model_config)
    # Hash failures occur before the Store is contacted or skeleton constructed.
    (target / "rank_0/tensor.data_0").write_bytes(b"broken")
    with pytest.raises(ValueError):
        loader.load_model(config, model_config)
    assert len(calls) == 2


def test_worker_exports_actual_rank_and_world(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from sllm_store import vllm_plugin as plugin

    monkeypatch.setattr(plugin, "require_version", lambda: None)
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 1
    distributed.get_tensor_model_parallel_world_size = lambda: 2
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    transport = types.ModuleType("sllm_store.torch")
    calls = []
    transport.save_dict = lambda state, path: calls.append((state, path))
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    worker = plugin.ServerlessLLMWorkerExtension()
    worker.model_runner = types.SimpleNamespace(model=torch.nn.Linear(2, 2))
    receipt = worker.export_serverless_model(str(tmp_path))
    assert receipt["rank"] == 1 and receipt["world_size"] == 2
    assert calls[0][1] == str(tmp_path / "rank_1")
    assert all(t.dtype == torch.uint8 for t in calls[0][0].values())
    layout = json.loads((tmp_path / "rank_1/tensor-layout.json").read_text())
    assert layout["rank"] == 1 and layout["world_size"] == 2
    assert layout["weight_phase"] == "execution-ready"
    with pytest.raises(FileExistsError):
        worker.export_serverless_model(str(tmp_path))


def test_loader_native_transport_forwards_private_address(
    loader_api, monkeypatch, tmp_path
):
    import torch

    extra = {
        "artifact_path": str(tmp_path / "model"),
        "identity": identity(),
        "store_address": "127.0.0.1:18073",
    }
    loader = loader_api.ServerlessLLMModelLoader(
        types.SimpleNamespace(model_loader_extra_config=extra, device=None)
    )
    transport = types.ModuleType("sllm_store.torch")
    calls = []
    transport.load_dict = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    loader._load_storage(tmp_path / "model/rank_1", torch.device("cuda:3"))
    assert calls == [
        (
            ("model/rank_1", {"": 3}),
            {
                "storage_path": str(tmp_path),
                "server_address": "127.0.0.1:18073",
            },
        )
    ]
    with pytest.raises(ValueError, match="CUDA"):
        loader._load_storage(tmp_path / "model/rank_1", torch.device("cpu"))


@pytest.mark.parametrize(
    "address",
    [None, "remote:8073", "127.0.0.1:0", "127.0.0.1:70000", "127.0.0.1:12:34"],
)
def test_loader_refuses_ambiguous_store_address(loader_api, tmp_path, address):
    with pytest.raises(ValueError):
        loader_api.ServerlessLLMModelLoader(
            types.SimpleNamespace(
                model_loader_extra_config={
                    "artifact_path": str(tmp_path),
                    "identity": identity(),
                    "store_address": address,
                },
                device=None,
            )
        )


def test_loader_model_path_must_match_artifact(loader_api, tmp_path):
    import torch

    api = loader_api
    loader = api.ServerlessLLMModelLoader(
        types.SimpleNamespace(
            model_loader_extra_config={
                "artifact_path": str(tmp_path),
                "identity": identity(),
                "store_address": "127.0.0.1:18073",
            },
            device=None,
        )
    )
    model_config = types.SimpleNamespace(model="hf/other", dtype=torch.float32)
    with pytest.raises(ValueError, match="model path"):
        loader.download_model(model_config)


@pytest.mark.parametrize("mismatch", ["dtype", "world"])
def test_loader_rejects_runtime_identity_before_store(
    loader_api, monkeypatch, tmp_path, mismatch
):
    import torch

    loader = loader_api.ServerlessLLMModelLoader(
        types.SimpleNamespace(
            model_loader_extra_config={
                "artifact_path": str(tmp_path),
                "identity": identity(),
                "store_address": "127.0.0.1:18073",
            },
            device=None,
        )
    )
    if mismatch == "world":
        monkeypatch.setattr(
            loader_api, "tensor_parallel_identity", lambda: (1, 2)
        )
        monkeypatch.setattr(loader_api, "validate_artifact", lambda *_: None)
    config = types.SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.float16 if mismatch == "dtype" else torch.float32,
    )
    with pytest.raises(ValueError, match="dtype|world"):
        loader._validate(config)


def test_candidate_cannot_bypass_index_bounds(tmp_path):
    import hashlib

    from sllm_store.artifact import validate_candidate_artifact

    rank = tmp_path / "rank_0"
    rank.mkdir()
    (rank / "tensor.data_0").write_bytes(bytes(8))
    (rank / "tensor_index.json").write_text(
        json.dumps({"storage_0": [0, 99, [99], [1], "torch.uint8"]})
    )
    files = {
        path.relative_to(tmp_path).as_posix(): {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in rank.iterdir()
    }
    manifest = {"schema_version": 1, "identity": identity(), "files": files}
    with pytest.raises(ValueError, match="exceeds"):
        validate_candidate_artifact(tmp_path, identity(), manifest)


@pytest.mark.parametrize(
    "configured, expected", [("cuda", "cuda:3"), ("cuda:5", "cuda:5")]
)
def test_loader_resolves_cuda_device_once_for_all_restore_steps(
    loader_api, monkeypatch, tmp_path, configured, expected
):
    import torch

    api = loader_api
    load = types.SimpleNamespace(
        model_loader_extra_config={
            "artifact_path": str(tmp_path),
            "identity": identity(),
            "store_address": "127.0.0.1:18073",
        },
        device=None,
    )
    loader = api.ServerlessLLMModelLoader(load)
    source = torch.nn.Linear(2, 2)
    storages, layout = api.capture_state(source)
    monkeypatch.setattr(loader, "_validate", lambda _: (tmp_path, layout))
    current_queries = []

    def current_device():
        current_queries.append(True)
        return 3

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    seen = []
    monkeypatch.setattr(
        loader,
        "_load_storage",
        lambda directory, device: seen.append(device) or storages,
    )
    cpu_bind = api.bind_state

    def bind(model, state, sidecar, *, device):
        seen.append(device)
        cpu_bind(model, state, sidecar, device="cpu")

    monkeypatch.setattr(api, "bind_state", bind)
    processing = sys.modules["vllm.model_executor.model_loader.utils"]
    monkeypatch.setattr(
        processing,
        "process_weights_after_loading",
        lambda model, config, device: seen.append(device),
    )
    model_config = types.SimpleNamespace(
        dtype=torch.float32,
        quantization=None,
        hf_config=types.SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
    )
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        device_config=types.SimpleNamespace(device=torch.device(configured)),
        factory=lambda: torch.nn.Linear(2, 2),
    )
    restored = loader.load_model(config, model_config)
    assert torch.equal(restored.weight, source.weight)
    assert seen == [torch.device(expected)] * 3
    assert len(current_queries) == (1 if configured == "cuda" else 0)


@pytest.mark.parametrize("candidate", [False, True])
def test_download_validates_artifact_without_distributed_state(
    loader_api, monkeypatch, tmp_path, candidate
):
    import torch

    from sllm_store.artifact import ArtifactTransaction

    target = tmp_path / "model"
    with ArtifactTransaction(target) as tx:
        rank = tx.path / "rank_0"
        rank.mkdir()
        (rank / "tensor.data_0").write_bytes(bytes(8))
        (rank / "tensor_index.json").write_text(
            json.dumps({"storage_0": [0, 8, [8], [1], "torch.uint8"]})
        )
        tx.commit(
            identity(), lambda *_: {"passed": True, "method": "test-only"}
        )
    extra = {
        "artifact_path": str(target),
        "identity": identity(),
        "store_address": "127.0.0.1:18073",
    }
    if candidate:
        extra["candidate_manifest"] = json.loads(
            (target / "artifact-manifest.json").read_text()
        )
        (target / "artifact-commit.json").unlink()
        (target / "artifact-manifest.json").unlink()
    loader = loader_api.ServerlessLLMModelLoader(
        types.SimpleNamespace(model_loader_extra_config=extra, device=None)
    )

    def no_tp():
        pytest.fail(
            "download_model must not require distributed initialization"
        )

    monkeypatch.setattr(loader_api, "tensor_parallel_identity", no_tp)
    model_config = types.SimpleNamespace(model=str(target), dtype=torch.float32)
    loader.download_model(model_config)
    for bad in [
        types.SimpleNamespace(model="remote/model", dtype=torch.float32),
        types.SimpleNamespace(model=str(target), dtype=torch.float16),
    ]:
        with pytest.raises(ValueError):
            loader.download_model(bad)
    (target / "rank_0/tensor.data_0").write_bytes(bytes([1]) * 8)
    with pytest.raises(ValueError, match="inventory|digest"):
        loader.download_model(model_config)


@pytest.mark.parametrize("explicit_root", [False, True])
def test_loader_store_root_layout(
    loader_api, monkeypatch, tmp_path, explicit_root
):
    import torch

    path = tmp_path / "vllm" / "model"
    extra = {
        "artifact_path": str(path),
        "identity": identity(),
        "store_address": "127.0.0.1:18073",
    }
    if explicit_root:
        extra["store_root"] = str(tmp_path)
    loader = loader_api.ServerlessLLMModelLoader(
        types.SimpleNamespace(model_loader_extra_config=extra, device=None)
    )
    calls = []
    transport = types.ModuleType("sllm_store.torch")
    transport.load_dict = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    loader._load_storage(path / "rank_0", torch.device("cuda:0"))
    assert calls[0][0][0] == (
        "vllm/model/rank_0" if explicit_root else "model/rank_0"
    )
    assert calls[0][1]["storage_path"] == str(
        tmp_path if explicit_root else path.parent
    )


@pytest.mark.parametrize("root", ["relative", "/outside-model-root"])
def test_loader_rejects_invalid_store_root(loader_api, tmp_path, root):
    extra = {
        "artifact_path": str(tmp_path / "model"),
        "identity": identity(),
        "store_address": "127.0.0.1:18073",
        "store_root": root,
    }
    with pytest.raises(ValueError, match="store_root"):
        loader_api.ServerlessLLMModelLoader(
            types.SimpleNamespace(model_loader_extra_config=extra, device=None)
        )
