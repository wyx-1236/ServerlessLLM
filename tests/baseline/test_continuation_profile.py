"""CPU checks for opt-in numerical qualification; never load a real engine."""

import ast
import builtins
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts" / "baseline"


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("continuation_profile")


def test_default_profile_does_not_change_model_or_worker_environment(profile):
    assert profile.worker_environment({}) == {}
    assert profile.backend_overrides({}) == {}
    profile.validate_environment({}, {})
    profile.validate_environment({}, {"VLLM_BATCH_INVARIANT": "0"})


def test_opt_in_requires_matching_startup_environment(profile):
    config = {"numerical_profile": "batch-invariant-v1"}
    profile.validate_environment(config, {"VLLM_BATCH_INVARIANT": "1"})
    assert profile.worker_environment(config) == {"VLLM_BATCH_INVARIANT": "1"}
    assert profile.backend_overrides(config) == {
        "attention_config": {"backend": "FLASH_ATTN"},
        "worker_extension_cls": (
            "sllm.backends.vllm_qualification.NumericalQualificationWorker"
        ),
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("default", "1"),
        ("default", "garbage"),
        ("batch-invariant-v1", None),
        ("batch-invariant-v1", "0"),
        ("batch-invariant-v1", "true"),
        ("unknown", "1"),
        (None, "1"),
    ],
)
def test_conflicting_or_unknown_profile_fails(profile, name, value):
    with pytest.raises(ValueError):
        profile.validate_environment(
            {"numerical_profile": name},
            {} if value is None else {"VLLM_BATCH_INVARIANT": value},
        )


def worker_report():
    return {
        "schema_version": 1,
        "engine_version": "0.31.0",
        "pid": 101,
        "cgroup": "0::/owned\n",
        "gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111",
        "device_name": "NVIDIA A100-SXM4-40GB",
        "compute_capability": [8, 0],
        "environment": "1",
        "batch_invariant_initialized": True,
        "use_v2_model_runner": True,
        "runner_class": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "dtype": "torch.bfloat16",
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "attention_layers": [
            {
                "name": "model.layers.0.self_attn.attn",
                "backend": "FLASH_ATTN",
                "batch_invariant_enabled": True,
            }
        ],
    }


def test_complete_worker_report_passes_without_mutation(profile):
    report = worker_report()
    before = copy.deepcopy(report)
    profile.validate_worker_reports(
        [report], report["gpu_uuid"], report["cgroup"]
    )
    assert report == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine_version", "0.30.0"),
        ("pid", 0),
        ("environment", "0"),
        ("batch_invariant_initialized", False),
        ("batch_invariant_initialized", 1),
        ("use_v2_model_runner", False),
        ("runner_class", "legacy.Runner"),
        ("compute_capability", [9, 0]),
        ("device_name", "NVIDIA H100"),
        ("dtype", "torch.float16"),
        ("enforce_eager", False),
        ("tensor_parallel_size", 2),
        ("attention_layers", []),
        (
            "attention_layers",
            [
                {
                    "name": "a",
                    "backend": "FLASH_ATTN",
                    "batch_invariant_enabled": False,
                }
            ],
        ),
        (
            "attention_layers",
            [
                {
                    "name": "a",
                    "backend": "TRITON_ATTN",
                    "batch_invariant_enabled": True,
                }
            ],
        ),
        ("gpu_uuid", "GPU-22222222-2222-2222-2222-222222222222"),
        ("cgroup", "0::/other\n"),
    ],
)
def test_report_cannot_claim_wrong_or_uninitialized_runtime(
    profile, field, value
):
    report = worker_report()
    expected_uuid, expected_cgroup = report["gpu_uuid"], report["cgroup"]
    report[field] = value
    with pytest.raises(ValueError):
        profile.validate_worker_reports(
            [report], expected_uuid, expected_cgroup
        )


@pytest.mark.parametrize("reports", [[], [{}, {}], [{}], None])
def test_tp1_requires_exactly_one_complete_report(profile, reports):
    with pytest.raises(ValueError):
        profile.validate_worker_reports(
            reports, worker_report()["gpu_uuid"], "0::/owned\n"
        )


def test_worker_environment_does_not_replace_gpu_or_temp_identity(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = importlib.import_module("qualify_migration")
    value = module.worker_environment(
        "GPU-owned", tmp_path / "node", extra_env={"VLLM_BATCH_INVARIANT": "1"}
    )
    assert value == {
        "CUDA_VISIBLE_DEVICES": "GPU-owned",
        "TMPDIR": str(tmp_path / "node"),
        "VLLM_BATCH_INVARIANT": "1",
    }
    with pytest.raises(ValueError):
        module.worker_environment(
            "GPU-owned",
            tmp_path / "bad",
            extra_env={"CUDA_VISIBLE_DEVICES": "other"},
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_both_private_nodes_receive_profile_before_creation_and_restore_parent(
    profile, monkeypatch, tmp_path, enabled
):
    module = importlib.import_module("qualify_migration")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-devices")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    calls = []

    def add_node(**kwargs):
        calls.append((copy.deepcopy(kwargs), dict(os.environ)))

    config = {
        "numerical_profile": "batch-invariant-v1" if enabled else "default"
    }
    environments = module.add_worker_nodes(
        SimpleNamespace(add_node=add_node),
        ["GPU-1", "GPU-2"],
        tmp_path,
        {},
        extra_env=profile.worker_environment(config),
    )
    assert len(calls) == len(environments) == 2
    for index, (kwargs, environ) in enumerate(calls):
        assert kwargs["env_vars"] == environments[index]
        assert environ["CUDA_VISIBLE_DEVICES"] == f"GPU-{index + 1}"
        assert environ["VLLM_BATCH_INVARIANT"] == ("1" if enabled else "0")
        assert kwargs["env_vars"].get("VLLM_BATCH_INVARIANT") == (
            "1" if enabled else None
        )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-devices"
    assert os.environ["VLLM_BATCH_INVARIANT"] == "0"


def test_worker_probe_reads_initialized_state_without_initializing_or_importing(
    monkeypatch, profile
):
    from sllm.backends.vllm_qualification import NumericalQualificationWorker

    report = worker_report()
    device_reads = []
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        get_device_properties=lambda device: device_reads.append(device)
        or SimpleNamespace(
            uuid=report["gpu_uuid"],
            name=report["device_name"],
            major=8,
            minor=0,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(__version__="0.31.0")
    )
    state = SimpleNamespace(_batch_invariant_MODE=True)
    module_name = "vllm.model_executor.determinism.batch_invariant"
    monkeypatch.setitem(sys.modules, module_name, state)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    real_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: report["cgroup"]
        if str(path) == "/proc/self/cgroup"
        else real_read(path, *a, **kw),
    )
    name = report["attention_layers"][0]["name"]
    layer = SimpleNamespace(impl=SimpleNamespace(batch_invariant_enabled=True))
    group = SimpleNamespace(
        layer_names=[name],
        backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
    )
    runner = type(
        "GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"}
    )()
    runner.attn_groups = [[group]]
    worker = NumericalQualificationWorker()
    worker.device, worker.model_runner = "cuda:0", runner
    worker.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            dtype="torch.bfloat16", enforce_eager=True
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        compilation_config=SimpleNamespace(
            static_forward_context={name: layer}
        ),
    )
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "vllm"}:
            raise AssertionError("Probe must not initialize dependencies")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    actual = worker.sllm_numerical_runtime()
    assert actual == {**report, "pid": os.getpid()}
    profile.validate_worker_reports(
        [actual], report["gpu_uuid"], report["cgroup"]
    )
    assert device_reads == ["cuda:0"] and state._batch_invariant_MODE is True
    # An environment flag alone is not proof; missing module state remains false.
    monkeypatch.delitem(sys.modules, module_name)
    assert (
        worker.sllm_numerical_runtime()["batch_invariant_initialized"] is False
    )
    cuda.is_initialized = lambda: False
    device_reads.clear()
    with pytest.raises(RuntimeError, match="initialized Worker"):
        worker.sllm_numerical_runtime()
    assert device_reads == []


@pytest.mark.parametrize("enabled", [False, True])
def test_pinned_sm80_initialization_installs_batch_invariant_operations(
    enabled,
):
    from fetch_vllm_resume_source import MANIFEST, verify_source

    root = os.environ.get("SLLM_VLLM_031_SOURCE")
    if not root:
        pytest.skip("Set SLLM_VLLM_031_SOURCE for pinned-source CPU checks")
    root = Path(root)
    verify_source(root, json.loads(MANIFEST.read_text()))
    path = root / "vllm/model_executor/determinism/batch_invariant.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in {"init_batch_invariance", "enable_batch_invariant_mode"}
    ]
    assert len(tree.body) == 2
    installed, initialization = [], []
    library = SimpleNamespace(
        impl=lambda name, *args, **kwargs: installed.append(name)
    )
    cuda = SimpleNamespace(
        matmul=SimpleNamespace(), preferred_blas_library=lambda **kw: None
    )
    namespace = dict(
        _batch_invariant_MODE=False,
        _batch_invariant_LIB=None,
        envs=SimpleNamespace(VLLM_BATCH_INVARIANT=enabled),
        torch=SimpleNamespace(
            library=SimpleNamespace(Library=lambda *args: library),
            backends=SimpleNamespace(
                cuda=cuda,
                cudnn=SimpleNamespace(
                    conv=SimpleNamespace(), rnn=SimpleNamespace()
                ),
            ),
        ),
        current_platform=SimpleNamespace(
            dispatch_key="CUDA",
            is_cuda=lambda: True,
            is_xpu=lambda: False,
            is_device_capability_family=lambda value: value == 80,
        ),
        get_max_shared_memory_bytes=lambda: 163840,
        is_torch_equal_or_newer=lambda version: True,
        resolve_tuned_matmul_configs=lambda: initialization.append("resolve"),
        override_envs_for_invariance=lambda: initialization.append(
            "environment"
        ),
    )
    for name in (
        "mm_batch_invariant",
        "addmm_batch_invariant",
        "matmul_batch_invariant",
        "linear_batch_invariant",
        "mean_batch_invariant",
        "_log_softmax_batch_invariant",
        "softmax_batch_invariant",
        "bmm_batch_invariant",
    ):
        namespace[name] = object()
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["init_batch_invariance"]()
    assert namespace["_batch_invariant_MODE"] is enabled
    if enabled:
        assert initialization == ["resolve", "environment"]
        assert {
            "aten::mm",
            "aten::addmm",
            "aten::matmul",
            "aten::linear",
            "aten::_log_softmax",
            "aten::_softmax",
            "aten::bmm",
        } <= set(installed)
        count = len(installed)
        namespace["enable_batch_invariant_mode"]()
        assert len(installed) == count
    else:
        assert not installed and not initialization
