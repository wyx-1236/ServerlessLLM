"""CPU guards for the explicitly invoked, real-engine D1 qualification."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "baseline"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "qualify_migration", SCRIPTS / "qualify_migration.py"
)
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


@pytest.mark.parametrize("prefix", ["", "GPU-"])
def test_cuda_uuid_normalizes_only_valid_uuid_representation(prefix):
    bare = "5b3578d9-81fc-d58e-966d-6492adb4e0bb"
    assert qualification.canonical_gpu_uuid(prefix + bare) == "GPU-" + bare
    assert (
        qualification.canonical_gpu_uuid(prefix + bare.upper()) == "GPU-" + bare
    )


def test_cuda_uuid_does_not_match_a_different_approved_device():
    observed = "5b3578d9-81fc-d58e-966d-6492adb4e0bb"
    wrong = "GPU-5b3578d9-81fc-d58e-966d-6492adb4e0bc"
    assert qualification.canonical_gpu_uuid(
        observed
    ) != qualification.canonical_gpu_uuid(wrong)


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        "",
        "0",
        "GPU-",
        "gpu-5b3578d9-81fc-d58e-966d-6492adb4e0bb",
        "GPU-GPU-5b3578d9-81fc-d58e-966d-6492adb4e0bb",
        " 5b3578d9-81fc-d58e-966d-6492adb4e0bb",
        "5b3578d9-81fc-d58e-966d-6492adb4e0bb\n",
        "5b3578d9-81fc-d58e-966d-6492adb4e0bg",
        "5b3578d981fcd58e966d6492adb4e0bb",
    ],
)
def test_cuda_uuid_rejects_malformed_identifiers(value):
    with pytest.raises(ValueError):
        qualification.canonical_gpu_uuid(value)


@pytest.fixture
def approval(tmp_path):
    artifact = tmp_path / "store" / "vllm" / "Qwen3-0.6B-stageB"
    artifact.mkdir(parents=True)
    (artifact / "artifact-manifest.json").write_text("{}")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "passed": True,
                "wheel_sha256": "f" * 64,
                "deployment": {
                    "commit": "abc123",
                    "all_tracked_files_match_commit": True,
                },
            }
        )
    )
    return dict(
        approved=True,
        server="inspur-gpu-12",
        action="soft lock GPU 1,3",
        scope="stage-d1-migration",
        physical_gpu_indices=[1, 3],
        gpu_uuids=[
            "GPU-11111111-1111-1111-1111-111111111111",
            "GPU-33333333-3333-3333-3333-333333333333",
        ],
        deadline_epoch=2000,
        output=str(tmp_path / "owned"),
        ray_directory=str(tmp_path / "owned" / "ray-root"),
        unit="d1-test",
        parent_network_namespace=10,
        native_verification_path=str(verification),
        wheel_sha256="f" * 64,
        commit="abc123",
        artifact_path=str(artifact),
        source_snapshot={"path": str(snapshot), "revision": "abc123"},
    )


def test_import_is_stdlib_only():
    # AST catches accidental imports even if the suite already imported Ray.
    import ast

    tree = ast.parse((SCRIPTS / "qualify_migration.py").read_text())
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        (
            node.module
            if isinstance(node, ast.ImportFrom)
            else node.names[0].name
        ).split(".")[0]
        in sys.stdlib_module_names
        for node in imports
    )


def test_exact_approval_and_owned_paths(approval):
    env = {"CUDA_VISIBLE_DEVICES": ",".join(approval["gpu_uuids"])}
    qualification.validate_approval(
        approval, env, 1000, "0::/system.slice/d1-test.service\n"
    )
    for field, value in [
        ("approved", False),
        ("server", "other"),
        ("action", "hard lock GPU 1,3"),
        ("scope", "stage-c-service"),
        ("physical_gpu_indices", [1, 1]),
        ("physical_gpu_indices", [True, 3]),
        ("gpu_uuids", approval["gpu_uuids"][:1]),
        ("deadline_epoch", 1119),
        ("unit", "other.service"),
        ("wheel_sha256", "wrong"),
        ("commit", "wrong"),
        ("ray_directory", str(Path(approval["output"]).parent)),
    ]:
        with pytest.raises((ValueError, RuntimeError)):
            qualification.validate_approval(
                {**approval, field: value},
                env,
                1000,
                "0::/system.slice/d1-test.service\n",
            )
    with pytest.raises(ValueError):
        qualification.validate_approval(
            approval,
            {"CUDA_VISIBLE_DEVICES": "1,3"},
            1000,
            "0::/system.slice/d1-test.service\n",
        )


def test_worker_env_and_parent_visibility_are_per_node(
    approval, tmp_path, monkeypatch
):
    original = ",".join(approval["gpu_uuids"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", original)
    envs = [
        qualification.worker_environment(uuid, tmp_path / str(i))
        for i, uuid in enumerate(approval["gpu_uuids"])
    ]
    assert envs[0]["TMPDIR"] != envs[1]["TMPDIR"]
    for env, uuid in zip(envs, approval["gpu_uuids"]):
        assert Path(env["TMPDIR"]).is_dir()
        assert env["CUDA_VISIBLE_DEVICES"] == uuid
        with pytest.raises(RuntimeError):
            with qualification.parent_environment(env):
                assert os.environ["CUDA_VISIBLE_DEVICES"] == uuid
                raise RuntimeError("node creation failed")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == original


def test_short_ray_root_is_bound_to_run_stamp(approval):
    approved = {
        **approval,
        "unit": "sllm-d1-migration-20261022-120000",
        "output": "/tmp/evidence/gpu-stage-d1-20261022-120000",
        "ray_directory": "/tmp/sd1-20261022-120000",
    }
    env = {"CUDA_VISIBLE_DEVICES": ",".join(approved["gpu_uuids"])}
    cgroup = "0::/system.slice/sllm-d1-migration-20261022-120000.service\n"
    qualification.validate_approval(approved, env, 1000, cgroup)
    for root in ("/tmp/other", "/tmp/sd1-different", "/tmp"):
        with pytest.raises(ValueError):
            qualification.validate_approval(
                {**approved, "ray_directory": root}, env, 1000, cgroup
            )


def test_log_reader_excludes_old_bytes_and_rejects_truncation(tmp_path):
    log = tmp_path / "worker-owned-123.out"
    log.write_text("old run payload\n")
    offsets = {str(log): (log.stat().st_ino, log.stat().st_size)}
    with log.open("a") as stream:
        stream.write("new payload\n")
    assert qualification.read_run_logs(tmp_path, offsets) == "new payload\n"
    log.write_text("cut")
    with pytest.raises(RuntimeError):
        qualification.read_run_logs(tmp_path, offsets)


def test_failed_actor_identity_is_new_and_owned_even_during_blocking_init(
    tmp_path,
):
    process = tmp_path / "123"
    process.mkdir()
    cgroup = "0::/system.slice/d1.service\n"
    (process / "cgroup").write_text(cgroup)
    rows = {
        "old": {"Name": "model_old", "Pid": 123, "Address": {"NodeID": "node"}},
        "new": {
            "Name": "model_target",
            "Pid": 123,
            "Address": {"NodeID": "node"},
        },
    }
    observed = qualification.new_target_identity(
        rows, {"old"}, "model", "source", "node", cgroup, tmp_path
    )
    assert observed["pid"] == 123 and observed["instance_id"] == "model_target"
    assert (
        qualification.new_target_identity(
            rows, {"old", "new"}, "model", "source", "node", cgroup, tmp_path
        )
        is None
    )
    with pytest.raises(ValueError):
        qualification.new_target_identity(
            rows, {"old"}, "model", "source", "wrong-node", cgroup, tmp_path
        )
    (process / "cgroup").write_text("0::/other\n")
    with pytest.raises(ValueError):
        qualification.new_target_identity(
            rows, {"old"}, "model", "source", "node", cgroup, tmp_path
        )


def test_trace_pairs_require_exact_owned_actor_iteration_and_tokens():
    record = dict(
        source_instance_id="source",
        target_instance_id="target",
        iteration=0,
        tokens=[list(range(70))],
    )
    snapshot = "Migration token snapshot: " + repr(record)
    replay = "Migration replay input: " + repr(record)
    pair = qualification.trace_pairs(
        snapshot + "\n" + replay, "source", "target"
    )
    assert pair == [record]
    assert not qualification.trace_pairs(
        snapshot + "\n" + replay, "old", "target"
    )
    assert not qualification.trace_pairs(snapshot, "source", "target")
    changed = {**record, "tokens": [[999]]}
    assert not qualification.trace_pairs(
        snapshot + "\nMigration replay input: " + repr(changed),
        "source",
        "target",
    )
    assert not qualification.trace_pairs(
        "Migration token snapshot: __import__('os')", "source", "target"
    )


def test_prefix_counter_deltas_are_target_and_model_specific(tmp_path):
    model = "Qwen3-0.6B-stageB"

    def row(metric, value, name=model):
        return (
            json.dumps(
                [metric, metric, {"model_name": name, "engine": "0"}, "help"]
            ),
            value,
            0,
            0,
        )

    rows = [
        row("vllm:prefix_cache_queries", 128),
        row("vllm:prefix_cache_hits", 64),
        row("vllm:prefix_cache_hits", 9000, "unrelated"),
    ]
    before = qualification.counter_totals(rows, model)
    after = qualification.counter_totals(
        [
            row("vllm:prefix_cache_queries", 256),
            row("vllm:prefix_cache_hits", 128),
        ],
        model,
    )
    assert qualification.cache_delta(before, after) == {
        "queries": 128,
        "hits": 64,
    }
    with pytest.raises(ValueError):
        qualification.cache_delta(after, before)
    with pytest.raises(ValueError):
        qualification.cache_delta(before, before)
    owned = tmp_path / "counter_123.db"
    owned.write_bytes(b"counter data")
    assert qualification.counter_file(tmp_path, 123) == owned
    with pytest.raises(ValueError):
        qualification.counter_file(tmp_path, 456)


@pytest.mark.asyncio
async def test_fault_configuration_restores_even_on_failure():
    class Router:
        backend_config = {
            "max_model_len": 512,
            "kv_cache_memory_bytes": 256 * 1024**2,
        }

    router = Router()
    original = router.backend_config

    async def remote(fn, *args):
        return fn(router, *args)

    router.__ray_call__ = SimpleNamespace(remote=remote)
    with pytest.raises(RuntimeError, match="actual failure"):
        async with qualification.fault_configuration(
            router, {"max_model_len": 16}
        ):
            assert original["max_model_len"] == 512
            assert router.backend_config["max_model_len"] == 16
            raise RuntimeError("actual failure")
    assert router.backend_config == original


def test_pass_requires_every_check_and_cleanup():
    result = {
        "checks": dict.fromkeys(qualification.REQUIRED_CHECKS, True),
        "cleanup_completed": True,
        "cleanup_errors": [],
    }
    assert qualification.runtime_passed(result)
    for name in qualification.REQUIRED_CHECKS:
        assert not qualification.runtime_passed(
            {**result, "checks": {**result["checks"], name: False}}
        )
    assert not qualification.runtime_passed(
        {**result, "cleanup_errors": ["store failed"]}
    )
    assert not qualification.runtime_passed(
        {**result, "error": "engine failed"}
    )
