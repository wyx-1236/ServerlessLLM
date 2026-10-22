"""Real transaction/source files; GPU child commands are strict CPU doubles."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux preparation supervisor"
)


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "sllm_store"))
    from sllm import vllm_preparation as preparation
    from sllm_store import vllm_plugin as plugin

    monkeypatch.setattr(plugin, "require_version", lambda: None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"]}'
    )
    (source / "model.safetensors").write_bytes(b"test-input-only")
    calls = []
    faults = set()

    def run(command, log_path, **kwargs):
        calls.append(command)
        Path(log_path).write_text("CPU command double\n")
        if "sllm.vllm_prepare_worker" not in command:
            if "namespace" in faults:
                raise RuntimeError("network namespace unavailable")
            return
        mode = command[command.index("sllm.vllm_prepare_worker") + 1]
        request = json.loads(Path(command[-1]).read_text())
        reference = Path(request["reference_path"])
        if mode == "export":
            rank = Path(request["artifact_path"]) / "rank_0"
            rank.mkdir()
            (rank / "tensor.data_0").write_bytes(b"12345678")
            (rank / "tensor_index.json").write_text(
                json.dumps(
                    {
                        "storage_0": [0, 8, [8], [1], "torch.uint8"],
                    }
                )
            )
            reference.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "identity": request["identity"],
                        "engine_options": request["engine_options"],
                        "validation": request["validation"],
                        "outputs": [{}, {}],
                    }
                )
            )
            report = dict(
                rank=0,
                world_size=1,
                writer=plugin.WRITER_ID,
                plugin=plugin.PLUGIN_ID,
                layout=plugin.LAYOUT_ID,
                weight_phase="execution-ready",
            )
            reports = (
                [report, report] if "duplicate_rank" in faults else [report]
            )
            result = dict(
                schema_version=1,
                rank_reports=reports,
                reference_sha256=hashlib.sha256(
                    reference.read_bytes()
                ).hexdigest(),
            )
        else:
            assert mode == "verify"
            assert "source_path" not in request
            assert "candidate_manifest" in request
            assert "unshare" in Path(command[0]).name
            if "restore" in faults:
                raise RuntimeError("restore child failed")
            inode = request["parent_network_namespace"]
            result = dict(
                schema_version=1,
                passed=True,
                method="independent-offline-native-store-token-logprob-v1",
                tokens_verified=True,
                logprobs_verified=True,
                request_count=2,
                reference_sha256=hashlib.sha256(
                    reference.read_bytes()
                ).hexdigest(),
                isolation=dict(
                    parent_network_namespace=inode,
                    network_namespace=inode
                    if "isolation" in faults
                    else inode + 1,
                    interfaces=["lo"],
                ),
            )
            if "reference" in faults:
                reference.write_text("tampered")
        Path(request["result_path"]).write_text(json.dumps(result))

    monkeypatch.setattr(preparation, "run_preparation_process", run)
    return preparation, source, tmp_path / "models", calls, faults


def test_complete_prepare_then_reuse_without_child_launch(pipeline):
    preparation, source, storage, calls, _ = pipeline
    first = preparation.prepare_vllm_model(
        "alias/model", str(source), "bfloat16", storage_path=storage
    )
    assert first["verification"]["tokens_verified"] is True
    assert (storage / "vllm/alias/model/artifact-commit.json").exists()
    before = len(calls)
    second = preparation.prepare_vllm_model(
        "alias/model", str(source), "bfloat16", storage_path=storage
    )
    assert second == first
    assert len(calls) == before


@pytest.mark.parametrize(
    "fault",
    ["duplicate_rank", "restore", "isolation", "reference", "namespace"],
)
def test_failure_never_publishes_or_deletes_neighbor(pipeline, fault):
    preparation, source, storage, _, faults = pipeline
    faults.add(fault)
    neighbor = storage / "vllm/neighbor"
    neighbor.mkdir(parents=True)
    (neighbor / "keep").write_bytes(b"keep")
    with pytest.raises((ValueError, RuntimeError)):
        preparation.prepare_vllm_model(
            "model", str(source), "bfloat16", storage_path=storage
        )
    assert not (storage / "vllm/model").exists()
    assert not list((storage / "vllm").glob(".model.prepare-*"))
    assert (neighbor / "keep").read_bytes() == b"keep"


def test_existing_incomplete_target_fails_without_mutation_or_launch(pipeline):
    preparation, source, storage, calls, _ = pipeline
    target = storage / "vllm/model"
    target.mkdir(parents=True)
    (target / "incomplete").write_bytes(b"old")
    with pytest.raises(ValueError):
        preparation.prepare_vllm_model(
            "model", str(source), "bfloat16", storage_path=storage
        )
    assert not calls
    assert (target / "incomplete").read_bytes() == b"old"


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "bad\\name", "bad:name"]
)
def test_model_alias_cannot_escape_storage(pipeline, name):
    preparation, source, storage, calls, _ = pipeline
    with pytest.raises(ValueError, match="model name"):
        preparation.prepare_vllm_model(
            name, str(source), "bfloat16", storage_path=storage
        )
    assert not calls


def test_prefrozen_hub_receipt_prevents_network_during_gpu_batch(
    pipeline, monkeypatch
):
    preparation, source, storage, _, _ = pipeline
    frozen = preparation.freeze_source(
        str(source), revision=None, cache_dir=storage / "cache"
    )
    frozen.update(source="actual/source", revision="a" * 40)
    monkeypatch.setattr(
        preparation,
        "freeze_source",
        lambda *a, **kw: pytest.fail("unexpected source download"),
    )
    manifest = preparation.prepare_vllm_model(
        "model",
        "actual/source",
        "bfloat16",
        storage_path=storage,
        revision="a" * 40,
        source_snapshot=frozen,
    )
    assert manifest["identity"]["source"] == "actual/source"
    assert manifest["identity"]["revision"] == "a" * 40


def test_changed_prefrozen_receipt_never_launches_gpu_child(pipeline):
    preparation, source, storage, calls, _ = pipeline
    frozen = preparation.freeze_source(
        str(source), revision=None, cache_dir=storage / "cache"
    )
    (source / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        preparation.prepare_vllm_model(
            "model",
            str(source),
            "bfloat16",
            storage_path=storage,
            source_snapshot=frozen,
        )
    assert not calls


@pytest.mark.parametrize(
    "source_id,revision",
    [
        ("actual/source", "main"),
        ("actual/source", None),
        ("local-snapshot", "sha256:" + "0" * 64),
        (None, "a" * 40),
    ],
)
def test_prefrozen_receipt_requires_immutable_identity(
    pipeline, source_id, revision
):
    preparation, source, storage, calls, _ = pipeline
    frozen = preparation.freeze_source(
        str(source), revision=None, cache_dir=storage / "cache"
    )
    frozen.update(source=source_id, revision=revision)
    with pytest.raises(ValueError, match="source identity"):
        preparation.prepare_vllm_model(
            "model",
            str(source),
            "bfloat16",
            storage_path=storage,
            source_snapshot=frozen,
        )
    assert not calls
