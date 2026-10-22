"""Disk-only artifact transaction tests; no tensor libraries or model required."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sllm_store"))


def api():
    return importlib.import_module("sllm_store.artifact")


IDENTITY = {
    "model": "test/model",
    "revision": "frozen-revision",
    "engine": "vllm",
    "engine_version": "0.31.0",
    "world_size": 1,
    "dtype": "bfloat16",
    "writer": "native-test",
}


def populate(root):
    rank = root / "rank_0"
    rank.mkdir()
    (rank / "tensor.data_0").write_bytes(bytes(range(16)))
    (rank / "tensor_index.json").write_text(
        json.dumps(
            {
                "weight": [0, 16, [4], [1], "torch.float32"],
            }
        )
    )
    (root / "config.json").write_text('{"model_type":"test"}')


def verified(root, manifest):
    assert (root / "rank_0/tensor.data_0").read_bytes() == bytes(range(16))
    return {"passed": True, "method": "test-byte-oracle"}


def test_committed_artifact_rechecks_bytes_and_identity(tmp_path):
    artifact = api()
    target = tmp_path / "model"
    with artifact.ArtifactTransaction(target) as transaction:
        populate(transaction.path)
        transaction.commit(IDENTITY, verified)
    manifest = artifact.validate_artifact(target, IDENTITY)
    assert manifest["identity"] == IDENTITY
    assert manifest["verification"]["passed"] is True
    marker = json.loads((target / "artifact-commit.json").read_text())
    assert (
        marker["manifest_sha256"]
        == hashlib.sha256(
            (target / "artifact-manifest.json").read_bytes()
        ).hexdigest()
    )


@pytest.mark.parametrize("mutation", ["truncate", "change", "extra", "missing"])
def test_reuse_rejects_changed_inventory(tmp_path, mutation):
    artifact = api()
    target = tmp_path / "model"
    with artifact.ArtifactTransaction(target) as transaction:
        populate(transaction.path)
        transaction.commit(IDENTITY, verified)
    data = target / "rank_0/tensor.data_0"
    if mutation == "truncate":
        data.write_bytes(b"short")
    elif mutation == "change":
        data.write_bytes(b"x" * 16)
    elif mutation == "extra":
        (target / "unexpected.bin").write_bytes(b"extra")
    else:
        data.unlink()
    with pytest.raises(ValueError):
        artifact.validate_artifact(target, IDENTITY)


def test_failed_verification_cleans_only_own_staging(tmp_path):
    artifact = api()
    target = tmp_path / "model"
    neighbor = tmp_path / "existing"
    neighbor.mkdir()
    with pytest.raises(ValueError, match="verification"):
        with artifact.ArtifactTransaction(target) as transaction:
            staging = transaction.path
            populate(staging)
            transaction.commit(IDENTITY, lambda *_: {"passed": False})
    assert not staging.exists()
    assert not target.exists()
    assert neighbor.exists()


def test_existing_directory_is_never_replaced(tmp_path):
    artifact = api()
    target = tmp_path / "model"
    target.mkdir()
    (target / "preserve").write_text("old")
    with artifact.ArtifactTransaction(target) as transaction:
        populate(transaction.path)
        with pytest.raises(FileExistsError):
            transaction.commit(IDENTITY, verified)
    assert (target / "preserve").read_text() == "old"
    assert not (target / "artifact-commit.json").exists()


def test_foreign_empty_directory_created_during_flush_is_not_replaced(
    tmp_path, monkeypatch
):
    artifact = api()
    target = tmp_path / "model"
    sync = artifact._sync_directory

    def concurrent_creator(directory):
        if Path(directory).name.startswith(".model.prepare-"):
            target.mkdir(exist_ok=True)
        return sync(directory)

    monkeypatch.setattr(artifact, "_sync_directory", concurrent_creator)
    with artifact.ArtifactTransaction(target) as transaction:
        populate(transaction.path)
        with pytest.raises(FileExistsError):
            transaction.commit(IDENTITY, verified)
    assert target.is_dir()
    assert not list(target.iterdir())


def test_missing_rank_prevents_verifier_and_publication(tmp_path):
    artifact = api()
    called = []
    with artifact.ArtifactTransaction(tmp_path / "model") as transaction:
        populate(transaction.path)
        with pytest.raises(ValueError, match="rank"):
            transaction.commit(
                dict(IDENTITY, world_size=2), lambda *_: called.append(True)
            )
    assert not called


def test_directory_alone_and_mismatched_identity_are_not_reusable(tmp_path):
    artifact = api()
    target = tmp_path / "model"
    target.mkdir()
    with pytest.raises(ValueError):
        artifact.validate_artifact(target, IDENTITY)
    target.rmdir()
    with artifact.ArtifactTransaction(target) as transaction:
        populate(transaction.path)
        transaction.commit(IDENTITY, verified)
    with pytest.raises(ValueError, match="identity"):
        artifact.validate_artifact(target, dict(IDENTITY, revision="different"))


def test_symlink_payload_is_rejected(tmp_path):
    artifact = api()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    with artifact.ArtifactTransaction(tmp_path / "model") as transaction:
        populate(transaction.path)
        try:
            (transaction.path / "link").symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable on this host")
        with pytest.raises(ValueError, match="symlink"):
            transaction.commit(IDENTITY, verified)


def test_verifier_cannot_change_payload_after_inventory(tmp_path):
    artifact = api()

    def changing_verifier(root, manifest):
        (root / "rank_0/tensor.data_0").write_bytes(b"changed")
        return {"passed": True}

    with artifact.ArtifactTransaction(tmp_path / "model") as transaction:
        populate(transaction.path)
        with pytest.raises(ValueError, match="changed"):
            transaction.commit(IDENTITY, changing_verifier)
    assert not (tmp_path / "model").exists()


def test_foreign_publication_lock_is_not_removed(tmp_path):
    artifact = api()
    lock = tmp_path / ".model.publish-lock"
    lock.mkdir()
    with artifact.ArtifactTransaction(tmp_path / "model") as transaction:
        populate(transaction.path)
        with pytest.raises(FileExistsError):
            transaction.commit(IDENTITY, verified)
    assert lock.exists()
    assert not (tmp_path / "model").exists()


def test_commit_marker_cannot_be_reused_with_edited_manifest(tmp_path):
    artifact = api()
    target = tmp_path / "model"
    with artifact.ArtifactTransaction(target) as transaction:
        populate(transaction.path)
        transaction.commit(IDENTITY, verified)
    manifest = target / "artifact-manifest.json"
    value = json.loads(manifest.read_text())
    value["identity"]["revision"] = "different"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        artifact.validate_artifact(target, dict(IDENTITY, revision="different"))


@pytest.mark.parametrize(
    "invalid", ["offset", "size", "shape", "stride", "dtype", "partition"]
)
def test_invalid_tensor_index_rejected_before_restore(tmp_path, invalid):
    artifact = api()
    with artifact.ArtifactTransaction(tmp_path / "model") as transaction:
        populate(transaction.path)
        rank = transaction.path / "rank_0"
        record = [0, 16, [4], [1], "torch.float32"]
        if invalid == "offset":
            record[0] = 8
        elif invalid == "size":
            record[1] = 17
        elif invalid == "shape":
            record[2] = [5]
        elif invalid == "stride":
            record[3] = [-1]
        elif invalid == "dtype":
            record[4] = "invalid"
        else:
            (rank / "tensor.data_2").write_bytes(b"gap")
        (rank / "tensor_index.json").write_text(json.dumps({"weight": record}))
        called = []
        with pytest.raises(ValueError):
            transaction.commit(IDENTITY, lambda *_: called.append(True))
        assert not called
