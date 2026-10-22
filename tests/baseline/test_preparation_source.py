"""Real disk source freezing and metadata copy, with hub network doubled."""

import json
import sys
import types

import pytest


def test_local_source_identity_changes_when_weight_bytes_change(tmp_path):
    from sllm.vllm_preparation import freeze_source

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3"}')
    weight = source / "model.safetensors"
    weight.write_bytes(b"first")
    first = freeze_source(
        str(source), revision=None, cache_dir=tmp_path / "cache"
    )
    weight.write_bytes(b"other")
    second = freeze_source(
        str(source), revision=None, cache_dir=tmp_path / "cache"
    )
    assert first["revision"].startswith("sha256:")
    assert first["revision"] != second["revision"]
    assert first["source"] == "local-snapshot"


def test_hub_source_uses_requested_repo_and_frozen_revision(
    monkeypatch, tmp_path
):
    from sllm.vllm_preparation import freeze_source

    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weight")
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    revision = "a" * 40
    frozen = freeze_source(
        "actual/source", revision=revision, cache_dir=tmp_path / "cache"
    )
    assert calls[0]["repo_id"] == "actual/source"
    assert calls[0]["revision"] == revision
    assert frozen["revision"] == revision
    assert frozen["source"] == "actual/source"


@pytest.mark.parametrize("revision", [None, "main", "latest", "abc"])
def test_hub_refuses_floating_or_missing_revision_before_network(
    tmp_path, revision
):
    from sllm.vllm_preparation import freeze_source

    with pytest.raises(ValueError, match="revision"):
        freeze_source(
            "repo/model", revision=revision, cache_dir=tmp_path / "cache"
        )


def test_copy_metadata_excludes_weights_and_rechecks_frozen_source(tmp_path):
    from sllm.vllm_preparation import copy_source_metadata, freeze_source

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (source / "tokenizer.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weight")
    frozen = freeze_source(
        str(source), revision=None, cache_dir=tmp_path / "cache"
    )
    output = tmp_path / "output"
    output.mkdir()
    copy_source_metadata(frozen, output)
    assert (output / "config.json").read_bytes() == (
        source / "config.json"
    ).read_bytes()
    assert (output / "tokenizer.json").exists()
    assert not (output / "model.safetensors").exists()
    (source / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        copy_source_metadata(frozen, output)
