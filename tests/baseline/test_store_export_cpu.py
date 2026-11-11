"""Public export entry points with real parsing and CPU preparation doubles."""

import importlib.util
import logging
import os
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "sllm_store/examples/save_vllm_model.py"


@pytest.fixture
def export_boundary(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "sllm_store"))
    from sllm_store import vllm_preparation

    prepare = Mock(return_value={"artifact": "verified"})
    monkeypatch.setattr(vllm_preparation, "prepare_vllm_model", prepare)
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = Mock()
    hub.HfApi.return_value.model_info.return_value.sha = "a" * 40
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("STORAGE_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    return prepare, hub, source


@pytest.fixture
def store_cli(monkeypatch, export_boundary):
    pytest.importorskip("click")
    # Only execution dependencies are replaced. Click and the export adapter
    # run normally, without importing CUDA or starting a Store service.
    modules = {
        "torch": {"float16": "float16"},
        "sllm_store.server": {"serve": Mock()},
        "sllm_store.utils": {"to_num_bytes": Mock()},
        "sllm_store.transformers": {
            name: Mock()
            for name in ("save_model", "save_lora", "load_model", "load_lora")
        },
        "transformers": {
            name: Mock()
            for name in (
                "AutoModelForCausalLM",
                "AutoConfig",
                "AutoTokenizer",
                "BitsAndBytesConfig",
            )
        },
        "peft": {"PeftModel": Mock()},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "store_cli_cpu_probe", ROOT / "sllm_store/sllm_store/cli.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


@pytest.mark.parametrize("path_mode", ["explicit", "environment", "default"])
def test_store_save_preserves_arguments_and_destination(
    store_cli, export_boundary, monkeypatch, tmp_path, path_mode
):
    from click.testing import CliRunner

    prepare, hub, source = export_boundary
    args = [
        "save",
        "--model",
        "alias/model",
        "--backend",
        "vllm",
        "--local-model-path",
        str(source),
        "--tensor-parallel-size",
        "2",
    ]
    expected = tmp_path / "models"
    if path_mode != "default":
        monkeypatch.setenv("STORAGE_PATH", "~/environment")
        expected = tmp_path / "environment"
    if path_mode == "explicit":
        args += ["--storage-path", "~/explicit"]
        expected = tmp_path / "explicit"
    result = CliRunner().invoke(store_cli.cli, args)
    assert result.exit_code == 0, result.output
    prepare.assert_called_once_with(
        "alias/model",
        str(source),
        "float16",
        2,
        storage_path=str(expected / "vllm"),
        revision=None,
        flat_layout=True,
    )
    hub.HfApi.assert_not_called()


def test_store_save_freezes_default_hub_revision(store_cli, export_boundary):
    from click.testing import CliRunner

    prepare, hub, _ = export_boundary
    result = CliRunner().invoke(
        store_cli.cli, ["save", "--model", "org/model", "--backend", "vllm"]
    )
    assert result.exit_code == 0, result.output
    hub.HfApi.return_value.model_info.assert_called_once_with("org/model")
    assert prepare.call_args.args[:2] == ("org/model", "org/model")
    assert prepare.call_args.kwargs["revision"] == "a" * 40


@pytest.mark.parametrize("revision", [None, "main", "a" * 39])
def test_unresolved_hub_revision_never_prepares(
    store_cli, export_boundary, revision
):
    from click.testing import CliRunner

    prepare, hub, _ = export_boundary
    hub.HfApi.return_value.model_info.return_value.sha = revision
    result = CliRunner().invoke(
        store_cli.cli, ["save", "--model", "org/model", "--backend", "vllm"]
    )
    assert result.exit_code == 1
    prepare.assert_not_called()


def test_missing_local_source_never_falls_back_to_hub(
    store_cli, export_boundary, tmp_path
):
    prepare, hub, _ = export_boundary
    with pytest.raises(RuntimeError, match="local model"):
        store_cli.VllmModelDownloader().download_vllm_model(
            "org/model", "float16", local_model_path=str(tmp_path / "missing")
        )
    hub.HfApi.assert_not_called()
    prepare.assert_not_called()


def test_prepare_failure_preserves_existing_output(
    store_cli, export_boundary, tmp_path
):
    from click.testing import CliRunner

    prepare, _, source = export_boundary
    target = tmp_path / "vllm/org/model"
    target.mkdir(parents=True)
    previous = target / "keep"
    previous.write_bytes(b"existing artifact")
    prepare.side_effect = ValueError("verification failed")
    result = CliRunner().invoke(
        store_cli.cli,
        [
            "save",
            "--model",
            "org/model",
            "--backend",
            "vllm",
            "--storage-path",
            str(tmp_path),
            "--local-model-path",
            str(source),
        ],
    )
    assert result.exit_code == 1
    prepare.assert_called_once()
    assert previous.read_bytes() == b"existing artifact"


def test_hub_failure_is_reported_without_preparation(
    store_cli, export_boundary
):
    prepare, hub, _ = export_boundary
    failure = OSError("Hub is unavailable")
    hub.HfApi.return_value.model_info.side_effect = failure
    with pytest.raises(RuntimeError, match="Hub is unavailable") as caught:
        store_cli.VllmModelDownloader().download_vllm_model(
            "org/model", "float16"
        )
    assert caught.value.__cause__ is failure
    prepare.assert_not_called()


@pytest.mark.parametrize("options", [{"pattern": "weight"}, {"max_size": 0}])
def test_legacy_export_parameters_remain_accepted(
    store_cli, export_boundary, options, caplog, monkeypatch
):
    prepare, _, source = export_boundary
    monkeypatch.setattr(logging.getLogger("sllm_store"), "propagate", True)
    with caplog.at_level(logging.WARNING):
        result = store_cli.VllmModelDownloader().download_vllm_model(
            "org/model", "bfloat16", local_model_path=str(source), **options
        )
    assert result is None
    prepare.assert_called_once()
    assert "ignored" in caplog.text


@pytest.mark.parametrize("path_mode", ["explicit", "environment", "default"])
def test_export_example_preserves_layout_and_path_priority(
    export_boundary, monkeypatch, tmp_path, path_mode
):
    prepare, _, source = export_boundary
    args = [
        str(EXAMPLE),
        "--model-name",
        "org/model",
        "--local-model-path",
        str(source),
        "--tensor-parallel-size",
        "2",
    ]
    expected = tmp_path / "models"
    if path_mode in {"explicit", "environment"}:
        args += ["--storage-path", str(tmp_path / "explicit")]
        expected = tmp_path / "explicit"
    if path_mode == "environment":
        expected = tmp_path / "environment"
        monkeypatch.setenv("STORAGE_PATH", str(expected))
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    prepare.assert_called_once_with(
        "org/model",
        str(source),
        "float16",
        2,
        storage_path=str(expected),
        revision=None,
        flat_layout=True,
    )


def test_store_commands_and_save_options_preserved(store_cli):
    assert set(store_cli.cli.commands) == {"start", "save", "load"}
    assert {option.name for option in store_cli.save.params} == {
        "model_name",
        "backend",
        "adapter_name",
        "tensor_parallel_size",
        "local_model_path",
        "storage_path",
    }


def test_transformers_save_still_uses_existing_path(
    store_cli, export_boundary, tmp_path
):
    from click.testing import CliRunner

    prepare, hub, _ = export_boundary
    result = CliRunner().invoke(
        store_cli.cli,
        [
            "save",
            "--model",
            "org/model",
            "--backend",
            "transformers",
            "--storage-path",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    model = store_cli.AutoModelForCausalLM.from_pretrained.return_value
    store_cli.save_model.assert_called_once_with(
        model, os.path.join(str(tmp_path), "org/model")
    )
    prepare.assert_not_called()
    hub.HfApi.assert_not_called()
