"""Actual Click parsing with service operations replaced, no server startup."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def cli(monkeypatch):
    pytest.importorskip("click")
    helpers = types.ModuleType("sllm.cli._cli_utils")
    for name in ("delete_model", "deploy_model", "show_status", "start_server"):
        setattr(helpers, name, Mock())
    monkeypatch.setitem(sys.modules, "sllm.cli._cli_utils", helpers)
    path = Path(__file__).parents[2] / "sllm/cli/clic.py"
    spec = importlib.util.spec_from_file_location("cpu_cli_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cli, helpers


def test_default_start_flags_are_preserved(cli):
    from click.testing import CliRunner

    command, helpers = cli
    result = CliRunner().invoke(command, ["start"])
    assert result.exit_code == 0, result.output
    helpers.start_server.assert_called_once_with(
        host="0.0.0.0",
        port=8343,
        enable_storage_aware=False,
        enable_migration=False,
    )


def test_explicit_start_flags_are_forwarded(cli):
    from click.testing import CliRunner

    command, helpers = cli
    result = CliRunner().invoke(
        command,
        [
            "start",
            "--host",
            "127.0.0.1",
            "--port",
            "9123",
            "--enable-storage-aware",
            "--enable-migration",
        ],
    )
    assert result.exit_code == 0, result.output
    helpers.start_server.assert_called_once_with(
        host="127.0.0.1",
        port=9123,
        enable_storage_aware=True,
        enable_migration=True,
    )


def test_invalid_port_never_starts_service(cli):
    from click.testing import CliRunner

    command, helpers = cli
    result = CliRunner().invoke(command, ["start", "--port", "not-a-number"])
    assert result.exit_code != 0
    helpers.start_server.assert_not_called()
