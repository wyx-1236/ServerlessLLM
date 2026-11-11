"""Use the checkout's Store Python package without building native extensions."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def store_python_source(monkeypatch):
    source = str(Path(__file__).parents[2] / "sllm_store")
    monkeypatch.syspath_prepend(source)
    # Real CPU child processes must resolve the same local package as pytest.
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", source + os.pathsep + inherited if inherited else source
    )
