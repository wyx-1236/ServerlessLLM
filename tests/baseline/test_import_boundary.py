"""Importing configuration must never import an execution engine."""

import subprocess
import sys
from pathlib import Path


def test_configuration_import_cannot_load_gpu_dependencies():
    root = Path(__file__).parents[2]
    script = """
import importlib.abc
import sys
class NoEngineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','vllm','peft','transformers','ray'}:
            raise AssertionError('Unexpected execution dependency: ' + fullname)
sys.meta_path.insert(0, NoEngineImports())
import sllm.backends.vllm_config
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
