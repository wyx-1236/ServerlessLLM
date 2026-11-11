"""Real Python packaging with native extension compilation explicitly replaced."""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import venv
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def run_python(script, cwd, *args, python=sys.executable):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(python), "-c", textwrap.dedent(script), *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_store_preparation_imports_without_main_package(tmp_path):
    shutil.copytree(ROOT / "sllm_store/sllm_store", tmp_path / "sllm_store")
    run_python(
        """
        import importlib.abc
        import sys
        class BlockExecution(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'sllm', 'torch', 'vllm', 'ray'}:
                    raise AssertionError('unexpected dependency: ' + fullname)
        sys.meta_path.insert(0, BlockExecution())
        from sllm_store import vllm_preparation, vllm_prepare_worker
        assert callable(vllm_preparation.prepare_vllm_model)
        assert callable(vllm_prepare_worker.main)
    """,
        tmp_path,
    )


BUILD_SCRIPT = """
import os
import subprocess
import sys
import types
import setuptools
import setuptools.build_meta as backend

# Only the existing CUDA discovery and native build are replaced. Real setup.py,
# metadata, build_py, sdist, wheel creation and package data rules run normally.
extension = types.ModuleType('torch.utils.cpp_extension')
extension.CUDA_HOME = '/cpu-test-only'
extension.ROCM_HOME = None
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch.utils'] = types.ModuleType('torch.utils')
sys.modules['torch.utils.cpp_extension'] = extension
original_check = subprocess.check_output
def check(command, **kwargs):
    if command == ['/cpu-test-only/bin/nvcc', '-V']:
        return 'CPU packaging fixture'
    return original_check(command, **kwargs)
subprocess.check_output = check
original_setup = setuptools.setup
def setup(**kwargs):
    assert len(kwargs['ext_modules']) == 2
    kwargs['ext_modules'] = []
    kwargs['cmdclass'].pop('build_ext')
    kwargs['cmdclass'].pop('install')
    original_setup(**kwargs)
setuptools.setup = setup
# PEP 517 frontends need not put the project on their import path.
sys.path = [path for path in sys.path if path and os.path.abspath(path) != os.getcwd()]
command, destination = sys.argv[1], sys.argv[-1]
if command == 'sdist':
    backend.build_sdist(destination)
else:
    assert command == 'bdist_wheel'
    backend.build_wheel(destination)
"""


@pytest.mark.parametrize("wheel_source", ["checkout", "sdist"])
def test_sdist_to_wheel_preserves_standalone_preparation(
    tmp_path, wheel_source
):
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    checkout = tmp_path / "checkout"
    source = checkout / "sllm_store"
    shutil.copytree(
        ROOT / "sllm_store",
        source,
        ignore=shutil.ignore_patterns(
            "__pycache__", "build", "dist", "*.egg-info", "*.so"
        ),
    )
    shutil.copyfile(ROOT / "uv.lock", checkout / "uv.lock")
    expected = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    run_python(BUILD_SCRIPT, source, "sdist", "--dist-dir", tmp_path / "sdist")
    archive = next((tmp_path / "sdist").glob("*.tar.gz"))
    unpacked = tmp_path / "unpacked"
    with tarfile.open(archive) as package:
        package.extractall(unpacked, filter="data")
    isolated = next(unpacked.iterdir())
    assert (
        isolated / "sllm_store/preparation-lock.sha256"
    ).read_text().strip() == expected
    assert not (unpacked / "uv.lock").exists()
    run_python(
        BUILD_SCRIPT,
        source if wheel_source == "checkout" else isolated,
        "bdist_wheel",
        "--dist-dir",
        tmp_path / "wheel",
    )
    wheel = next((tmp_path / "wheel").glob("*.whl"))
    assert not (source / "sllm_store/preparation-lock.sha256").exists()
    with zipfile.ZipFile(wheel) as package:
        names = package.namelist()
        assert "sllm_store/vllm_preparation.py" in names
        assert "sllm_store/vllm_prepare_worker.py" in names
        assert (
            package.read("sllm_store/preparation-lock.sha256").decode().strip()
            == expected
        )
        assert not any(name.startswith("sllm/") for name in names)
        entry = next(
            name for name in names if name.endswith("/entry_points.txt")
        )
        assert (
            "sllm-store = sllm_store.cli:main" in package.read(entry).decode()
        )
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    interpreter = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    installation = subprocess.run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert installation.returncode == 0, (
        installation.stdout + installation.stderr
    )
    run_python(
        """
        import importlib.abc
        import sys
        from pathlib import Path
        class BlockExecution(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'sllm', 'torch', 'vllm', 'ray'}:
                    raise AssertionError('unexpected dependency: ' + fullname)
        sys.meta_path.insert(0, BlockExecution())
        from sllm_store import vllm_preparation as preparation
        from sllm_store._build_provenance import dependency_lock_sha256
        from sllm_store.vllm_export import export_vllm_model
        assert dependency_lock_sha256() == sys.argv[1]
        calls = []
        preparation.prepare_vllm_model = lambda *a, **kw: calls.append((a, kw))
        export_vllm_model('org/model', 'float16', storage_path='models', local_model_path='.')
        assert calls[0][0] == ('org/model', '.', 'float16', 1)
    """,
        tmp_path,
        expected,
        python=interpreter,
    )
    # Exercise the exact module entry used by export, verifier and watchdog.
    result = subprocess.run(
        [str(interpreter), "-m", "sllm_store.vllm_prepare_worker", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "PYTHONPATH"
        },
    )
    assert result.returncode == 0, result.stderr
    assert "{export,verify,store}" in result.stdout


def test_packaged_provenance_is_independent_of_neighboring_checkout(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "sllm_store"))
    from sllm_store._build_provenance import dependency_lock_sha256

    package = tmp_path / "installed/sllm_store"
    package.mkdir(parents=True)
    (package / "preparation-lock.sha256").write_text("a" * 64 + "\n")
    (tmp_path / "uv.lock").write_text("unrelated checkout")
    assert dependency_lock_sha256(package) == "a" * 64


def test_legacy_preparation_imports_share_the_store_implementation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "sllm_store"))
    import sllm.vllm_preparation as old_preparation
    import sllm.vllm_prepare_worker as old_worker
    import sllm_store.vllm_preparation as preparation
    import sllm_store.vllm_prepare_worker as worker

    assert old_preparation is preparation
    assert old_worker is worker


@pytest.mark.parametrize("contents", [None, "", "main", "a" * 63])
def test_missing_or_invalid_packaged_provenance_fails(
    tmp_path, contents, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "sllm_store"))
    from sllm_store._build_provenance import dependency_lock_sha256

    package = tmp_path / "installed/sllm_store"
    package.mkdir(parents=True)
    if contents is not None:
        (package / "preparation-lock.sha256").write_text(contents)
    with pytest.raises((FileNotFoundError, ValueError)):
        dependency_lock_sha256(package)
