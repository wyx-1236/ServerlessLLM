"""Include the preparation lock identity in wheels and standalone sdists."""

from pathlib import Path

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

from sllm_store._build_provenance import dependency_lock_sha256


def write_provenance(source, destination):
    digest = dependency_lock_sha256(source)
    path = Path(destination) / "preparation-lock.sha256"
    path.parent.mkdir(parents=True, exist_ok=True)
    # sdist may hard-link existing package data. Replace the build output only.
    path.unlink(missing_ok=True)
    path.write_text(digest + "\n", encoding="ascii")


class BuildPy(build_py):
    def run(self):
        super().run()
        write_provenance(
            self.get_package_dir("sllm_store"),
            Path(self.build_lib) / "sllm_store",
        )


class Sdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        source = self.get_finalized_command("build_py").get_package_dir(
            "sllm_store"
        )
        write_provenance(source, Path(base_dir) / "sllm_store")
