"""Dependency-lock identity for checkout and installed preparation runtimes."""

import hashlib
import re
from pathlib import Path


def dependency_lock_sha256(package_dir=None):
    package = (
        Path(package_dir) if package_dir is not None else Path(__file__).parent
    )
    receipt = package / "preparation-lock.sha256"
    if not receipt.exists():
        lock = package.resolve().parents[1] / "uv.lock"
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    digest = receipt.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid packaged preparation dependency-lock digest")
    return digest
