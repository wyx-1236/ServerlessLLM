"""Disk integrity and publication for versioned Store artifacts.

Tensor restoration is deliberately supplied by the preparation pipeline: byte
integrity alone cannot establish that an engine can load the exported state.
"""

import hashlib
import json
import math
import os
import shutil
import tempfile
import sys
from pathlib import Path

MANIFEST = "artifact-manifest.json"
COMMIT = "artifact-commit.json"
RESERVED = {MANIFEST, COMMIT}
ITEM_SIZES = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.float64": 8,
    "torch.complex64": 8,
    "torch.complex128": 16,
}


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid artifact metadata: {path}") from error


def _write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_no_replace(source, target):
    """Atomically publish without replacing even an empty foreign directory."""
    if os.name == "nt":
        # Windows rename already rejects an existing destination.
        os.rename(source, target)
        return
    if sys.platform != "linux":
        raise RuntimeError(
            "Atomic no-replace publication requires Linux or Windows"
        )
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-replace publication requires renameat2")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100, RENAME_NOREPLACE=1; fail closed on unsupported filesystems.
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Artifact contains non-regular file: {path}")
        name = path.relative_to(root).as_posix()
        if name not in RESERVED:
            result[name] = {
                "size": path.stat().st_size,
                "sha256": _digest(path),
            }
    return result


def _check_ranks(root, identity):
    world_size = identity.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Invalid artifact world_size")
    expected = {f"rank_{rank}" for rank in range(world_size)}
    actual = {path.name for path in root.glob("rank_*")}
    if actual != expected:
        raise ValueError(
            f"Artifact rank mismatch: expected {expected}, got {actual}"
        )
    for rank in sorted(expected):
        directory = root / rank
        if not (directory / "tensor_index.json").is_file():
            raise ValueError(f"Missing tensor index for rank {rank}")
        if not (directory / "tensor.data_0").is_file():
            raise ValueError(f"Missing tensor data for rank {rank}")
        _check_tensor_index(directory)


def _check_tensor_index(directory):
    partitions = list(directory.glob("tensor.data_*"))
    expected = {f"tensor.data_{number}" for number in range(len(partitions))}
    if {path.name for path in partitions} != expected:
        raise ValueError(f"Invalid tensor partition sequence: {directory}")
    total_size = sum(path.stat().st_size for path in partitions)
    index = _read_json(directory / "tensor_index.json")
    if not isinstance(index, dict) or not index:
        raise ValueError(f"Invalid tensor index: {directory}")
    ranges = set()
    for name, record in index.items():
        if not isinstance(record, list) or len(record) != 5:
            raise ValueError(f"Invalid tensor index record: {name}")
        offset, size, shape, stride, dtype = record
        if (
            type(offset) is not int
            or offset < 0
            or offset % 8
            or type(size) is not int
            or size < 0
            or offset + size > total_size
        ):
            raise ValueError(f"Tensor exceeds data files: {name}")
        if (
            not isinstance(shape, list)
            or not isinstance(stride, list)
            or len(shape) != len(stride)
            or any(type(x) is not int or x < 0 for x in shape + stride)
            or not isinstance(dtype, str)
            or dtype not in ITEM_SIZES
        ):
            raise ValueError(f"Invalid tensor layout: {name}")
        required = 0
        if math.prod(shape):
            required = ITEM_SIZES[dtype] * (
                1
                + sum(
                    (length - 1) * step
                    for length, step in zip(shape, stride, strict=True)
                )
            )
        if required > size:
            raise ValueError(f"Tensor view exceeds storage: {name}")
        if size:
            ranges.add((offset, offset + size))
    previous_end = 0
    for start, end in sorted(ranges):
        if start < previous_end:
            raise ValueError("Overlapping non-identical tensor storage ranges")
        previous_end = end


def _check_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("Invalid artifact identity")
    for key in (
        "model",
        "revision",
        "engine",
        "engine_version",
        "dtype",
        "writer",
    ):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"Missing artifact identity field: {key}")
    # Require JSON identities so manifests survive a process boundary unchanged.
    if json.loads(json.dumps(identity, allow_nan=False)) != identity:
        raise ValueError("Artifact identity must contain only JSON values")


def validate_candidate_artifact(path, expected_identity, candidate_manifest):
    """Validate an explicit transaction candidate for isolated restore only.

    This grants no publication/reuse status. It checks the same identity,
    ranks, index bounds and complete byte inventory as a committed artifact.
    """
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Artifact is not a regular directory: {root}")
    files = _inventory(root)
    if (
        not isinstance(candidate_manifest, dict)
        or candidate_manifest.get("schema_version") != 1
    ):
        raise ValueError("Unsupported candidate manifest schema")
    _check_identity(expected_identity)
    if candidate_manifest.get("identity") != expected_identity:
        raise ValueError("Artifact identity mismatch")
    _check_ranks(root, expected_identity)
    if candidate_manifest.get("files") != files:
        raise ValueError("Artifact file inventory or digest mismatch")
    return candidate_manifest


def validate_artifact(path, expected_identity):
    """Rehash every file before reuse; reject legacy/uncommitted directories."""
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Artifact is not a regular directory: {root}")
    # Reject links outside the artifact before reading any metadata.
    files = _inventory(root)
    manifest = _read_json(root / MANIFEST)
    marker = _read_json(root / COMMIT)
    if not isinstance(manifest, dict) or not isinstance(marker, dict):
        raise ValueError("Invalid artifact manifest or commit marker")
    if manifest.get("schema_version") != 1 or marker.get("schema_version") != 1:
        raise ValueError("Unsupported artifact schema")
    if marker.get("manifest_sha256") != _digest(root / MANIFEST):
        raise ValueError("Artifact commit digest mismatch")
    if manifest.get("identity") != expected_identity:
        raise ValueError("Artifact identity mismatch")
    _check_identity(expected_identity)
    _check_ranks(root, expected_identity)
    if manifest.get("files") != files:
        raise ValueError("Artifact file inventory or digest mismatch")
    receipt = manifest.get("verification")
    if not isinstance(receipt, dict) or receipt.get("passed") is not True:
        raise ValueError("Artifact has no successful restore verification")
    return manifest


class ArtifactTransaction:
    """Own one sibling staging directory; never overwrite an existing target."""

    def __init__(self, target):
        self.target = Path(os.path.abspath(target))
        self.path = None
        self._committed = False

    def __enter__(self):
        if self.path is not None:
            raise RuntimeError("Artifact transaction cannot be entered twice")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(
            tempfile.mkdtemp(
                prefix=f".{self.target.name}.prepare-",
                dir=self.target.parent,
            )
        )
        return self

    def commit(self, identity, verify_restore):
        """Publish only after caller's independent restore check succeeds.

        The callback receives (staging_path, candidate_manifest) and must return
        a JSON receipt with passed=true. It must implement the engine/tensor
        comparison; this module only protects disk identity and publication.
        """
        if self.path is None or self._committed:
            raise RuntimeError("Artifact transaction is not active")
        if self.target.exists() or self.target.is_symlink():
            raise FileExistsError(self.target)
        _check_identity(identity)
        _check_ranks(self.path, identity)
        if any((self.path / name).exists() for name in RESERVED):
            raise ValueError(
                "Staging directory already contains commit metadata"
            )
        files = _inventory(self.path)
        # Freeze a JSON copy: a callback must not mutate the intended identity.
        manifest = json.loads(
            json.dumps(
                {
                    "schema_version": 1,
                    "identity": identity,
                    "files": files,
                },
                allow_nan=False,
            )
        )
        receipt = verify_restore(self.path, json.loads(json.dumps(manifest)))
        if not isinstance(receipt, dict) or receipt.get("passed") is not True:
            raise ValueError("Artifact restore verification failed")
        if _inventory(self.path) != files:
            raise ValueError("Artifact changed during restore verification")
        manifest["verification"] = receipt
        _write_json(self.path / MANIFEST, manifest)
        _write_json(
            self.path / COMMIT,
            {
                "schema_version": 1,
                "manifest_sha256": _digest(self.path / MANIFEST),
            },
        )
        validate_artifact(self.path, identity)
        # Exporters serialize publication; this lock never claims GPU resources.
        lock = self.target.with_name(f".{self.target.name}.publish-lock")
        lock.mkdir()
        try:
            if self.target.exists() or self.target.is_symlink():
                raise FileExistsError(self.target)
            for payload in self.path.rglob("*"):
                if payload.is_file():
                    # Windows FlushFileBuffers requires a writable descriptor.
                    mode = "r+b" if os.name == "nt" else "rb"
                    with payload.open(mode) as stream:
                        os.fsync(stream.fileno())
            for directory in self.path.rglob("*"):
                if directory.is_dir():
                    _sync_directory(directory)
            _sync_directory(self.path)
            _publish_no_replace(self.path, self.target)
            self._committed = True
            _sync_directory(self.target.parent)
        finally:
            lock.rmdir()
        return manifest

    def __exit__(self, *_):
        if not self._committed and self.path is not None:
            shutil.rmtree(self.path)
