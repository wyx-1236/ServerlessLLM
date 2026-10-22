"""Bounded Linux subprocess ownership for model preparation.

This supervisor does not request GPU resources. Its callers may launch engine
work only within a separately approved server/GPU batch.
"""

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}


def _source_inventory(root):
    inventory = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", ".cache"} for part in relative.parts):
            continue
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("Source directory symlinks are not frozen")
            continue
        if not path.is_file():
            raise ValueError(f"Source contains non-regular file: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        inventory[relative.as_posix()] = {
            "size": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    if "config.json" not in inventory:
        raise ValueError("Frozen model source has no config.json")
    if not any(Path(name).suffix in WEIGHT_SUFFIXES for name in inventory):
        raise ValueError("Frozen model source has no model weights")
    return inventory


def freeze_source(source, *, revision, cache_dir):
    """Freeze source identity without importing an engine or requesting a GPU.

    Hub sources require an immutable commit. Local snapshots are identified by
    their actual complete file digests, not by an unverified claimed Hub ref.
    Hub cache file symlinks are hashed by contents and materialized on copy.
    """
    local = Path(source)
    if local.exists():
        if not local.is_dir():
            raise ValueError("Model source must be a directory")
        if revision is not None:
            raise ValueError(
                "Local sources use content revision, omit Hub revision"
            )
        root = local.resolve()
        source_id = "local-snapshot"
    else:
        if not isinstance(revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", revision
        ):
            raise ValueError(
                "Hub source requires a fixed 40-character revision"
            )
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=source,
                revision=revision,
                cache_dir=str(cache_dir),
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.tiktoken",
                    "*.py",
                    "*.jinja",
                    "*.md",
                    "*.safetensors",
                    "*.bin",
                    "*.pt",
                    "*.pth",
                    "*.ckpt",
                ],
            )
        ).resolve()
        source_id = source
    files = _source_inventory(root)
    digest = hashlib.sha256(
        json.dumps(
            files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "path": str(root),
        "source": source_id,
        "revision": revision
        if source_id != "local-snapshot"
        else f"sha256:{digest}",
        "snapshot_sha256": digest,
        "files": files,
    }


def check_frozen_source(snapshot):
    if _source_inventory(Path(snapshot["path"])) != snapshot["files"]:
        raise ValueError("Frozen source changed during preparation")
    digest = hashlib.sha256(
        json.dumps(
            snapshot["files"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if snapshot.get("snapshot_sha256") != digest:
        raise ValueError("Frozen source manifest digest mismatch")


def copy_source_metadata(snapshot, output):
    """Copy config/tokenizer/code into an owned staging path, never weights."""
    check_frozen_source(snapshot)
    root = Path(snapshot["path"])
    output = Path(output)
    for name in snapshot["files"]:
        relative = Path(name)
        if relative.suffix in WEIGHT_SUFFIXES:
            continue
        if relative.parts[0].startswith("rank_") or relative.parts[0] in {
            "artifact-manifest.json",
            "artifact-commit.json",
        }:
            raise ValueError(
                "Source metadata conflicts with Store artifact layout"
            )
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)


def _group_has_live_processes(group_id):
    # Ignore zombies: they have no CUDA context or open resources to release.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def _stop_group(group_id, timeout):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group_id, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while _group_has_live_processes(group_id):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        else:
            return
    raise RuntimeError(f"Preparation process group {group_id} failed cleanup")


def run_preparation_process(
    command,
    log_path,
    *,
    timeout,
    cleanup_timeout=5,
    env=None,
    cwd=None,
):
    """Wait for one owned process; stop its group on success/failure/timeout.

    Export and independent restore use separate invocations, so the original
    engine must be gone before restore starts. GPU qualification must also check
    actual CUDA clients/allocations: a process that deliberately changes its
    session is outside the guarantees of POSIX process-group cleanup.
    """
    if sys.platform != "linux":
        raise RuntimeError("Model preparation requires the Linux runtime")
    if timeout <= 0 or cleanup_timeout <= 0:
        raise ValueError("Preparation deadlines must be positive")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        timed_out = False
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            try:
                _stop_group(process.pid, cleanup_timeout)
            finally:
                # Reap the owned leader after killing its group as well.
                process.wait(timeout=cleanup_timeout)
        if timed_out:
            raise RuntimeError(f"Preparation timed out; see {log_path}")
        if process.returncode:
            raise RuntimeError(
                f"Preparation exit code {process.returncode}; see {log_path}"
            )


def _read_record(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid preparation record: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid preparation record: {path}")
    return value


def _write_record(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_vllm_model(
    model_name,
    source,
    dtype,
    tensor_parallel_size=1,
    *,
    storage_path,
    revision=None,
    source_snapshot=None,
    engine_options=None,
    validation=None,
    timeout=1800,
):
    """Prepare/reuse one immutable artifact through isolated native processes.

    This may use GPUs and must only be invoked within an explicitly approved
    batch. Source freezing and namespace capability can be prepared beforehand.
    A successful return does not release a caller's GPU lock or advance a stage.
    """
    from sllm_store.artifact import ArtifactTransaction, validate_artifact
    from sllm_store.vllm_plugin import (
        ENGINE_VERSION,
        LAYOUT_ID,
        PLUGIN_ID,
        WRITER_ID,
        require_version,
    )

    if (
        not isinstance(model_name, str)
        or not model_name
        or model_name.startswith("/")
        or "\\" in model_name
        or ":" in model_name
        or any(part in {"", ".", ".."} for part in model_name.split("/"))
    ):
        raise ValueError("Invalid model name")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Preparation requires an explicit supported dtype")
    if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError("Invalid tensor parallel size")
    if not math.isfinite(timeout) or not 0 < timeout <= 1800:
        raise ValueError("Preparation timeout must be within 1800 seconds")
    require_version()  # Distribution metadata only; no engine/CUDA import.
    deadline = time.monotonic() + timeout

    def remaining():
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise RuntimeError("Preparation exceeded its overall deadline")
        return budget

    options = dict(
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=64 * 1024**2,
        enforce_eager=True,
        enable_prefix_caching=True,
        seed=0,
        distributed_executor_backend="mp",
    )
    if engine_options is not None:
        if not isinstance(engine_options, dict) or set(engine_options) - set(
            options
        ):
            raise ValueError("Unsupported preparation engine options")
        options.update(engine_options)
    if (
        options["dtype"] != dtype
        or options["tensor_parallel_size"] != tensor_parallel_size
    ):
        raise ValueError("Preparation engine identity conflict")
    from sllm.vllm_prepare_worker import SAMPLING, STORE, validate_request

    validation = (
        validation
        if validation is not None
        else dict(
            prompts=["The capital of France is", "1 + 1 ="],
            sampling=dict(SAMPLING),
            logprob_atol=0,
        )
    )
    # Freeze JSON values, including caller-owned nested dictionaries.
    options, validation = json.loads(
        json.dumps(
            [options, validation],
            allow_nan=False,
        )
    )
    storage = Path(storage_path).expanduser().resolve()
    target = storage / "vllm" / model_name
    snapshot = None
    if source_snapshot is not None:
        snapshot = (
            _read_record(source_snapshot)
            if isinstance(source_snapshot, (str, os.PathLike))
            else json.loads(json.dumps(source_snapshot, allow_nan=False))
        )
        if (
            not isinstance(snapshot, dict)
            or set(snapshot)
            != {"path", "source", "revision", "snapshot_sha256", "files"}
            or not Path(snapshot["path"]).is_absolute()
        ):
            raise ValueError("Invalid frozen source receipt")
        check_frozen_source(snapshot)
        source_identity = snapshot["source"]
        frozen_revision = snapshot["revision"]
        if (
            not isinstance(source_identity, str)
            or not source_identity
            or not isinstance(frozen_revision, str)
            or (
                source_identity == "local-snapshot"
                and frozen_revision != f"sha256:{snapshot['snapshot_sha256']}"
            )
            or (
                source_identity != "local-snapshot"
                and not re.fullmatch(r"[0-9a-f]{40}", frozen_revision)
            )
        ):
            raise ValueError("Invalid frozen source identity")
        if (
            source != snapshot["source"]
            and Path(source).resolve() != Path(snapshot["path"]).resolve()
        ):
            raise ValueError("Frozen source repository/path mismatch")
        if revision is not None and revision != snapshot["revision"]:
            raise ValueError("Frozen source revision mismatch")
        source_id, source_revision = snapshot["source"], snapshot["revision"]
    elif Path(source).exists():
        snapshot = freeze_source(
            source, revision=None, cache_dir=storage / ".source-cache"
        )
        if revision is not None and revision != snapshot["revision"]:
            raise ValueError("Local source content revision mismatch")
        source_id, source_revision = snapshot["source"], snapshot["revision"]
    elif isinstance(revision, str) and re.fullmatch(
        r"sha256:[0-9a-f]{64}", revision
    ):
        source_id, source_revision = "local-snapshot", revision
    elif (
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
        and not Path(source).is_absolute()
    ):
        source_id, source_revision = source, revision
    else:
        raise ValueError("Missing source or frozen source revision")
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    identity = dict(
        model=model_name,
        source=source_id,
        revision=source_revision,
        engine="vllm",
        engine_version=ENGINE_VERSION,
        dtype=f"torch.{dtype}",
        world_size=tensor_parallel_size,
        writer=WRITER_ID,
        plugin=PLUGIN_ID,
        layout=LAYOUT_ID,
        weight_phase="execution-ready",
        dependency_lock_sha256=_file_sha256(lock),
        engine_options=options,
        validation=validation,
    )
    if target.exists() or target.is_symlink():
        stored = _read_record(target / "artifact-manifest.json").get("identity")
        if not isinstance(stored, dict):
            raise ValueError("Existing target has no valid artifact identity")
        digest = stored.get("snapshot_sha256")
        if not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError("Existing artifact has no frozen source digest")
        identity["snapshot_sha256"] = (
            snapshot["snapshot_sha256"] if snapshot else digest
        )
        if (
            source_id == "local-snapshot"
            and source_revision != f"sha256:{identity['snapshot_sha256']}"
        ):
            raise ValueError("Existing local snapshot identity mismatch")
        return validate_artifact(target, identity)
    if snapshot is None:
        if source_id == "local-snapshot":
            raise ValueError("Local source is unavailable for fresh export")
        snapshot = freeze_source(
            source, revision=revision, cache_dir=storage / ".source-cache"
        )
    identity["snapshot_sha256"] = snapshot["snapshot_sha256"]
    records = storage / ".preparation-records" / uuid.uuid4().hex
    records.mkdir(parents=True)
    _write_record(records / "source.json", snapshot)
    if sys.platform != "linux":
        raise RuntimeError("Model preparation requires Linux namespace support")
    unshare = shutil.which("unshare")
    if unshare is None:
        raise RuntimeError("Offline verification requires util-linux unshare")
    namespace = os.stat("/proc/self/ns/net").st_ino
    prefix = [
        unshare,
        "--user",
        "--map-root-user",
        "--net",
        "--fork",
        "--kill-child=SIGKILL",
    ]
    probe = (
        "import os,socket,sys; "
        "assert os.stat('/proc/self/ns/net').st_ino != int(sys.argv[1]); "
        "assert sorted(name for _,name in socket.if_nameindex()) == ['lo']"
    )
    run_preparation_process(
        prefix + [sys.executable, "-c", probe, str(namespace)],
        records / "namespace-probe.log",
        timeout=remaining(),
    )
    owner_start = (
        Path(f"/proc/{os.getpid()}/stat")
        .read_text()
        .rsplit(")", 1)[1]
        .split()[19]
    )
    reference = records / "native-reference.json"
    with ArtifactTransaction(target) as transaction:
        copy_source_metadata(snapshot, transaction.path)
        common = dict(
            schema_version=1,
            artifact_path=str(transaction.path),
            identity=identity,
            engine_options=options,
            validation=validation,
            reference_path=str(reference),
            store=dict(STORE),
            owner_pid=os.getpid(),
            owner_start_time=owner_start,
        )

        def launch(mode, request, isolated=False):
            validate_request(mode, request)
            path = records / f"{mode}-request.json"
            _write_record(path, request)
            command = [
                sys.executable,
                "-m",
                "sllm.vllm_prepare_worker",
                mode,
                "--request",
                str(path),
            ]
            run_preparation_process(
                (prefix if isolated else []) + command,
                records / f"{mode}.log",
                timeout=remaining(),
            )
            remaining()
            result = _read_record(request["result_path"])
            if result.get("schema_version") != 1:
                raise ValueError("Invalid preparation result schema")
            return result

        exported = launch(
            "export",
            dict(
                common,
                source_path=snapshot["path"],
                result_path=str(records / "export-result.json"),
            ),
        )
        check_frozen_source(snapshot)
        reports = exported.get("rank_reports")
        if (
            not isinstance(reports, list)
            or len(reports) != tensor_parallel_size
        ):
            raise ValueError("Incomplete rank export reports")
        ranks = []
        for report in reports:
            if (
                not isinstance(report, dict)
                or any(
                    report.get(key) != identity[key]
                    for key in ("writer", "plugin", "layout", "weight_phase")
                )
                or report.get("world_size") != tensor_parallel_size
            ):
                raise ValueError("Rank export identity mismatch")
            ranks.append(report.get("rank"))
        if any(type(rank) is not int for rank in ranks) or set(ranks) != set(
            range(tensor_parallel_size)
        ):
            raise ValueError("Duplicate or missing exported rank")
        reference_digest = _file_sha256(reference)
        if exported.get("reference_sha256") != reference_digest:
            raise ValueError("Native reference digest mismatch")
        native = _read_record(reference)
        if any(
            native.get(key) != common[key]
            for key in ("identity", "engine_options", "validation")
        ):
            raise ValueError("Native reference configuration mismatch")

        def verify(path, manifest):
            result = launch(
                "verify",
                dict(
                    common,
                    candidate_manifest=manifest,
                    parent_network_namespace=namespace,
                    result_path=str(records / "verify-result.json"),
                ),
                isolated=True,
            )
            isolation = result.get("isolation", {})
            if (
                result.get("passed") is not True
                or result.get("tokens_verified") is not True
                or result.get("logprobs_verified") is not True
                or result.get("method")
                != "independent-offline-native-store-token-logprob-v1"
                or result.get("request_count") != len(validation["prompts"])
                or result.get("reference_sha256") != reference_digest
                or _file_sha256(reference) != reference_digest
                or isolation.get("parent_network_namespace") != namespace
                or type(isolation.get("network_namespace")) is not int
                or isolation["network_namespace"] == namespace
                or isolation.get("interfaces") != ["lo"]
            ):
                raise ValueError(
                    "Invalid independent restore verification receipt"
                )
            return result

        return transaction.commit(identity, verify)
