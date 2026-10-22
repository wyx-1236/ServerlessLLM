"""Isolated, explicitly invoked vLLM 0.31 export/restore process.

Importing this module is CPU-only. Calling ``run`` is a GPU execution boundary
and is reserved for an approved batch; it is never an import-time probe.
"""

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

SAMPLING = {
    "temperature": 0,
    "max_tokens": 8,
    "ignore_eos": False,
    "logprobs": 5,
}
STORE = {
    "mem_pool_size": 2 * 1024**3,
    "chunk_size": 64 * 1024**2,
    "num_thread": 2,
}
ENGINE_OPTIONS = {
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "kv_cache_memory_bytes",
    "enforce_eager",
    "enable_prefix_caching",
    "seed",
    "distributed_executor_backend",
}
TOKENIZATION = {"add_special_tokens": True}


def _external(path, artifact):
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.resolve().is_relative_to(
        artifact.resolve()
    ):
        raise ValueError("Record paths must be absolute and outside artifact")
    return candidate


def validate_request(mode, request):
    if mode not in {"export", "verify", "store"} or not isinstance(
        request, dict
    ):
        raise ValueError("Invalid worker mode/request")
    common = {
        "schema_version",
        "artifact_path",
        "identity",
        "engine_options",
        "validation",
        "reference_path",
        "result_path",
        "store",
        "owner_pid",
        "owner_start_time",
    }
    required = common | (
        {"source_path"}
        if mode == "export"
        else {"candidate_manifest", "parent_network_namespace"}
    )
    if mode == "store":
        required = {"schema_version", "artifact_path", "store", "port"}
    if set(request) != required or request.get("schema_version") != 1:
        raise ValueError("Unsupported request fields/schema version")
    artifact = Path(request["artifact_path"])
    if (
        not artifact.is_absolute()
        or not artifact.is_dir()
        or artifact.is_symlink()
    ):
        raise ValueError("artifact_path must be an existing absolute directory")
    if request["store"] != STORE:
        raise ValueError("Unsupported private Store budget")
    if mode == "store":
        if (
            type(request["port"]) is not int
            or not 0 < request["port"] < 65536
            or request["port"] == 8073
        ):
            raise ValueError("Invalid private Store port")
        return request
    if (
        type(request["owner_pid"]) is not int
        or request["owner_pid"] <= 0
        or not isinstance(request["owner_start_time"], str)
        or not request["owner_start_time"].isdigit()
    ):
        raise ValueError("Invalid owner process identity")
    for key in ("reference_path", "result_path"):
        _external(request[key], artifact)
    if request["reference_path"] == request["result_path"]:
        raise ValueError("Reference/result paths must differ")
    if mode == "export":
        source = Path(request["source_path"])
        if not source.is_absolute() or not source.is_dir():
            raise ValueError("source_path must be an absolute frozen directory")
    elif not isinstance(request["candidate_manifest"], dict):
        raise ValueError("Invalid candidate manifest")
    options = request["engine_options"]
    if not isinstance(options, dict) or set(options) - ENGINE_OPTIONS:
        raise ValueError("Unsupported engine options")
    for key in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_memory_bytes",
    ):
        if key in options and (
            type(options[key]) is not int or options[key] <= 0
        ):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("enforce_eager", "enable_prefix_caching"):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if "seed" in options and (
        type(options["seed"]) is not int or options["seed"] < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    if options.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Explicit supported dtype required")
    identity = request["identity"]
    if not isinstance(identity, dict) or options.get(
        "tensor_parallel_size"
    ) != identity.get("world_size"):
        raise ValueError("Engine TP/identity mismatch")
    if identity.get("dtype") != "torch." + options["dtype"]:
        raise ValueError("Engine dtype/identity mismatch")
    if options.get("distributed_executor_backend", "mp") != "mp":
        raise ValueError("Only mp executor supported")
    validation = request["validation"]
    if not isinstance(validation, dict) or set(validation) != {
        "prompts",
        "sampling",
        "logprob_atol",
    }:
        raise ValueError("Invalid validation specification")
    if validation["sampling"] != SAMPLING or validation["logprob_atol"] != 0:
        raise ValueError(
            "Validation sampling/tolerance must match frozen profile"
        )
    prompts = validation["prompts"]
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p for p in prompts)
    ):
        raise ValueError("Validation requires nonempty raw prompts")
    json.dumps(request, allow_nan=False)
    return request


def serialize_outputs(outputs):
    records = []
    for output in outputs:
        if (
            not output.finished
            or len(output.outputs) != 1
            or not output.prompt_token_ids
        ):
            raise ValueError("Incomplete or non-single generation output")
        completion = output.outputs[0]
        if any(
            type(token) is not int or token < 0
            for token in output.prompt_token_ids
        ):
            raise ValueError("Invalid prompt token IDs")
        if (
            not completion.token_ids
            or completion.logprobs is None
            or len(completion.logprobs) != len(completion.token_ids)
        ):
            raise ValueError("Missing token/logprob output")
        if any(
            type(token) is not int or token < 0
            for token in completion.token_ids
        ):
            raise ValueError("Invalid generated token IDs")
        if completion.finish_reason not in {"length", "stop"}:
            raise ValueError("Unsupported generation finish reason")
        count = len(completion.token_ids)
        if count > SAMPLING["max_tokens"] or (
            completion.finish_reason == "length"
            and count != SAMPLING["max_tokens"]
        ):
            raise ValueError("Generation token count violates frozen sampling")
        probabilities = []
        for selected, position in zip(
            completion.token_ids, completion.logprobs
        ):
            if (
                not isinstance(position, dict)
                or len(position) < SAMPLING["logprobs"]
                or selected not in position
            ):
                raise ValueError("Incomplete top logprobs/selected token entry")
            for token, value in position.items():
                if type(token) is not int or token < 0:
                    raise ValueError("Invalid top logprob token IDs")
                rank = getattr(value, "rank", None)
                if type(rank) is not int or rank <= 0:
                    raise ValueError("Missing or invalid top logprob rank")
            values = {
                str(token): {
                    "logprob": float(value.logprob),
                    "rank": value.rank,
                }
                for token, value in position.items()
            }
            if not all(
                math.isfinite(value["logprob"]) for value in values.values()
            ):
                raise ValueError("Nonfinite logprob")
            probabilities.append(values)
        records.append(
            dict(
                prompt_token_ids=list(output.prompt_token_ids),
                token_ids=list(completion.token_ids),
                logprobs=probabilities,
                finish_reason=completion.finish_reason,
                stop_reason=completion.stop_reason,
            )
        )
    return records


def validate_prompt_tokens(tokenizer, prompts, outputs):
    if len(prompts) != len(outputs):
        raise ValueError("Raw prompt/tokenizer reference count mismatch")
    for prompt, output in zip(prompts, outputs):
        encoded = tokenizer.encode(prompt, **TOKENIZATION)
        if encoded != output["prompt_token_ids"]:
            raise ValueError(
                "Raw prompt tokenizer encoding differs from native reference"
            )


def compare_outputs(reference, actual, atol):
    if atol != 0 or len(reference) != len(actual) or not reference:
        raise ValueError("Invalid reference count/tolerance")
    for expected, received in zip(reference, actual):
        for field in ("prompt_token_ids", "token_ids"):
            if expected[field] != received[field]:
                raise ValueError(f"Generation token mismatch: {field}")
        if any(
            expected[field] != received[field]
            for field in ("finish_reason", "stop_reason")
        ):
            raise ValueError("Generation EOS/stop semantics mismatch")
        if expected["logprobs"] != received["logprobs"]:
            raise ValueError("Generation top logprob mismatch (frozen atol=0)")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _engine_api():
    from sllm_store.vllm_plugin import require_version

    require_version()
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def _shutdown(engine):
    # v0.31.0 LLM has no public shutdown; its fixed EngineCoreClient does:
    # https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/core_client.py
    # shutdown(self, timeout: float | None = None) -> None
    engine.llm_engine.engine_core.shutdown(timeout=10)


def _export(request):
    llm, sampling = _engine_api()
    engine = llm(
        model=request["source_path"],
        tokenizer=request["source_path"],
        worker_extension_cls="sllm_store.vllm_plugin.ServerlessLLMWorkerExtension",
        **request["engine_options"],
    )
    try:
        outputs = serialize_outputs(
            engine.generate(
                request["validation"]["prompts"],
                sampling(**SAMPLING),
                use_tqdm=False,
                tokenization_kwargs=TOKENIZATION,
            )
        )
        if len(outputs) != len(request["validation"]["prompts"]):
            raise ValueError("Reference request count mismatch")
        validate_prompt_tokens(
            engine.get_tokenizer(), request["validation"]["prompts"], outputs
        )
        reference = dict(
            schema_version=1,
            identity=request["identity"],
            engine_options=request["engine_options"],
            validation=request["validation"],
            eos_token_id=engine.get_tokenizer().eos_token_id,
            outputs=outputs,
        )
        # Native reference comes from execution-ready state before named RPC export.
        reports = engine.collective_rpc(
            "export_serverless_model", args=(request["artifact_path"],)
        )
        _write_json(request["reference_path"], reference)
        return dict(
            schema_version=1,
            rank_reports=reports,
            reference_sha256=_digest(request["reference_path"]),
        )
    finally:
        _shutdown(engine)


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except (ValueError, AttributeError):
        return False


def offline_audit(event, args):
    """Python audit boundary; OS network isolation also covers native libraries."""
    if event == "socket.getaddrinfo" and not _loopback(args[0]):
        raise PermissionError("Offline restore refuses non-loopback DNS")
    if event == "socket.connect":
        sock, address = args
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _loopback(
            address[0]
        ):
            raise PermissionError(
                "Offline restore refuses non-loopback connection"
            )
    if event == "open" and os.environ.get("SLLM_VERIFY_ARTIFACT"):
        name, mode, flags = args
        if isinstance(name, (str, bytes)):
            path = Path(os.fsdecode(name)).resolve()
            artifact = Path(os.environ["SLLM_VERIFY_ARTIFACT"])
            inside = path.is_relative_to(artifact)
            if inside and (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            ):
                raise PermissionError("Restore artifact is read-only")
            if not inside and path.suffix in {
                ".safetensors",
                ".bin",
                ".pt",
                ".pth",
                ".ckpt",
                ".gguf",
            }:
                raise PermissionError(
                    "Offline restore refuses external original weights"
                )


def check_network_namespace(
    parent_inode, *, namespace_inode=None, interfaces=None
):
    if type(parent_inode) is not int or parent_inode <= 0:
        raise RuntimeError("Missing parent network namespace identity")
    if namespace_inode is None:
        if sys.platform != "linux":
            raise RuntimeError(
                "Offline verification requires Linux network namespace"
            )
        namespace_inode = os.stat("/proc/self/ns/net").st_ino
    if namespace_inode == parent_inode:
        raise RuntimeError(
            "Verification must use an independent network namespace"
        )
    if interfaces is None:
        interfaces = socket.if_nameindex()
    names = sorted(name for _, name in interfaces)
    if names != ["lo"]:
        raise RuntimeError(
            "Offline namespace must have only loopback interfaces"
        )
    return dict(
        parent_network_namespace=parent_inode,
        network_namespace=namespace_inode,
        interfaces=names,
    )


def _enable_loopback():
    # SIOCGIFFLAGS/SIOCSIFFLAGS, IFF_UP: private netns CAP_NET_ADMIN only.
    import fcntl
    import struct

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
        value = fcntl.ioctl(
            control.fileno(), 0x8913, struct.pack("16sH14x", b"lo", 0)
        )
        flags = struct.unpack("16sH14x", value)[1]
        fcntl.ioctl(
            control.fileno(), 0x8914, struct.pack("16sH14x", b"lo", flags | 1)
        )


@contextmanager
def _private_store(request):
    # Parent supervisor owns the encompassing process group, including timeouts.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    if port == 8073:
        raise RuntimeError("Refusing shared Store port")
    with tempfile.TemporaryDirectory(
        prefix="sllm-private-store-", dir=Path(request["result_path"]).parent
    ) as directory:
        config = dict(
            schema_version=1,
            artifact_path=request["artifact_path"],
            store=request["store"],
            port=port,
        )
        config_path = Path(directory) / "request.json"
        _write_json(config_path, config)
        log_path = (
            Path(request["result_path"]).parent / f"private-store-{port}.log"
        )
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sllm.vllm_prepare_worker",
                    "store",
                    "--request",
                    str(config_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_store(process, f"127.0.0.1:{port}")
                yield f"127.0.0.1:{port}"
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)


def _wait_store(process, address):
    import grpc

    from sllm_store.proto import storage_pb2, storage_pb2_grpc

    deadline = time.monotonic() + 30
    with grpc.insecure_channel(address) as channel:
        stub = storage_pb2_grpc.StorageStub(channel)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Private Store exited before readiness")
            try:
                config = stub.GetServerConfig(
                    storage_pb2.GetServerConfigRequest(), timeout=0.5
                )
                if (
                    config.mem_pool_size != STORE["mem_pool_size"]
                    or config.chunk_size != STORE["chunk_size"]
                ):
                    raise RuntimeError("Private Store budget mismatch")
                return
            except grpc.RpcError:
                time.sleep(0.05)
    raise TimeoutError("Private Store readiness timed out")


def _verify(request):
    isolation = check_network_namespace(request["parent_network_namespace"])
    _enable_loopback()
    from sllm_store.artifact import validate_candidate_artifact

    validate_candidate_artifact(
        request["artifact_path"],
        request["identity"],
        request["candidate_manifest"],
    )
    reference = json.loads(
        Path(request["reference_path"]).read_text(encoding="utf-8")
    )
    for field in ("schema_version", "identity", "engine_options", "validation"):
        if reference.get(field) != request[field]:
            raise ValueError(f"Reference identity mismatch: {field}")
    llm, sampling = _engine_api()
    with _private_store(request) as address:
        engine = llm(
            model=request["artifact_path"],
            tokenizer=request["artifact_path"],
            load_format="serverless_llm",
            model_loader_extra_config=dict(
                artifact_path=request["artifact_path"],
                identity=request["identity"],
                store_address=address,
                candidate_manifest=request["candidate_manifest"],
            ),
            **request["engine_options"],
        )
        try:
            validate_prompt_tokens(
                engine.get_tokenizer(),
                request["validation"]["prompts"],
                reference["outputs"],
            )
            prompts = [
                {"prompt_token_ids": output["prompt_token_ids"]}
                for output in reference["outputs"]
            ]
            actual = serialize_outputs(
                engine.generate(prompts, sampling(**SAMPLING), use_tqdm=False)
            )
            compare_outputs(
                reference["outputs"],
                actual,
                request["validation"]["logprob_atol"],
            )
            if engine.get_tokenizer().eos_token_id != reference["eos_token_id"]:
                raise ValueError("Tokenizer EOS mismatch")
        finally:
            _shutdown(engine)
    validate_candidate_artifact(
        request["artifact_path"],
        request["identity"],
        request["candidate_manifest"],
    )
    return dict(
        schema_version=1,
        passed=True,
        method="independent-offline-native-store-token-logprob-v1",
        tokens_verified=True,
        logprobs_verified=True,
        reference_sha256=_digest(request["reference_path"]),
        request_count=len(actual),
        isolation=isolation,
    )


def run(mode, request):
    validate_request(mode, request)
    if mode == "store":
        from sllm_store.server import serve

        return asyncio.run(
            serve(
                host="127.0.0.1",
                port=request["port"],
                storage_path=str(Path(request["artifact_path"]).parent),
                registration_required=False,
                **request["store"],
            )
        )
    return _export(request) if mode == "export" else _verify(request)


def parse_process_identity(stat):
    fields = stat.rsplit(")", 1)[1].split()
    return fields[0], fields[19]


def _process_alive(pid, start_time):
    try:
        state, actual = parse_process_identity(
            Path(f"/proc/{pid}/stat").read_text()
        )
        return state not in {"Z", "X"} and actual == start_time
    except (OSError, ValueError, IndexError):
        return False


def _watchdog(owner_pid, owner_start, runtime_pid, runtime_start, group):
    if (
        sys.platform != "linux"
        or os.getpgrp() != group
        or os.getsid(0) != group
    ):
        raise RuntimeError("Watchdog requires the owned private session")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("ready", flush=True)
    while _process_alive(owner_pid, owner_start) and _process_alive(
        runtime_pid, runtime_start
    ):
        time.sleep(0.1)
    os.killpg(group, signal.SIGTERM)
    time.sleep(2)
    os.killpg(group, signal.SIGKILL)


@contextmanager
def _owner_watchdog(request):
    if sys.platform != "linux":
        raise RuntimeError(
            "GPU preparation requires an owned Linux private session"
        )
    group = os.getpgrp()
    owner = request["owner_pid"]
    if (
        os.getsid(0) != group
        or os.getsid(owner) == group
        or not _process_alive(owner, request["owner_start_time"])
    ):
        raise RuntimeError("Preparation is not in an owned private session")
    _, runtime_start = parse_process_identity(
        Path("/proc/self/stat").read_text()
    )
    command = [
        sys.executable,
        "-m",
        "sllm.vllm_prepare_worker",
        "watchdog",
        str(owner),
        request["owner_start_time"],
        str(os.getpid()),
        runtime_start,
        str(group),
    ]
    child = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            if (
                not selector.select(timeout=5)
                or child.stdout.readline() != b"ready\n"
            ):
                raise RuntimeError("Preparation watchdog failed to start")
        yield
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        child.stdout.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "watchdog":
        if len(sys.argv) != 7:
            raise ValueError("Invalid watchdog invocation")
        return _watchdog(
            int(sys.argv[2]),
            sys.argv[3],
            int(sys.argv[4]),
            sys.argv[5],
            int(sys.argv[6]),
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "verify", "store"))
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    validate_request(args.mode, request)
    if args.mode == "verify":
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        ):
            os.environ[name] = "1"
        os.environ["VLLM_HOST_IP"] = "127.0.0.1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["SLLM_VERIFY_ARTIFACT"] = str(
            Path(request["artifact_path"]).resolve()
        )
        # Spawned Python engine workers get the same diagnostic audit policy.
        # The inherited kernel network namespace is the native-code boundary.
        with tempfile.TemporaryDirectory(
            prefix="sllm-offline-", dir=Path(request["result_path"]).parent
        ) as directory:
            (Path(directory) / "sitecustomize.py").write_text(
                "import sys\nfrom sllm.vllm_prepare_worker import offline_audit\nsys.addaudithook(offline_audit)\n",
                encoding="utf-8",
            )
            os.environ["PYTHONPATH"] = os.pathsep.join([directory, *sys.path])
            sys.addaudithook(offline_audit)
            with _owner_watchdog(request):
                result = run(args.mode, request)
    else:
        if args.mode == "export":
            with _owner_watchdog(request):
                result = run(args.mode, request)
        else:
            result = run(args.mode, request)
    if args.mode != "store":
        _write_json(request["result_path"], result)


if __name__ == "__main__":
    main()
