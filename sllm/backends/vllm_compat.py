"""The only vLLM 0.31.0 construction and lifecycle compatibility boundary."""

import inspect
import json
import os
from dataclasses import fields
from pathlib import Path

from sllm.backends.vllm_config import resolve_vllm_config


def engine_args(model, config):
    import vllm

    projected = resolve_vllm_config(
        model,
        config,
        engine_version=vllm.__version__,
        engine_fields={field.name for field in fields(vllm.AsyncEngineArgs)},
        storage_path=os.getenv("STORAGE_PATH", os.path.expanduser("~/models")),
    )
    if projected["load_format"] == "serverless_llm":
        from sllm_store.artifact import validate_artifact

        path = Path(projected["model"]).resolve()
        manifest = json.loads((path / "artifact-manifest.json").read_text())
        identity = manifest["identity"]
        validate_artifact(path, identity)
        if identity.get("engine_version") != "0.31.0":
            raise ValueError("Artifact engine version mismatch")
        if projected.get("tensor_parallel_size", 1) != identity.get(
            "world_size"
        ):
            raise ValueError("Artifact TP world size mismatch")
        dtype = projected.get("dtype", "auto")
        aliases = {"half": "float16", "float": "float32", "bf16": "bfloat16"}
        dtype = aliases.get(
            str(dtype).removeprefix("torch."), str(dtype).removeprefix("torch.")
        )
        if dtype != "auto" and f"torch.{dtype}" != identity.get("dtype"):
            raise ValueError("Artifact dtype mismatch")
        projected["dtype"] = identity["dtype"].removeprefix("torch.")
        root = Path(
            os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
        ).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact must be contained in store_root")
        extra = dict(projected.get("model_loader_extra_config") or {})
        if set(extra) - {
            "artifact_path",
            "identity",
            "store_address",
            "store_root",
        }:
            raise ValueError("Invalid Store loader extra config")
        if (
            extra.get("artifact_path", str(path)) != str(path)
            or extra.get("identity", identity) != identity
        ):
            raise ValueError("Conflicting Store artifact identity")
        if extra.get("store_root", str(root)) != str(root):
            raise ValueError("Conflicting Store store_root")
        address = extra.get("store_address")
        if not isinstance(address, str) or not address.startswith("127.0.0.1:"):
            raise ValueError("store_address must specify a local private Store")
        try:
            port = int(address.split(":")[1])
        except ValueError as error:
            raise ValueError("Invalid Store port") from error
        if not 0 < port < 65536 or address != f"127.0.0.1:{port}":
            raise ValueError("Invalid Store port")
        projected["model"] = str(path)
        projected["model_loader_extra_config"] = dict(
            artifact_path=str(path),
            identity=identity,
            store_address=address,
            store_root=str(root),
        )
    return vllm.AsyncEngineArgs(**projected)


async def abort(engine, request_id):
    # v0.31 AsyncLLM.abort is async; the public request id is positional.
    await engine.abort(request_id)


async def close(engine):
    if engine is None:
        return
    result = engine.shutdown()
    if inspect.isawaitable(result):
        await result


def create(args):
    from vllm import AsyncLLMEngine

    class OwnedEngine(AsyncLLMEngine):
        def __init__(self, *constructor_args, **constructor_kwargs):
            try:
                super().__init__(*constructor_args, **constructor_kwargs)
            except BaseException:
                # v0.31 shutdown explicitly tolerates partial construction via
                # getattr and closes EngineCore, not just the frontend object.
                self.shutdown()
                raise

    return OwnedEngine.from_engine_args(args)
