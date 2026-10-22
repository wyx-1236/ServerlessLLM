"""Project SLLM configuration onto the fixed vLLM engine API without imports.

The caller supplies fields from the installed AsyncEngineArgs dataclass.
Configuration projection does not construct an engine or initialize CUDA.
"""

import os
from collections.abc import Collection, Mapping
from typing import Any


def resolve_vllm_config(
    model: str,
    config: Mapping[str, Any],
    *,
    engine_version: str,
    engine_fields: Collection[str],
    storage_path: str,
) -> dict[str, Any]:
    if engine_version != "0.31.0":
        raise ValueError(f"Unsupported vLLM version: {engine_version}")

    host_fields = {
        "pretrained_model_name_or_path",
        "torch_dtype",
        "task",
        "trace_debug",
        "device_map",
        "hf_model_class",
        "source_snapshot",
    }
    unknown = set(config) - set(engine_fields) - host_fields
    if unknown:
        raise ValueError(f"Unsupported backend fields: {sorted(unknown)}")
    if config.get("device_map", "auto") != "auto":
        raise ValueError("device_map is not a vLLM placement control")

    result = {
        key: value
        for key, value in config.items()
        if key in engine_fields and key not in host_fields
    }
    if config.get("torch_dtype") is not None:
        dtype = config["torch_dtype"]
        if result.get("dtype", dtype) != dtype:
            raise ValueError("Conflicting torch_dtype and dtype")
        result["dtype"] = dtype

    task = config.get("task", "auto")
    tasks = {
        "auto": {},
        "generate": {"runner": "generate"},
        "embed": {"runner": "pooling", "convert": "embed"},
        "embedding": {"runner": "pooling", "convert": "embed"},
        "classify": {"runner": "pooling", "convert": "classify"},
    }
    if task not in tasks:
        raise ValueError(f"Unsupported legacy task: {task}")
    for key, value in tasks[task].items():
        if key in result and result[key] != value:
            raise ValueError(f"Conflicting task and {key}")
        result[key] = value

    if config.get("load_format") is not None:
        source = config.get("pretrained_model_name_or_path")
        if not source:
            raise ValueError(
                "Explicit load_format requires pretrained_model_name_or_path"
            )
        result["model"] = source
    else:
        result["model"] = os.path.join(storage_path, "vllm", model)
        result["load_format"] = "serverless_llm"
    result.setdefault("enforce_eager", False)
    result.setdefault("enable_prefix_caching", True)
    result.setdefault("distributed_executor_backend", "mp")
    invalid = set(result) - set(engine_fields)
    if invalid:
        raise ValueError(f"Missing engine API fields: {sorted(invalid)}")
    return result
