"""Lazy vLLM 0.31.0 general plugin and string-RPC Worker extension.

Importing this module imports neither vLLM, Torch nor the native Store module.
"""

import json
from importlib import metadata
from pathlib import Path

ENGINE_VERSION = "0.31.0"
PLUGIN_ID = "sllm-store-vllm-0.31.0-v1"
LAYOUT_ID = "sllm-storage-views-v1"
WRITER_ID = "sllm-native-store-v1"
_registered = False


def require_version():
    if metadata.version("vllm") != ENGINE_VERSION:
        raise RuntimeError("ServerlessLLM Store requires vLLM 0.31.0 exactly")


def register():
    """Register once per process through vLLM's supported loader registry."""
    global _registered
    require_version()
    if _registered:
        return
    from vllm.model_executor.model_loader import register_model_loader
    from sllm_store.vllm_loader import ServerlessLLMModelLoader

    register_model_loader("serverless_llm")(ServerlessLLMModelLoader)
    _registered = True


def tensor_parallel_identity():
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    rank = get_tensor_model_parallel_rank()
    world = get_tensor_model_parallel_world_size()
    if type(rank) is not int or type(world) is not int or not 0 <= rank < world:
        raise ValueError("Invalid tensor parallel rank/world size")
    return rank, world


class ServerlessLLMWorkerExtension:
    """Configure worker_extension_cls; export via named collective RPC."""

    def export_serverless_model(self, path):
        require_version()
        from sllm_store.tensor_layout import capture_state
        from sllm_store.torch import save_dict

        root = Path(path)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValueError(
                "Export path must be an existing absolute directory"
            )
        rank, world = tensor_parallel_identity()
        directory = root / f"rank_{rank}"
        directory.mkdir()  # Never overwrite a rank from another attempt.
        storages, layout = capture_state(self.model_runner.model)
        layout.update(
            rank=rank,
            world_size=world,
            plugin=PLUGIN_ID,
            layout=LAYOUT_ID,
            writer=WRITER_ID,
            weight_phase="execution-ready",
        )
        save_dict(storages, str(directory))
        with (directory / "tensor-layout.json").open(
            "x", encoding="utf-8"
        ) as f:
            json.dump(layout, f, sort_keys=True, allow_nan=False)
        return {
            "rank": rank,
            "world_size": world,
            "writer": WRITER_ID,
            "plugin": PLUGIN_ID,
            "layout": LAYOUT_ID,
            "weight_phase": "execution-ready",
        }
