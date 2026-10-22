"""Fixed-version Store loader for bounded Qwen3 dense execution state.

API reference: vLLM v0.31.0 BaseModelLoader and weight_cache/ipc_loader.py.
CPU construction is deliberate: no duplicate full CUDA model allocation.
This candidate adapter still requires real engine/GPU qualification.
"""

import json
from pathlib import Path

import torch
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from sllm_store.artifact import validate_artifact, validate_candidate_artifact
from sllm_store.tensor_layout import bind_state, capture_state
from sllm_store.vllm_plugin import (
    ENGINE_VERSION,
    LAYOUT_ID,
    PLUGIN_ID,
    WRITER_ID,
    require_version,
    tensor_parallel_identity,
)


class ServerlessLLMModelLoader(BaseModelLoader):
    """Extra config requires artifact_path, exact identity and store_address.

    candidate_manifest is exclusively for the isolated pre-publication verifier;
    ordinary backend configuration must omit it. There is no directory fallback.
    """

    def __init__(self, load_config):
        require_version()
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config
        allowed = {
            "artifact_path",
            "identity",
            "store_address",
            "store_root",
            "candidate_manifest",
        }
        if not isinstance(extra, dict) or set(extra) - allowed:
            raise ValueError("Invalid Store loader extra config")
        path = extra.get("artifact_path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("artifact_path must be an absolute directory")
        self.path = Path(path).resolve()
        root = extra.get("store_root", str(self.path.parent))
        if not isinstance(root, str) or not Path(root).is_absolute():
            raise ValueError("store_root must be an absolute directory")
        self.store_root = Path(root).resolve()
        if not self.path.is_relative_to(self.store_root):
            raise ValueError("Artifact must be contained in store_root")
        self.identity = extra.get("identity")
        required = dict(
            engine="vllm",
            engine_version=ENGINE_VERSION,
            plugin=PLUGIN_ID,
            layout=LAYOUT_ID,
            writer=WRITER_ID,
            weight_phase="execution-ready",
        )
        if not isinstance(self.identity, dict) or any(
            self.identity.get(k) != v for k, v in required.items()
        ):
            raise ValueError("Unsupported Store artifact identity")
        self.identity = json.loads(json.dumps(self.identity, allow_nan=False))
        address = extra.get("store_address")
        if not isinstance(address, str) or not address.startswith("127.0.0.1:"):
            raise ValueError("store_address must specify a local private Store")
        try:
            port = int(address.split(":")[1])
        except ValueError as error:
            raise ValueError("Invalid Store port") from error
        if not 0 < port < 65536 or address != f"127.0.0.1:{port}":
            raise ValueError("Invalid Store port")
        self.store_address = address
        self.candidate = extra.get("candidate_manifest")
        if "candidate_manifest" in extra and not isinstance(
            self.candidate, dict
        ):
            raise ValueError("Invalid candidate_manifest")

    def _validate_artifact(self, model_config):
        model_path = Path(model_config.model)
        if (
            not model_path.is_absolute()
            or model_path.resolve() != self.path.resolve()
        ):
            raise ValueError("Model model path must match Store artifact_path")
        if str(model_config.dtype) != self.identity.get("dtype"):
            raise ValueError("Artifact dtype mismatch")
        if self.candidate is None:
            validate_artifact(self.path, self.identity)
        else:
            validate_candidate_artifact(
                self.path, self.identity, self.candidate
            )

    def _validate(self, model_config):
        self._validate_artifact(model_config)
        rank, world = tensor_parallel_identity()
        if world != self.identity.get("world_size"):
            raise ValueError("Artifact TP world size mismatch")
        directory = self.path / f"rank_{rank}"
        layout = json.loads(
            (directory / "tensor-layout.json").read_text(encoding="utf-8")
        )
        expected = dict(
            rank=rank,
            world_size=world,
            plugin=PLUGIN_ID,
            layout=LAYOUT_ID,
            writer=WRITER_ID,
            weight_phase="execution-ready",
        )
        if any(layout.get(k) != v for k, v in expected.items()):
            raise ValueError("Rank tensor layout identity mismatch")
        return directory, layout

    def download_model(self, model_config):
        # Preparation runs before distributed Worker initialization.
        self._validate_artifact(model_config)

    def load_weights(self, model, model_config):
        raise NotImplementedError("In-place Store reload is not qualified")

    def _load_storage(self, directory, device):
        if device.type != "cuda":
            raise ValueError("Native Store restore requires CUDA")
        from sllm_store.torch import load_dict

        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return load_dict(
            directory.resolve().relative_to(self.store_root).as_posix(),
            {"": index},
            storage_path=str(self.store_root),
            server_address=self.store_address,
        )

    def load_model(self, vllm_config, model_config, prefix=""):
        require_version()
        architecture = model_config.hf_config.architectures
        if architecture != ["Qwen3ForCausalLM"]:
            raise ValueError("CPU skeleton is restricted to dense Qwen3")
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.tensor_parallel_size != self.identity.get("world_size")
            or getattr(parallel, "enable_expert_parallel", False)
            or getattr(vllm_config, "quant_config", None) is not None
            or model_config.quantization is not None
            or getattr(vllm_config, "lora_config", None) is not None
            or getattr(vllm_config, "speculative_config", None) is not None
            or model_config.dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("Unsupported Store engine configuration")
        directory, layout = self._validate(model_config)
        load_device = self.load_config.device
        if load_device is None:
            load_device = vllm_config.device_config.device
        device = torch.device(load_device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        from vllm.utils.torch_utils import set_default_torch_dtype
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )
        from vllm.model_executor.utils import weights_already_processed

        with set_default_torch_dtype(model_config.dtype):
            with torch.device("cpu"):
                model = self.create_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            storages = self._load_storage(directory, device)
            bind_state(model, storages, layout, device=device)
            # v0.31.0 official IPC loader uses this context to rebuild runtime
            # Python state while preserving already processed tensor weights.
            with weights_already_processed():
                process_weights_after_loading(model, model_config, device)
            _, after = capture_state(model)
            if (
                after["tensors"] != layout["tensors"]
                or after["storages"] != layout["storages"]
            ):
                raise ValueError(
                    "Post-load hook changed execution tensor state"
                )
        return model.eval()
