"""Opt-in, read-only Worker RPC for the pinned D2 numerical qualification.

Ordinary serving does not install this extension. Importing it is CPU-only;
the RPC may only inspect an already initialized, separately approved Worker.
"""

import os
import sys
from pathlib import Path


class NumericalQualificationWorker:
    def sllm_numerical_runtime(self):
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            raise RuntimeError(
                "Numerical inspection requires an initialized Worker"
            )
        vllm = sys.modules["vllm"]
        invariant = sys.modules.get(
            "vllm.model_executor.determinism.batch_invariant"
        )
        config = self.vllm_config
        runner = self.model_runner
        props = torch.cuda.get_device_properties(self.device)
        layers = []
        for groups in runner.attn_groups:
            for group in groups:
                for name in group.layer_names:
                    layer = config.compilation_config.static_forward_context[
                        name
                    ]
                    layers.append(
                        {
                            "name": name,
                            "backend": group.backend.get_name(),
                            "batch_invariant_enabled": getattr(
                                layer.impl, "batch_invariant_enabled", None
                            ),
                        }
                    )
        return {
            "schema_version": 1,
            "engine_version": vllm.__version__,
            "pid": os.getpid(),
            "cgroup": Path("/proc/self/cgroup").read_text(),
            "gpu_uuid": str(props.uuid),
            "device_name": props.name,
            "compute_capability": [props.major, props.minor],
            "environment": os.environ.get("VLLM_BATCH_INVARIANT"),
            "batch_invariant_initialized": getattr(
                invariant, "_batch_invariant_MODE", False
            ),
            "use_v2_model_runner": config.use_v2_model_runner,
            "runner_class": type(runner).__module__
            + "."
            + type(runner).__name__,
            "dtype": str(config.model_config.dtype),
            "enforce_eager": config.model_config.enforce_eager,
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "attention_layers": layers,
        }
