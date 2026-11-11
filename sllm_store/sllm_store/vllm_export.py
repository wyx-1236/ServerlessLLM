"""Compatibility adapter for the public Store vLLM export entry points."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def export_vllm_model(
    model_name,
    torch_dtype,
    tensor_parallel_size=1,
    *,
    storage_path,
    local_model_path=None,
    pattern=None,
    max_size=None,
):
    """Export into storage_path/model_name through verified preparation.

    Requires the Store vLLM extra and the pinned Linux preparation environment.
    Launching GPU workers needs the same approval as preparation.
    """
    try:
        from sllm_store.vllm_preparation import prepare_vllm_model

        if pattern is not None or max_size is not None:
            # The original patch accepted these arguments but ignored them.
            logger.warning(
                "pattern and max_size are ignored, as in the original exporter"
            )
        source = model_name
        revision = None
        if local_model_path is not None:
            local = Path(local_model_path).expanduser()
            if not local.is_dir():
                raise ValueError(f"Invalid local model directory: {local}")
            source = str(local)
        else:
            from huggingface_hub import HfApi

            # Preserve the model-name-only command while freezing the resolved
            # default branch before the pipeline downloads or validates weights.
            revision = HfApi().model_info(model_name).sha
            if not isinstance(revision, str) or not re.fullmatch(
                r"[0-9a-f]{40}", revision
            ):
                raise ValueError(
                    "Hub did not return an immutable model revision"
                )
        prepare_vllm_model(
            model_name,
            source,
            torch_dtype,
            tensor_parallel_size,
            storage_path=storage_path,
            revision=revision,
            flat_layout=True,
        )
    except Exception as error:
        # The preparation transaction owns cleanup. Never remove an existing
        # target, which may belong to a prior run or another concurrent caller.
        raise RuntimeError(
            f"Failed to save {model_name} for vllm backend: {error}"
        ) from error
