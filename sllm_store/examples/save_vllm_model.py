import argparse
import os
from typing import Optional


class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        storage_path: str = os.path.expanduser("~/models"),
        local_model_path: Optional[str] = None,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        from sllm_store.vllm_export import export_vllm_model

        # Preserve this example's historical environment-first path precedence.
        storage_path = os.getenv(
            "STORAGE_PATH", storage_path
        ) or os.path.expanduser("~/models")
        export_vllm_model(
            model_name,
            torch_dtype,
            tensor_parallel_size,
            storage_path=storage_path,
            local_model_path=local_model_path,
            pattern=pattern,
            max_size=max_size,
        )


parser = argparse.ArgumentParser(
    description="Save a model from HuggingFace model hub."
)
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub.",
)
parser.add_argument(
    "--local-model-path",
    type=str,
    required=False,
    help="Local path to the model snapshot.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default=os.path.expanduser("~/models"),
    help="Local path to save the model.",
)
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Tensor parallel size.",
)

args = parser.parse_args()

model_name = args.model_name
local_model_path = args.local_model_path
storage_path = args.storage_path
tensor_parallel_size = args.tensor_parallel_size

downloader = VllmModelDownloader()
downloader.download_vllm_model(
    model_name,
    "float16",
    tensor_parallel_size=tensor_parallel_size,
    storage_path=storage_path,
    local_model_path=local_model_path,
)
