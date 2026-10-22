# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import importlib
import logging
import os
from typing import Optional

import ray
from torch import nn
from transformers import AutoTokenizer

logger = logging.getLogger("ray")


# def get_directory_size(directory):
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for filename in filenames:
#             file_path = os.path.join(dirpath, filename)
#             if not os.path.islink(file_path):
#                 total_size += os.path.getsize(file_path)
#     return total_size


@ray.remote(num_cpus=1)
def download_transformers_model(
    model_name: str,
    pretrained_model_name_or_path: str,
    torch_dtype: str,
    hf_model_class: str,
) -> bool:
    storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    model_path = os.path.join(storage_path, "transformers", model_name)
    tokenizer_path = os.path.join(
        storage_path, "transformers", model_name, "tokenizer"
    )

    if os.path.exists(model_path):
        logger.info(f"{model_path} already exists")
        return True

    import torch

    torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        raise ValueError(f"Invalid torch_dtype: {torch_dtype}")

    logger.info(f"Downloading {model_path}")

    module = importlib.import_module("transformers")
    hf_model_cls = getattr(module, hf_model_class)
    model = hf_model_cls.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    from sllm_store.transformers import save_model

    logger.info(f"Saving {model_path}")
    try:
        save_model(model, model_path)
        tokenizer.save_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to save {model_path}: {e}")
        # shutil.rmtree(model_path)  # TODO: deal with error in save_model
        raise RuntimeError(
            f"Failed to save {model_name} for transformer backend: {e}"
        )

    # model_size = get_directory_size(model_path)
    # logger.info(f"{model_name} (size: {model_size}) downloaded")

    return True


@ray.remote(num_cpus=1)
def download_lora_adapter(
    base_model_name: str,
    adapter_name: str,
    adapter_name_or_path: str,
    hf_model_class: str,
    torch_dtype: str,
) -> bool:
    storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    adapter_path = os.path.join(
        storage_path, "transformers", adapter_name_or_path
    )

    if os.path.exists(adapter_path):
        logger.info(f"{adapter_path} already exists")
        return True

    import torch

    torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        raise ValueError(f"Invalid torch_dtype: {torch_dtype}")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        os.path.join(storage_path, "transformers", base_model_name),
        trust_remote_code=True,
    )
    config.torch_dtype = torch_dtype
    module = importlib.import_module("transformers")
    hf_model_cls = getattr(module, hf_model_class)
    base_model = hf_model_cls.from_config(
        config,
        trust_remote_code=True,
    ).to(config.torch_dtype)

    logger.info(f"Downloading {adapter_path}")
    from peft import PeftModel

    try:
        model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise RuntimeError(f"LoRA adapter load failed: {e}")

    from sllm_store.transformers import save_lora

    logger.info(f"Saving {adapter_path}")
    try:
        save_lora(model, adapter_path)
    except Exception as e:
        logger.error(f"Failed to save {adapter_path}: {e}")
        # shutil.rmtree(model_path)  # TODO: deal with error in save_model
        raise RuntimeError(
            f"Failed to save {adapter_name} for transformer backend: {e}"
        )

    return True


class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        pretrained_model_name_or_path: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
        *,
        revision: Optional[str] = None,
        source_snapshot=None,
    ):
        if pattern is not None or max_size is not None:
            raise ValueError(
                "pattern and max_size are unsupported for verified exports"
            )

        from sllm.vllm_preparation import prepare_vllm_model

        return prepare_vllm_model(
            model_name,
            pretrained_model_name_or_path or model_name,
            torch_dtype,
            tensor_parallel_size,
            storage_path=os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            ),
            revision=revision,
            source_snapshot=source_snapshot,
        )
