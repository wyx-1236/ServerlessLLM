"""CPU configuration contract; no engine or torch imports are permitted."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def resolve():
    module = importlib.import_module("sllm.backends.vllm_config")
    fields = json.loads(
        (
            Path(__file__).parent / "fixtures/vllm-0.31.0-engine-fields.json"
        ).read_text()
    )

    def project(config=None, version="0.31.0"):
        return module.resolve_vllm_config(
            "Qwen/model",
            config or {},
            engine_version=version,
            engine_fields=fields,
            storage_path="/models",
        )

    return project


def test_defaults_keep_store_and_original_cache_policy(resolve):
    result = resolve()
    assert Path(result["model"]) == Path("/models") / "vllm" / "Qwen/model"
    assert result["load_format"] == "serverless_llm"
    assert result["enforce_eager"] is False
    assert result["enable_prefix_caching"] is True
    assert result["distributed_executor_backend"] == "mp"
    assert "task" not in result


def test_aliases_do_not_mutate_input_or_leak_host_fields(resolve):
    config = {
        "torch_dtype": "bfloat16",
        "pretrained_model_name_or_path": "Qwen/source",
        "device_map": "auto",
        "hf_model_class": "AutoModelForCausalLM",
        "trace_debug": True,
        "tensor_parallel_size": 2,
    }
    before = config.copy()
    result = resolve(config)
    assert config == before
    assert result["dtype"] == "bfloat16"
    assert result["tensor_parallel_size"] == 2
    assert not (set(config) - {"tensor_parallel_size"}) & result.keys()


@pytest.mark.parametrize(
    "task,expected",
    [
        ("auto", {}),
        ("generate", {"runner": "generate"}),
        ("embed", {"runner": "pooling", "convert": "embed"}),
        ("embedding", {"runner": "pooling", "convert": "embed"}),
        ("classify", {"runner": "pooling", "convert": "classify"}),
    ],
)
def test_legacy_task_mapping(resolve, task, expected):
    result = resolve({"task": task})
    assert "task" not in result
    for key, value in expected.items():
        assert result[key] == value


@pytest.mark.parametrize("version", ["0.11.2", "0.31.1", "0.31.0+unknown"])
def test_unknown_engine_versions_are_not_guessed(resolve, version):
    with pytest.raises(ValueError, match="version"):
        resolve(version=version)


@pytest.mark.parametrize(
    "config,reason",
    [
        ({"tensor_paralel_size": 2}, "tensor_paralel_size"),
        ({"torch_dtype": "float16", "dtype": "bfloat16"}, "dtype"),
        ({"task": "generate", "runner": "pooling"}, "runner"),
        ({"task": "classify", "convert": "embed"}, "convert"),
        ({"task": "reward"}, "task"),
        ({"device_map": "cpu"}, "device_map"),
        ({"load_format": "safetensors"}, "pretrained_model_name_or_path"),
    ],
)
def test_unsupported_or_ambiguous_configuration_fails(resolve, config, reason):
    with pytest.raises(ValueError, match=reason):
        resolve(config)


def test_explicit_reference_loader_is_retained(resolve):
    result = resolve(
        {
            "load_format": "safetensors",
            "pretrained_model_name_or_path": "/snapshots/reference",
            "enforce_eager": True,
            "enable_prefix_caching": False,
        }
    )
    assert result["model"] == "/snapshots/reference"
    assert result["load_format"] == "safetensors"
    assert result["enforce_eager"] is True
    assert result["enable_prefix_caching"] is False
