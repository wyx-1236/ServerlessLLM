"""Pinned Transformers cache/streamer compatibility using a tiny offline model.

Run in the separate Transformers CPU profile. Store loading and CUDA placement
are replaced; model generation, caching, tokenization and streaming are real.
These cases cover bounded continuation budgets/EOS, not full migration or
native Store qualification.
"""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast
from transformers.cache_utils import Cache

ROOT = Path(__file__).parents[2]


@pytest.fixture
def backend(monkeypatch):
    assert torch.__version__ == "2.13.0+cpu"
    assert torch.version.cuda is None
    assert not torch.cuda.is_initialized()
    assert transformers.__version__ == "5.17.0"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    store = ModuleType("sllm_store.transformers")

    def unexpected_store(*args, **kwargs):
        pytest.fail("Native Store is outside this CPU profile")

    store.load_model = store.load_lora = store.save_lora = unexpected_store
    monkeypatch.setitem(sys.modules, "sllm_store.transformers", store)
    spec = importlib.util.spec_from_file_location(
        "transformers_cache_probe",
        ROOT / "sllm/backends/transformers_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[BOS]": 2,
                "[EOS]": 3,
                "hello": 4,
                "world": 5,
                **{f"token{index}": index for index in range(6, 32)},
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{{ messages[0]['content'] }}",
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = OPTForCausalLM(
            OPTConfig(
                vocab_size=32,
                hidden_size=16,
                word_embed_proj_dim=16,
                ffn_dim=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=64,
                dropout=0.0,
                attention_dropout=0.0,
                pad_token_id=1,
                bos_token_id=2,
                eos_token_id=None,
            )
        ).eval()
    instance = module.TransformersBackend("tiny-opt", {})
    instance.model = model
    instance.tokenizer = tokenizer
    instance.status = module.BackendStatus.RUNNING
    # Replace only the original hard-coded CUDA transfer; tokenization is real.
    monkeypatch.setattr(
        instance,
        "_tokenize",
        lambda prompt: tokenizer(prompt, return_tensors="pt"),
    )
    try:
        yield instance
    finally:
        torch.set_num_threads(threads)
        assert not torch.cuda.is_initialized()


def request(max_tokens=3):
    return {
        "model": "tiny-opt",
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
    }


def reference(backend, count=6):
    with torch.no_grad():
        return backend.model.generate(
            **backend._tokenize("hello world"),
            max_new_tokens=count,
        )


def test_normal_generation_preserves_response_and_streamer(backend):
    data = request()
    original = deepcopy(data)
    observed = []
    put = backend.inf_status.put

    def record(value):
        put(value)
        observed.append(backend.get_current_tokens())

    backend.inf_status.put = record
    result = backend.generate(data)
    expected = reference(backend, 3).tolist()[0]
    assert data == original
    assert result["model"] == "tiny-opt"
    assert result["object"] == "chat.completion"
    assert result["id"].startswith("chatcmpl-")
    assert result["choices"][0]["message"][
        "content"
    ] == backend.tokenizer.decode(expected[2:], skip_special_tokens=True)
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert observed == [[expected[:length]] for length in range(2, 6)]
    assert backend.get_current_tokens() == []


def test_real_streamer_preserves_preemption_envelope(backend):
    put = backend.inf_status.put

    def preempt(value):
        tokens = backend.inf_status.get()
        if tokens and len(tokens[0]) == 3:
            backend.inf_status.status = type(backend.status).DELETING
        put(value)

    backend.inf_status.put = preempt
    result = backend.generate(request())
    assert result == {
        "preempted": "True",
        "current_output": reference(backend, 2).tolist(),
        "completed_tokens": 2,
    }
    assert backend.get_current_tokens() == []


@pytest.mark.parametrize("lengths", [(3,), (3, 4, 5)])
def test_cache_replay_accepts_new_and_growing_snapshots(backend, lengths):
    expected = reference(backend).tolist()[0]
    cache = None
    for length in lengths:
        snapshot = [expected[:length]]
        original = deepcopy(snapshot)
        assert backend.resume_kv_cache(snapshot) is None
        assert snapshot == original
        assert isinstance(backend.past_key_values, Cache)
        if cache is not None:
            assert backend.past_key_values is cache
        cache = backend.past_key_values
        assert cache.get_seq_length() == length
        assert len(cache.layers) == backend.model.config.num_hidden_layers
        for layer in cache.layers:
            assert layer.keys.shape == (1, 2, length, 8)
            assert layer.values.shape == (1, 2, length, 8)
        assert backend.current_tokens.tolist() == [expected[: length + 1]]


def test_cached_continuation_accepts_longer_final_source_snapshot(backend):
    expected = reference(backend).tolist()[0]
    backend.resume_kv_cache([expected[:3]])
    backend.resume_kv_cache([expected[:4]])
    # Final source has advanced past the replay snapshot, allowing prefix reuse.
    final_source = [expected[:5]]
    data = request(max_tokens=3)
    original = deepcopy(data)
    result = backend.resume_generate(data, final_source)
    assert data == original
    assert final_source == [expected[:5]]
    assert result["choices"][0]["message"][
        "content"
    ] == backend.tokenizer.decode(expected[2:], skip_special_tokens=True)
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 6,
        "total_tokens": 8,
    }
    assert backend.get_current_tokens() == []


@pytest.mark.parametrize("warmup_length", [4, 6])
def test_continuation_uses_source_snapshot_not_longer_warmup(
    backend, warmup_length
):
    expected = reference(backend, 6).tolist()[0]
    backend.resume_kv_cache([expected[:warmup_length]])
    source = [expected[:4]]
    original = deepcopy(source)
    result = backend.resume_generate(request(max_tokens=2), source)
    assert source == original
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 4,
        "total_tokens": 6,
    }
    assert result["choices"][0]["message"][
        "content"
    ] == backend.tokenizer.decode(expected[2:6], skip_special_tokens=True)
    assert result["choices"][0]["finish_reason"] == "length"


def test_continuation_discards_a_divergent_cache_prefix(backend, monkeypatch):
    expected = reference(backend, 6).tolist()[0]
    backend.resume_kv_cache([[4, 5, 8, 9]])
    generate = Mock(wraps=backend.model.generate)
    monkeypatch.setattr(backend.model, "generate", generate)
    result = backend.resume_generate(request(2), [expected[:5]])
    assert generate.call_args.kwargs["past_key_values"] is None
    assert result["choices"][0]["message"][
        "content"
    ] == backend.tokenizer.decode(expected[2:7], skip_special_tokens=True)


def test_continuation_without_replay_uses_final_source_snapshot(backend):
    expected = reference(backend, 4).tolist()[0]
    result = backend.resume_generate(request(2), [expected[:4]])
    assert result["usage"]["completion_tokens"] == 4
    assert result["choices"][0]["message"][
        "content"
    ] == backend.tokenizer.decode(expected[2:], skip_special_tokens=True)


def test_zero_remaining_budget_does_not_generate(backend, monkeypatch):
    expected = reference(backend, 2).tolist()[0]
    backend.resume_kv_cache([expected])
    generate = Mock(side_effect=AssertionError("No remaining budget"))
    monkeypatch.setattr(backend.model, "generate", generate)
    result = backend.resume_generate(request(0), [expected])
    generate.assert_not_called()
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    assert result["choices"][0]["finish_reason"] == "length"
    assert backend.get_current_tokens() == []


def test_missing_budget_keeps_original_default_total(backend):
    expected = reference(backend, 10).tolist()[0]
    backend.resume_kv_cache([expected[:3]])
    data = request()
    data.pop("max_tokens")
    result = backend.resume_generate(data, [expected[:5]])
    assert "max_tokens" not in data
    assert result["usage"]["completion_tokens"] == 10
    assert result["choices"][0]["message"][
        "content"
    ] == backend.tokenizer.decode(expected[2:], skip_special_tokens=True)


def test_eos_on_last_budget_token_reports_stop(backend):
    source = reference(backend, 2).tolist()
    backend.resume_kv_cache([source[0][:3]])
    backend.model.generation_config.eos_token_id = 3
    backend.model.generation_config.forced_eos_token_id = 3
    result = backend.resume_generate(request(1), source)
    assert result["usage"]["completion_tokens"] == 3
    assert result["choices"][0]["finish_reason"] == "stop"


def test_terminal_source_eos_is_not_resumed(backend, monkeypatch):
    backend.model.generation_config.eos_token_id = 3
    generate = Mock(side_effect=AssertionError("Source already ended at EOS"))
    monkeypatch.setattr(backend.model, "generate", generate)
    result = backend.resume_generate(request(2), [[4, 5, 3]])
    generate.assert_not_called()
    assert result["usage"]["completion_tokens"] == 1
    assert result["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("second_snapshot", [[4, 5, 5], [4, 5], [4, 5, 8, 9]])
def test_replay_rebuilds_repeated_shorter_or_divergent_snapshots(
    backend, second_snapshot
):
    backend.resume_kv_cache([[4, 5, 5]])
    old_cache = backend.past_key_values
    with torch.no_grad():
        expected = backend.model.generate(
            torch.tensor([second_snapshot]), max_new_tokens=1
        )
    backend.resume_kv_cache([second_snapshot])
    assert backend.past_key_values is not old_cache
    assert backend.past_key_values.get_seq_length() == len(second_snapshot)
    assert torch.equal(backend.current_tokens, expected)


def test_failed_replay_drops_mutable_cache_before_retry(backend, monkeypatch):
    expected = reference(backend, 4).tolist()[0]
    backend.resume_kv_cache([expected[:3]])
    original_generate = backend.model.generate

    def fail_after_cache_mutation(*args, **kwargs):
        kwargs["past_key_values"].crop(1)
        raise RuntimeError("replay fault")

    monkeypatch.setattr(backend.model, "generate", fail_after_cache_mutation)
    with pytest.raises(RuntimeError, match="replay fault"):
        backend.resume_kv_cache([expected[:4]])
    assert backend.past_key_values is None
    monkeypatch.setattr(backend.model, "generate", original_generate)
    backend.resume_kv_cache([expected[:4]])
    assert backend.current_tokens.tolist() == [expected[:5]]


def test_interrupted_continuation_clears_streamer_and_consumed_cache(backend):
    expected = reference(backend, 4).tolist()[0]
    backend.resume_kv_cache([expected[:3]])
    backend.inf_status.status = type(backend.status).DELETING
    with pytest.raises(Exception, match="Backend is shutting down"):
        backend.resume_generate(request(2), [expected[:4]])
    assert backend.get_current_tokens() == []
    assert backend.past_key_values is None
