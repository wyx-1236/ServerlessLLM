"""Exercise EOS preparation with the pinned real tokenizer API, without weights."""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_continuation_flow import flow_factory as flow_factory
from test_stopping_qualification import (
    configure_stopping_flow,
)
from test_stopping_qualification import stopping as stopping


@pytest.mark.parametrize("mode", ["normal", "probe-exhausted"])
def test_real_chat_template_preserves_token_ids_and_evidence(
    stopping, request, tmp_path, mode
):
    if os.environ.get("SLLM_STOPPING_TOKENIZER_TEST_CHILD") != "1":
        if importlib.util.find_spec("transformers") is None:
            pytest.skip("Real tokenizer requires the Transformers CPU profile")
        had_torch = "torch" in sys.modules
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{Path(__file__).resolve()}::{request.node.name}",
                "-q",
                "--tb=short",
            ],
            env={
                **os.environ,
                "SLLM_STOPPING_TOKENIZER_TEST_CHILD": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert ("torch" in sys.modules) == had_torch
        return
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "user": 1, "[EOS]": 2}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    real = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        chat_template="{% for m in messages %}{{ m['role'] }} {{ m['content'] }}{% endfor %}",
    )
    messages = [{"role": "user", "content": stopping.evidence.EOS_PROMPTS[0]}]
    default = real.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    assert isinstance(default, transformers.BatchEncoding)
    expected = default["input_ids"]
    flow = request.getfixturevalue("flow_factory")()
    config, tokenizer = configure_stopping_flow(flow, tmp_path, mode)
    tokenizer.apply_chat_template = real.apply_chat_template
    result = {"checks": {}, "cases": [], "scenarios": []}
    if mode == "normal":
        asyncio.run(
            flow.run("eos", result, config, suite=stopping, tokenizer=tokenizer)
        )
    else:
        with pytest.raises(ValueError, match="No qualified EOS"):
            asyncio.run(
                flow.run(
                    "eos", result, config, suite=stopping, tokenizer=tokenizer
                )
            )
        assert not flow.allocation_started.is_set()
    saved = json.loads((flow.output / stopping.RESULT_NAME).read_text())
    scenario = saved["scenarios"][0]
    probe = scenario["stopping"]["probes"][0]
    assert probe["prompt_token_ids"] == expected
    assert probe["request"]["input_tokens"] == expected
    assert probe["history"]["prompt_token_ids"] == expected
    if mode == "normal":
        assert scenario["stopping_validation"]["qualified_request_ids"]
    else:
        assert len(scenario["stopping"]["probes"]) == 3
        assert "No qualified EOS" in scenario["preparation_error"]
