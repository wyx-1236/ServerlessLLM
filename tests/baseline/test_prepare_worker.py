import importlib
import json
from types import SimpleNamespace

import pytest


def worker():
    return importlib.import_module("sllm.vllm_prepare_worker")


def request(tmp_path, mode="export"):
    artifact = tmp_path / "artifact"
    artifact.mkdir(exist_ok=True)
    value = dict(
        schema_version=1,
        owner_pid=42,
        owner_start_time="123",
        artifact_path=str(artifact.resolve()),
        identity={"world_size": 1, "dtype": "torch.bfloat16"},
        engine_options={"dtype": "bfloat16", "tensor_parallel_size": 1},
        validation={
            "prompts": ["Hello"],
            "sampling": {
                "temperature": 0,
                "max_tokens": 8,
                "ignore_eos": False,
                "logprobs": 5,
            },
            "logprob_atol": 0,
        },
        reference_path=str((tmp_path / "reference.json").resolve()),
        result_path=str((tmp_path / (mode + "-result.json")).resolve()),
        store={
            "mem_pool_size": 2147483648,
            "chunk_size": 67108864,
            "num_thread": 2,
        },
    )
    if mode == "export":
        value["source_path"] = str(tmp_path.resolve())
    else:
        value["parent_network_namespace"] = 100
        value["candidate_manifest"] = {"identity": value["identity"]}
    return value


def test_runtime_module_is_available_without_execution_dependencies():
    assert callable(worker().run)


def test_request_rejects_unsupported_options_and_verify_source(tmp_path):
    module = worker()
    value = request(tmp_path)
    assert module.validate_request("export", value) == value
    value["engine_options"]["quantization"] = "fp8"
    with pytest.raises(ValueError, match="engine options"):
        module.validate_request("export", value)
    value = request(tmp_path, "verify")
    value["source_path"] = str(tmp_path)
    with pytest.raises(ValueError, match="fields"):
        module.validate_request("verify", value)


def output(token=12, logprob=-0.5):
    return SimpleNamespace(
        prompt_token_ids=[1, 2],
        finished=True,
        outputs=[
            SimpleNamespace(
                token_ids=[token] * 8,
                logprobs=[
                    {
                        token + offset: SimpleNamespace(
                            logprob=logprob - offset, rank=offset + 1
                        )
                        for offset in range(5)
                    }
                    for _ in range(8)
                ],
                finish_reason="length",
                stop_reason=None,
            )
        ],
    )


def test_generation_records_and_compares_token_logprob_and_stop_semantics():
    module = worker()
    reference = module.serialize_outputs([output()])
    assert reference[0]["prompt_token_ids"] == [1, 2]
    module.compare_outputs(reference, module.serialize_outputs([output()]), 0)
    with pytest.raises(ValueError, match="logprob"):
        module.compare_outputs(
            reference, module.serialize_outputs([output(logprob=-0.6)]), 0
        )
    with pytest.raises(ValueError, match="token"):
        module.compare_outputs(
            reference, module.serialize_outputs([output(token=13)]), 0
        )
    reference[0]["finish_reason"] = "stop"
    with pytest.raises(ValueError, match="semantics"):
        module.compare_outputs(
            reference, module.serialize_outputs([output()]), 0
        )


def test_export_uses_named_rpc_and_always_shutdown(tmp_path, monkeypatch):
    module = worker()
    value = request(tmp_path)
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(
                    shutdown=lambda **kw: calls.append(("shutdown", kw))
                )
            )

        def generate(
            self, prompts, sampling_params, *, use_tqdm, tokenization_kwargs
        ):
            assert prompts == ["Hello"] and use_tqdm is False
            assert tokenization_kwargs == {"add_special_tokens": True}
            return [output()]

        def collective_rpc(self, method, *, args):
            calls.append((method, args))
            return [{"rank": 0}]

        def get_tokenizer(self):
            return SimpleNamespace(
                eos_token_id=9, encode=lambda prompt, **kwargs: [1, 2]
            )

    monkeypatch.setattr(
        module, "_engine_api", lambda: (Engine, lambda **kwargs: kwargs)
    )
    result = module.run("export", value)
    assert result["rank_reports"] == [{"rank": 0}]
    assert ("export_serverless_model", (value["artifact_path"],)) in calls
    assert calls[-1] == ("shutdown", {"timeout": 10})
    assert calls[0]["worker_extension_cls"].endswith(
        "ServerlessLLMWorkerExtension"
    )
    assert (
        json.loads((tmp_path / "reference.json").read_text())["eos_token_id"]
        == 9
    )


def test_verify_requires_distinct_loopback_network_namespace(
    tmp_path, monkeypatch
):
    module = worker()
    with pytest.raises(RuntimeError, match="namespace"):
        module.check_network_namespace(
            100, namespace_inode=100, interfaces=[(1, "lo")]
        )
    with pytest.raises(RuntimeError, match="interfaces"):
        module.check_network_namespace(
            100, namespace_inode=101, interfaces=[(1, "lo"), (2, "eth0")]
        )
    assert module.check_network_namespace(
        100, namespace_inode=101, interfaces=[(1, "lo")]
    ) == {
        "parent_network_namespace": 100,
        "network_namespace": 101,
        "interfaces": ["lo"],
    }


def test_offline_audit_rejects_nonloopback_but_allows_local_ipc():
    module = worker()
    import socket

    with socket.socket() as client:
        with pytest.raises(PermissionError, match="non-loopback"):
            module.offline_audit("socket.connect", (client, ("8.8.8.8", 443)))
        module.offline_audit("socket.connect", (client, ("127.0.0.1", 19001)))
    with pytest.raises(PermissionError):
        module.offline_audit(
            "socket.getaddrinfo", ("huggingface.co", 443, 0, 0, 0)
        )


def test_export_failure_still_explicitly_shuts_down(tmp_path, monkeypatch):
    module = worker()
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(
                    shutdown=lambda **kw: calls.append(kw)
                )
            )

        def generate(self, *args, **kwargs):
            raise RuntimeError("injected generation failure")

    monkeypatch.setattr(
        module, "_engine_api", lambda: (Engine, lambda **kwargs: kwargs)
    )
    with pytest.raises(RuntimeError, match="injected"):
        module.run("export", request(tmp_path))
    assert calls == [{"timeout": 10}]


def test_watchdog_owner_identity_parser_handles_parentheses():
    module = worker()
    fields = ["S"] + ["0"] * 18 + ["123456"]
    assert module.parse_process_identity(
        "42 (worker (special)) " + " ".join(fields)
    ) == ("S", "123456")


def test_private_store_always_kills_owned_process_on_readiness_failure(
    tmp_path, monkeypatch
):
    module = worker()
    calls = []

    class Process:
        def __init__(self, *args, **kwargs):
            self.dead = False

        def poll(self):
            return 0 if self.dead else None

        def terminate(self):
            calls.append("terminate")
            self.dead = True

        def wait(self, **kwargs):
            calls.append("wait")

    monkeypatch.setattr(module.subprocess, "Popen", Process)

    def fail(*args):
        raise TimeoutError("readiness")

    monkeypatch.setattr(module, "_wait_store", fail)
    with pytest.raises(TimeoutError):
        with module._private_store(request(tmp_path, "verify")):
            pytest.fail("must never yield a nonready endpoint")
    assert calls == ["terminate", "wait"]


@pytest.mark.parametrize("encoded", [[1, 2], [2, 1]])
def test_verify_passes_only_after_independent_restore_and_checks_artifact(
    tmp_path, monkeypatch, encoded
):
    module = worker()
    value = request(tmp_path, "verify")
    reference = dict(
        schema_version=1,
        identity=value["identity"],
        engine_options=value["engine_options"],
        validation=value["validation"],
        eos_token_id=9,
        outputs=module.serialize_outputs([output()]),
    )
    (tmp_path / "reference.json").write_text(json.dumps(reference))
    calls = []
    import contextlib
    import sys

    @contextlib.contextmanager
    def private_store(req):
        calls.append("store-start")
        try:
            yield "127.0.0.1:19001"
        finally:
            calls.append("store-stop")

    class Engine:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(
                    shutdown=lambda **kw: calls.append("shutdown")
                )
            )

        def generate(self, prompts, params, *, use_tqdm):
            calls.append("generate")
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            return [output()]

        def get_tokenizer(self):
            return SimpleNamespace(
                eos_token_id=9, encode=lambda prompt, **kwargs: encoded
            )

    artifact = SimpleNamespace(
        validate_candidate_artifact=lambda *args: calls.append("artifact-check")
    )
    monkeypatch.setitem(sys.modules, "sllm_store.artifact", artifact)
    monkeypatch.setattr(module, "_private_store", private_store)
    monkeypatch.setattr(
        module, "check_network_namespace", lambda *args: {"interfaces": ["lo"]}
    )
    monkeypatch.setattr(module, "_enable_loopback", lambda: None)
    monkeypatch.setattr(
        module, "_engine_api", lambda: (Engine, lambda **kw: kw)
    )
    if encoded != [1, 2]:
        with pytest.raises(ValueError, match="tokenizer"):
            module.run("verify", value)
        assert "generate" not in calls
        assert calls[-2:] == ["shutdown", "store-stop"]
        return
    result = module.run("verify", value)
    assert (
        result["passed"]
        and result["tokens_verified"]
        and result["logprobs_verified"]
    )
    assert calls[0] == "artifact-check" and calls[-1] == "artifact-check"
    assert calls[-3:-1] == ["shutdown", "store-stop"]
    engine_options = next(item for item in calls if isinstance(item, dict))
    assert engine_options["model"] == value["artifact_path"]
    assert engine_options["load_format"] == "serverless_llm"
    assert (
        engine_options["model_loader_extra_config"]["store_address"]
        == "127.0.0.1:19001"
    )


def test_offline_file_guard_rejects_external_weights_and_artifact_writes(
    tmp_path, monkeypatch
):
    module = worker()
    import os

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setenv("SLLM_VERIFY_ARTIFACT", str(artifact.resolve()))
    with pytest.raises(PermissionError, match="original weights"):
        module.offline_audit(
            "open", (str(tmp_path / "model.safetensors"), "r", os.O_RDONLY)
        )
    with pytest.raises(PermissionError, match="read-only"):
        module.offline_audit(
            "open", (str(artifact / "config.json"), "w", os.O_WRONLY)
        )
    module.offline_audit(
        "open", (str(artifact / "rank_0/model.bin"), "r", os.O_RDONLY)
    )


@pytest.mark.skipif(
    __import__("sys").platform != "linux",
    reason="Linux private process session contract",
)
def test_real_cpu_watchdog_stops_private_runtime_after_owner_death(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    module = worker()
    owner = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    runtime = None
    try:
        _, start = module.parse_process_identity(
            __import__("pathlib").Path(f"/proc/{owner.pid}/stat").read_text()
        )
        script = 'import sys,time; from sllm.vllm_prepare_worker import _owner_watchdog; req={"owner_pid":int(sys.argv[1]),"owner_start_time":sys.argv[2]};\nwith _owner_watchdog(req):\n print("ready",flush=True); time.sleep(60)'
        runtime = subprocess.Popen(
            [sys.executable, "-c", script, str(owner.pid), start],
            start_new_session=True,
            stdout=subprocess.PIPE,
        )
        import selectors

        with selectors.DefaultSelector() as selector:
            selector.register(runtime.stdout, selectors.EVENT_READ)
            assert selector.select(10), "watchdog startup timed out"
            assert runtime.stdout.readline() == b"ready\n"
        owner.kill()
        owner.wait(timeout=5)
        assert runtime.wait(timeout=10) < 0
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)
        if runtime is not None:
            try:
                os.killpg(runtime.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            runtime.wait(timeout=5)
            runtime.stdout.close()


def test_request_rejects_boolean_tp_and_noninteger_budgets(tmp_path):
    module = worker()
    value = request(tmp_path)
    value["engine_options"]["tensor_parallel_size"] = True
    with pytest.raises(ValueError, match="positive integer"):
        module.validate_request("export", value)
    value = request(tmp_path)
    value["engine_options"]["kv_cache_memory_bytes"] = 64.5
    with pytest.raises(ValueError, match="positive integer"):
        module.validate_request("export", value)


@pytest.mark.parametrize(
    "case",
    [
        "missing-selected",
        "too-few",
        "missing-rank",
        "zero-rank",
        "bool-rank",
        "bad-prompt-id",
        "bad-generated-id",
        "bad-top-id",
        "too-many",
        "short-length",
        "unknown-finish",
    ],
)
def test_generation_rejects_incomplete_frozen_profile_output(case):
    module = worker()
    response = output()
    completion = response.outputs[0]
    if case == "missing-selected":
        completion.logprobs[0] = {
            100 + i: SimpleNamespace(logprob=-i, rank=i + 1) for i in range(5)
        }
    elif case == "too-few":
        completion.logprobs[0] = {12: SimpleNamespace(logprob=-0.5, rank=1)}
    elif case in {"missing-rank", "zero-rank", "bool-rank"}:
        entry = completion.logprobs[0][12]
        if case == "missing-rank":
            del entry.rank
        else:
            entry.rank = 0 if case == "zero-rank" else True
    elif case == "bad-prompt-id":
        response.prompt_token_ids = [True]
    elif case == "bad-generated-id":
        completion.token_ids[0] = "12"
    elif case == "bad-top-id":
        completion.logprobs[0]["12"] = completion.logprobs[0].pop(12)
    elif case == "too-many":
        completion.token_ids.append(12)
        completion.logprobs.append(completion.logprobs[0])
    elif case == "short-length":
        completion.token_ids = completion.token_ids[:1]
        completion.logprobs = completion.logprobs[:1]
    else:
        completion.finish_reason = "abort"
    with pytest.raises(ValueError):
        module.serialize_outputs([response])


def test_generation_accepts_short_natural_stop():
    response = output()
    completion = response.outputs[0]
    completion.token_ids = completion.token_ids[:1]
    completion.logprobs = completion.logprobs[:1]
    completion.finish_reason = "stop"
    assert worker().serialize_outputs([response])[0]["token_ids"] == [12]


def test_raw_prompt_encoding_matches_native_reference():
    calls = []

    def encode(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return [1, 2]

    tokenizer = SimpleNamespace(encode=encode)
    worker().validate_prompt_tokens(
        tokenizer, ["Hello"], [{"prompt_token_ids": [1, 2]}]
    )
    assert calls == [("Hello", {"add_special_tokens": True})]
    with pytest.raises(ValueError, match="tokenizer"):
        worker().validate_prompt_tokens(
            tokenizer, ["Hello"], [{"prompt_token_ids": [2, 1]}]
        )
