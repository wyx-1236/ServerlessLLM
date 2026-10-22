"""CPU-only qualification fault targeting; never inspect real GPU processes."""

import importlib.util
import signal
from pathlib import Path

import pytest


def load_helper():
    path = (
        Path(__file__).parents[2] / "scripts/baseline/qualification_process.py"
    )
    assert path.exists(), "Ownership helper must exist before fault injection"
    spec = importlib.util.spec_from_file_location("qualification_process", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proc_record(
    root, pid=42, cgroup="0::/system.slice/job.service\n", start=123
):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    (directory / "cgroup").write_text(cgroup)
    # stat fields 3..22; comm may contain spaces and parentheses.
    (directory / "stat").write_text(
        f"{pid} (VLLM::Worker (TP0)) S " + "0 " * 18 + str(start)
    )


def test_selects_exact_owned_worker(tmp_path):
    helper = load_helper()
    proc_record(tmp_path)
    selected = helper.select_worker(
        "GPU-approved, 42, VLLM::Worker_TP0\nGPU-other, 99, VLLM::Worker_TP0",
        "GPU-approved",
        "job",
        "0::/system.slice/job.service\n",
        tmp_path,
    )
    assert selected == {
        "pid": 42,
        "start_time": 123,
        "cgroup": "0::/system.slice/job.service\n",
    }


@pytest.mark.parametrize(
    "rows,cgroup,unit",
    [
        ("", "0::/system.slice/job.service\n", "job.service"),
        (
            "GPU-approved, 42, python",
            "0::/system.slice/job.service\n",
            "job.service",
        ),
        (
            "GPU-approved, 42, VLLM::Worker_TP0\nGPU-approved, 43, VLLM::Worker_TP1",
            "0::/system.slice/job.service\n",
            "job.service",
        ),
        (
            "GPU-approved, 42, VLLM::Worker_TP0",
            "0::/system.slice/foreign.service\n",
            "job.service",
        ),
        (
            "GPU-approved, 42, VLLM::Worker_TP0",
            "0::/system.slice/job.service\n",
            "other.service",
        ),
        (
            "GPU-approved, 42, VLLM::Worker_TP0",
            "0::/system.slice/job.service\n",
            "ob.service",
        ),
        (
            "GPU-approved, 0, VLLM::Worker",
            "0::/system.slice/job.service\n",
            "job.service",
        ),
    ],
)
def test_rejects_ambiguous_or_foreign_target(tmp_path, rows, cgroup, unit):
    helper = load_helper()
    proc_record(tmp_path, cgroup=cgroup)
    with pytest.raises(RuntimeError):
        helper.select_worker(
            rows,
            "GPU-approved",
            unit,
            "0::/system.slice/job.service\n",
            tmp_path,
        )


def test_pidfd_rechecks_identity_before_signal(tmp_path, monkeypatch):
    helper = load_helper()
    proc_record(tmp_path)
    identity = helper.read_identity(42, tmp_path)
    calls = []
    monkeypatch.setattr(helper.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(helper.os, "pidfd_open", lambda pid: 9, raising=False)
    monkeypatch.setattr(
        helper.os, "close", lambda fd: calls.append(("close", fd))
    )
    monkeypatch.setattr(
        helper.signal,
        "pidfd_send_signal",
        lambda *args: calls.append(args),
        raising=False,
    )
    helper.kill_worker(identity, tmp_path)
    assert calls == [(9, signal.SIGKILL), ("close", 9)]
    proc_record(tmp_path, start=124)
    calls.clear()
    with pytest.raises(RuntimeError, match="identity"):
        helper.kill_worker(identity, tmp_path)
    assert calls == [("close", 9)]


def test_pidfd_rechecks_cgroup_after_open(tmp_path, monkeypatch):
    helper = load_helper()
    proc_record(tmp_path)
    identity = helper.read_identity(42, tmp_path)
    signalled = []

    def open_moved_process(pid):
        proc_record(tmp_path, cgroup="0::/foreign.service\n")
        return 9

    monkeypatch.setattr(
        helper.os, "pidfd_open", open_moved_process, raising=False
    )
    monkeypatch.setattr(helper.os, "close", lambda fd: None)
    monkeypatch.setattr(
        helper.signal,
        "pidfd_send_signal",
        lambda *args: signalled.append(args),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="identity"):
        helper.kill_worker(identity, tmp_path)
    assert signalled == []


@pytest.mark.parametrize("wrapped", [False, True])
def test_intended_cache_failure_requires_new_specific_evidence(
    tmp_path, wrapped
):
    helper = load_helper()
    log = tmp_path / "worker-bad-job-42.err"
    log.write_text("old logs\n")
    offsets = helper.log_offsets(tmp_path, 42)
    marker = "To serve at least one request with the model's max seq len (512), KV cache is needed"
    error = marker
    if wrapped:
        error = "Engine core initialization failed; see root cause above"
        with log.open("a") as stream:
            stream.write(marker)
    evidence = helper.cache_failure_evidence(error, tmp_path, offsets, 42)
    assert evidence["source"] == (str(log) if wrapped else "exception")


def test_old_cache_errors_do_not_qualify_unrelated_failure(tmp_path):
    helper = load_helper()
    log = tmp_path / "worker-bad-job-42.out"
    log.write_text("No available memory for the cache blocks.")
    offsets = helper.log_offsets(tmp_path, 42)
    with pytest.raises(RuntimeError, match="KV cache"):
        helper.cache_failure_evidence(
            "Connection refused", tmp_path, offsets, 42
        )


def test_new_unrelated_actor_cache_error_does_not_qualify(tmp_path):
    helper = load_helper()
    offsets = helper.log_offsets(tmp_path, 42)
    (tmp_path / "worker-other-job-99.err").write_text(
        "No available memory for the cache blocks."
    )
    with pytest.raises(RuntimeError, match="KV cache"):
        helper.cache_failure_evidence(
            "Engine core initialization failed", tmp_path, offsets, actor_pid=42
        )
