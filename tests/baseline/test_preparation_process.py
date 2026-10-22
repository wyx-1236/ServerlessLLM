"""Real CPU child-process lifetime checks for the conversion supervisor."""

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux runtime only"
)


def api():
    from sllm.vllm_preparation import run_preparation_process

    return run_preparation_process


def test_success_writes_log_and_waits_for_exit(tmp_path):
    log = tmp_path / "success.log"
    api()([sys.executable, "-c", "print('cpu child complete')"], log, timeout=5)
    assert "cpu child complete" in log.read_text()


def test_child_failure_propagates_with_log_location(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="exit code 7"):
        api()([sys.executable, "-c", "raise SystemExit(7)"], log, timeout=5)
    assert log.exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_descendants_are_stopped_on_parent_exit_or_timeout(tmp_path, timeout):
    pid_file = tmp_path / "grandchild.pid"
    program = (
        "import os,time,pathlib; pid=os.fork(); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(pid)) if pid else None; "
        f"time.sleep(30) if not pid or {timeout!r} else None"
    )
    command = [sys.executable, "-c", program]
    if timeout:
        with pytest.raises(RuntimeError, match="timed out"):
            api()(
                command, tmp_path / "timeout.log", timeout=1, cleanup_timeout=2
            )
    else:
        api()(command, tmp_path / "normal.log", timeout=5, cleanup_timeout=2)
    pid = int(pid_file.read_text())
    stat = f"/proc/{pid}/stat"
    if os.path.exists(stat):
        with open(stat) as stream:
            assert stream.read().rsplit(")", 1)[1].split()[0] == "Z"
