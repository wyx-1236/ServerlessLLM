"""Fail-closed Linux process targeting for the externally owned GPU job."""

import csv
import io
import os
import signal
from pathlib import Path, PurePosixPath


def read_identity(pid, proc_root=Path("/proc")):
    directory = Path(proc_root) / str(pid)
    stat = (directory / "stat").read_text()
    # comm is parenthesized and can itself contain spaces or parentheses.
    fields = stat[stat.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "start_time": int(fields[19]),
        "cgroup": (directory / "cgroup").read_text(),
    }


def select_worker(rows, gpu_uuid, unit, owned_cgroup, proc_root=Path("/proc")):
    if unit and not unit.endswith(".service"):
        unit += ".service"
    candidates = []
    for row in csv.reader(io.StringIO(rows)):
        if len(row) != 3:
            raise RuntimeError("Malformed GPU process inventory")
        uuid, pid, name = (item.strip() for item in row)
        if uuid == gpu_uuid and (
            name == "VLLM::Worker" or name.startswith("VLLM::Worker_")
        ):
            candidates.append(int(pid))
    if len(candidates) != 1 or candidates[0] <= 0:
        raise RuntimeError(f"Expected exactly one Worker, found {candidates}")
    identity = read_identity(candidates[0], proc_root)
    owned_paths = [line.split(":", 2)[2] for line in owned_cgroup.splitlines()]
    if (
        not unit
        or not owned_paths
        or not all(unit in PurePosixPath(path).parts for path in owned_paths)
        or identity["cgroup"] != owned_cgroup
    ):
        raise RuntimeError("Worker is outside the exact owned service cgroup")
    return identity


def kill_worker(identity, proc_root=Path("/proc")):
    # Opening first pins the process; comparing after opening detects PID reuse.
    fd = os.pidfd_open(identity["pid"])
    try:
        if read_identity(identity["pid"], proc_root) != identity:
            raise RuntimeError("Worker identity changed before fault injection")
        signal.pidfd_send_signal(fd, signal.SIGKILL)
    finally:
        os.close(fd)


def log_offsets(directory, actor_pid):
    if not isinstance(actor_pid, int) or actor_pid <= 0:
        raise RuntimeError("Invalid qualification actor PID")
    return {
        str(path): (path.stat().st_ino, path.stat().st_size)
        for pattern in (
            f"worker-*-{actor_pid}.err",
            f"worker-*-{actor_pid}.out",
        )
        for path in Path(directory).glob(pattern)
        if path.is_file()
    }


def cache_failure_evidence(error, directory, offsets, actor_pid):
    def marker(text):
        if "No available memory for the cache blocks" in text:
            return "No available memory for the cache blocks"
        prefix = "To serve at least one request with the model's max seq len"
        if prefix in text and "KV cache" in text:
            return prefix
        return None

    direct = marker(error)
    if direct:
        return {"source": "exception", "marker": direct}
    for name, (inode, size) in log_offsets(directory, actor_pid).items():
        old_inode, old_size = offsets.get(name, (inode, 0))
        # Only bytes appended since the failed actor was created qualify.
        if old_inode != inode or size < old_size:
            continue
        with Path(name).open("rb") as stream:
            stream.seek(max(old_size, size - 2 * 1024**2))
            text = stream.read(2 * 1024**2).decode("utf-8", errors="replace")
        found = marker(text)
        if found:
            return {"source": name, "marker": found}
    raise RuntimeError(
        "Init failure has no new evidence of intended KV cache rejection"
    )
