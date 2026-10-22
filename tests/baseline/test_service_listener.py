"""Real TCP regression for private Store reachability without native/CUDA imports."""

import importlib.util
import socket
from pathlib import Path


def test_private_store_listens_on_backend_and_ray_worker_addresses(tmp_path):
    path = Path(__file__).parents[2] / "scripts/baseline/qualify_service.py"
    spec = importlib.util.spec_from_file_location(
        "service_qualification_probe", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.store_settings(tmp_path)
    # Bind an ephemeral test port using the exact service bind address. The
    # real harness first proves it is in an independent, loopback-only netns.
    with socket.socket() as listener:
        listener.bind((config["host"], 0))
        listener.listen()
        listener.settimeout(1)
        port = listener.getsockname()[1]
        for address in ["127.0.0.1", "127.0.0.2"]:
            with socket.create_connection((address, port), timeout=1):
                connection, _ = listener.accept()
                connection.close()
