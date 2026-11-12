"""Exercise numerical diagnostics across real Ray serialization, without GPUs."""

import asyncio
import importlib
import json
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from sllm.backends.vllm_qualification import NumericalQualificationMixin


@pytest.mark.asyncio
async def test_numerical_diagnostic_awaits_worker_rpc_across_real_ray(
    monkeypatch, tmp_path
):
    ray = pytest.importorskip("ray")
    from test_continuation_profile import worker_report

    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "scripts/baseline")
    )
    qualification = importlib.import_module("qualify_continuation")
    ray.cloudpickle.register_pickle_by_value(qualification)
    assert not ray.is_initialized()
    assert "torch" not in sys.modules and "vllm" not in sys.modules
    report = worker_report()

    class Engine:
        failure = False

        async def collective_rpc(self, method, *, timeout):
            assert method == "sllm_numerical_runtime" and timeout == 30
            await asyncio.sleep(0)
            if self.failure:
                raise RuntimeError("Worker diagnostic failed")
            return [report]

    class Backend(NumericalQualificationMixin):
        def __init__(self):
            self.engine = Engine()

        async def ready(self):
            return True

        async def fail_diagnostic(self):
            self.engine.failure = True

    async def legacy_read(actor):
        return await actor.engine.collective_rpc(
            "sllm_numerical_runtime", timeout=30
        )

    actor = None
    with tempfile.TemporaryDirectory(prefix="snr-") as directory:
        try:
            ray.init(
                address="local",
                namespace="numerical-cpu-" + uuid.uuid4().hex,
                num_cpus=1,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=80 * 1024**2,
                _temp_dir=directory,
            )
            actor = ray.remote(num_cpus=1, num_gpus=0)(Backend).remote()
            assert await asyncio.wait_for(actor.ready.remote(), 20)
            with pytest.raises(
                ray.exceptions.RayTaskError, match="cannot pickle 'coroutine'"
            ):
                await asyncio.wait_for(
                    actor.__ray_call__.remote(legacy_read), 10
                )
            result = {}
            await asyncio.wait_for(
                qualification.record_numerical_runtime(
                    {
                        "numerical_profile": "batch-invariant-v1",
                        "gpu_uuids": [report["gpu_uuid"]],
                    },
                    tmp_path,
                    result,
                    actor,
                    "greedy",
                    "source",
                    "0",
                    report["cgroup"],
                ),
                10,
            )
            assert result["numerical_observations"][0]["validated"] is True
            assert result["numerical_observations"][0]["workers"] == [report]
            await asyncio.wait_for(actor.fail_diagnostic.remote(), 10)
            with pytest.raises(
                ray.exceptions.RayTaskError, match="Worker diagnostic failed"
            ):
                await asyncio.wait_for(
                    qualification.record_numerical_runtime(
                        {
                            "numerical_profile": "batch-invariant-v1",
                            "gpu_uuids": [report["gpu_uuid"]],
                        },
                        tmp_path,
                        result,
                        actor,
                        "greedy",
                        "target",
                        "0",
                        report["cgroup"],
                    ),
                    10,
                )
            saved = json.loads(
                (tmp_path / "continuation-result.json").read_text()
            )
            failed = saved["numerical_observations"][-1]
            assert failed["validated"] is False
            assert "Worker diagnostic failed" in failed["error"]
            assert ray.available_resources().get("GPU", 0) == 0
            assert "torch" not in sys.modules and "vllm" not in sys.modules
        finally:
            if actor is not None:
                ray.kill(actor, no_restart=True)
            ray.shutdown()
            ray.cloudpickle.unregister_pickle_by_value(qualification)
    assert not ray.is_initialized()
