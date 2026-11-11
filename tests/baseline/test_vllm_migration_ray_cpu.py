"""Real Ray CPU admission/control capacity; no model or engine is loaded."""

import asyncio
import sys
import tempfile
import uuid

import pytest


@pytest.mark.asyncio
async def test_full_data_capacity_leaves_room_for_migration_control(
    monkeypatch,
):
    ray = pytest.importorskip("ray", reason="Linux CPU dependency check")
    import sllm.backends as backends
    from sllm.inference_instance import start_instance

    assert not ray.is_initialized(), "Do not attach to an existing cluster"
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules

    @ray.remote(num_cpus=0, num_gpus=0)
    class AdmissionWitness:
        def __init__(self):
            self.count = 0
            self.full = asyncio.Event()

        async def admitted(self):
            self.count += 1
            if self.count == 10:
                self.full.set()

        async def wait_full(self):
            await self.full.wait()
            return self.count

    class BackendProbe:
        def __init__(self, model_name, config):
            self.witness = config["witness"]
            self.release = asyncio.Event()
            self.active = 0

        async def generate(self):
            self.active += 1
            try:
                await self.witness.admitted.remote()
                await self.release.wait()
                return "completed"
            finally:
                self.active -= 1

        async def freeze_migration(self):
            assert "torch" not in sys.modules
            assert "vllm" not in sys.modules
            return self.active

        async def shutdown(self):
            self.release.set()

    # Exercise production actor creation without importing vLLM or Torch.
    monkeypatch.setitem(vars(backends), "VllmBackend", BackendProbe)
    actors = []
    with tempfile.TemporaryDirectory(prefix="smr-") as directory:
        try:
            ray.init(
                address="local",
                namespace="migration-cpu-" + uuid.uuid4().hex,
                num_cpus=1,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=80 * 1024 * 1024,
                _temp_dir=directory,
            )
            assert ray.available_resources().get("GPU", 0) == 0
            # First reproduce the old starvation, then check the production
            # default and the capacity explicitly supplied by the Router.
            for capacity in (10, None, 14):
                witness = AdmissionWitness.remote()
                actors.append(witness)
                startup = {"num_cpus": 1, "num_gpus": 0}
                if capacity is not None:
                    startup["max_concurrency"] = capacity
                backend = start_instance._function(
                    "migration-probe-" + uuid.uuid4().hex,
                    "vllm",
                    "cpu-model",
                    {"witness": witness},
                    startup,
                )
                actors.append(backend)
                requests = [backend.generate.remote() for _ in range(10)]
                assert (
                    await asyncio.wait_for(witness.wait_full.remote(), 20) == 10
                )
                freeze = backend.freeze_migration.remote()
                if capacity == 10:
                    ready, _ = ray.wait([freeze], timeout=0.5)
                    assert not ready, "The starvation control is not saturated"
                else:
                    assert await asyncio.wait_for(freeze, 5) == 10
                    await asyncio.wait_for(backend.shutdown.remote(), 5)
                    results = await asyncio.wait_for(
                        asyncio.gather(*requests), 5
                    )
                    assert results == ["completed"] * 10
                ray.kill(backend, no_restart=True)
                actors.remove(backend)
                ray.kill(witness, no_restart=True)
                actors.remove(witness)
            assert "torch" not in sys.modules
            assert "vllm" not in sys.modules
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
            ray.shutdown()
    assert not ray.is_initialized()
