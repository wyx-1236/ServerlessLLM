"""Linux CPU dependency smoke, explicitly not full Controller qualification."""

import sys
import tempfile
import uuid

import pytest


@pytest.fixture
def ray_directory():
    # Ray appends session/socket names; a pytest test-name path exceeds AF_UNIX.
    with tempfile.TemporaryDirectory(prefix="sr-") as directory:
        yield directory


@pytest.mark.asyncio
async def test_actual_http_to_ray_roundtrip_without_gpu(ray_directory):
    ray = pytest.importorskip("ray", reason="Linux CPU dependency smoke")
    httpx = pytest.importorskip("httpx")
    from sllm.app_lib import create_app

    assert "torch" not in sys.modules
    assert not ray.is_initialized(), "Do not attach to an existing cluster"
    try:
        ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=80 * 1024 * 1024,
            _temp_dir=ray_directory,
        )

        @ray.remote(num_cpus=1, num_gpus=0)
        class RouterProbe:
            async def inference(self, body, action):
                return {
                    "model": body["model"],
                    "action": action,
                    "request_id": body["request_id"],
                }

        model = "cpu-probe-" + uuid.uuid4().hex
        actor = RouterProbe.options(name=model, namespace="models").remote()
        assert ray.available_resources().get("GPU", 0) == 0
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()),
            base_url="http://local-test",
        ) as client:
            assert (await client.get("/health")).json() == {"status": "ok"}
            missing = await client.post("/v1/chat/completions", json={})
            assert missing.status_code == 400
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "request_id": "cpu-roundtrip",
                },
            )
            assert response.status_code == 200
            assert response.json() == {
                "model": model,
                "action": "generate",
                "request_id": "cpu-roundtrip",
            }
        ray.kill(actor)
        assert "torch" not in sys.modules
    finally:
        ray.shutdown()
    assert not ray.is_initialized()
