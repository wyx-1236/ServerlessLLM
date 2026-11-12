"""Pure-data configuration and evidence checks for D2 numerical profiles."""

PROFILE = "batch-invariant-v1"
ENVIRONMENT = {"VLLM_BATCH_INVARIANT": "1"}
WORKER_CLASS = "sllm.backends.vllm_qualification.NumericalQualificationWorker"
RUNNER_CLASS = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"


def selected(config):
    name = config.get("numerical_profile", "default")
    if name not in ("default", PROFILE):
        raise ValueError(f"Unknown numerical profile: {name!r}")
    return name == PROFILE


def validate_environment(config, environ):
    expected = ("1",) if selected(config) else (None, "0")
    if environ.get("VLLM_BATCH_INVARIANT") not in expected:
        raise ValueError("Numerical profile conflicts with startup environment")


def worker_environment(config):
    return dict(ENVIRONMENT) if selected(config) else {}


def backend_overrides(config):
    if not selected(config):
        return {}
    return {
        "attention_config": {"backend": "FLASH_ATTN"},
        "worker_extension_cls": WORKER_CLASS,
    }


def validate_worker_reports(reports, expected_uuid, expected_cgroup):
    from qualify_migration import canonical_gpu_uuid

    if not isinstance(reports, list) or len(reports) != 1:
        raise ValueError("Numerical profile requires one TP1 worker report")
    report = reports[0]
    if not isinstance(report, dict):
        raise ValueError("Invalid numerical worker report")
    expected = {
        "schema_version": 1,
        "engine_version": "0.31.0",
        "environment": "1",
        "batch_invariant_initialized": True,
        "use_v2_model_runner": True,
        "runner_class": RUNNER_CLASS,
        "dtype": "torch.bfloat16",
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "compute_capability": [8, 0],
        "cgroup": expected_cgroup,
    }
    for field, value in expected.items():
        actual = report.get(field)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Numerical worker mismatch: {field}={actual!r}")
    if (
        type(report.get("pid")) is not int
        or report["pid"] <= 0
        or "A100" not in str(report.get("device_name", ""))
        or canonical_gpu_uuid(report.get("gpu_uuid"))
        != canonical_gpu_uuid(expected_uuid)
    ):
        raise ValueError("Numerical worker device/process mismatch")
    layers = report.get("attention_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Missing numerical worker attention layers")
    names = set()
    for layer in layers:
        if (
            not isinstance(layer, dict)
            or not isinstance(layer.get("name"), str)
            or not layer["name"]
            or layer["name"] in names
            or layer.get("backend") != "FLASH_ATTN"
            or layer.get("batch_invariant_enabled") is not True
        ):
            raise ValueError("Numerical worker attention mismatch")
        names.add(layer["name"])


def runtime_verified(result, scenarios):
    config = result.get("approval", {})
    try:
        if not selected(config):
            return True
        uuids = config["gpu_uuids"]
        if (
            not isinstance(uuids, list)
            or len(uuids) != 2
            or len(set(uuids)) != 2
        ):
            return False
        records = result.get("numerical_observations", [])
        expected = {
            (scenario, role)
            for scenario in scenarios
            for role in ("source", "target")
        }
        if len(records) != len(expected):
            return False
        seen = set()
        for record in records:
            key = (record["scenario"], record["role"])
            if (
                key not in expected
                or key in seen
                or record.get("validated") is not True
                or record.get("error")
                or record.get("node") not in ("0", "1")
                or record["expected_uuid"] != uuids[int(record["node"])]
            ):
                return False
            validate_worker_reports(
                record["workers"],
                record["expected_uuid"],
                record["expected_cgroup"],
            )
            seen.add(key)
        return seen == expected and all(
            len(
                {
                    record["node"]
                    for record in records
                    if record["scenario"] == scenario
                }
            )
            == 2
            for scenario in scenarios
        )
    except (KeyError, TypeError, ValueError):
        return False
