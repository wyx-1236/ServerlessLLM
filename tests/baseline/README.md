# Baseline CPU contract suite

Run with the Linux `cpu` dependency closure described in
[environment.md](../../docs/baseline/environment.md). It does not install or
import Torch, vLLM, or the Store native extension. It never requests a GPU.

- `test_vllm_config.py` checks real configuration projection against a field-name
  snapshot from the pinned official engine source. This is not real constructor
  validation or model qualification.
- `test_import_boundary.py` blocks execution-dependency imports in a fresh
  interpreter, protecting the configuration-only boundary.
- `test_legacy_contract.py` exercises the unchanged original backend/downloader
  with unavoidable engine doubles. Tests named `original` or `incomplete`
  characterize inherited defects; their passing is not evidence of correctness.
  When those components are fixed, replace the defect observations with the
  intended behavior assertions and preserve the historical evidence externally.
- `test_cli_cpu.py` uses actual Click parsing with service callbacks replaced.
- `test_control_plane_cpu.py` exercises the actual FastAPI HTTP adapter and Ray
  object-reference roundtrip against a CPU actor. It forces a private local
  cluster, requests zero GPUs, checks no Torch import, and shuts down in finally.
  It is not the full Controller/Scheduler/Store/engine chain.

Windows-only development environments without Click/Ray skip those dependency
smokes explicitly. The Linux suite with the locked CPU closure must run them;
an unavailable Ray dependency is not a passing Linux acceptance result.
