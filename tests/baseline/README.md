# Baseline CPU contract suite

Run the light suite with the Linux `cpu` dependency closure described in
[environment.md](../../docs/baseline/environment.md). This profile does not
install Torch, vLLM, or the Store native extension. Torch-dependent plugin
checks explicitly skip there and run in the separate tensor profile below.
Neither profile requests a GPU.

- `test_vllm_config.py` checks real configuration projection against a field-name
  snapshot from the pinned official engine source. This is not real constructor
  validation or model qualification.
- `test_import_boundary.py` blocks execution-dependency imports in a fresh
  interpreter, protecting the configuration-only boundary.
- `test_legacy_contract.py` exercises the original backend
  with unavoidable engine doubles. Tests named `original` or `incomplete`
  characterize inherited defects; their passing is not evidence of correctness.
  When those components are fixed, replace the defect observations with the
  intended behavior assertions and preserve the historical evidence externally.
  Its downloader cases now check delegation to the transactional preparation
  pipeline; they no longer assert directory-only reuse or final-target deletion.
- `test_cli_cpu.py` uses actual Click parsing with service callbacks replaced.
- `test_control_plane_cpu.py` exercises the actual FastAPI HTTP adapter and Ray
  object-reference roundtrip against a CPU actor. It forces a private local
  cluster, requests zero GPUs, checks no Torch import, and shuts down in finally.
  It is not the full Controller/Scheduler/Store/engine chain.
- `test_native_writer.py` compiles the production disk writer alone with Linux
  g++; it checks real O_DIRECT files and injected I/O failures without CUDA.
- `test_native_reader.py` compiles the production partition reader without
  CUDA; real O_DIRECT reads cover logical tails, unaligned destination/range
  boundaries, multiple partitions, short reads, EINTR, I/O errors and truncation.
- `test_artifact.py` checks byte inventories, tensor bounds, publication and
  reuse rejection. Its verifier doubles do not qualify engine restoration.
- `test_store_load_contract.py` checks private Store endpoint forwarding and
  failed-confirm propagation with strict native/RPC doubles.
- `test_preparation_process.py` starts real CPU child processes and checks group
  cleanup on parent exit and timeout; it does not prove CUDA client cleanup.
- `test_preparation_source.py` and `test_preparation_pipeline.py` check frozen
  source identity, export/restore ordering and failure publication, using real
  files and strict child-command doubles. The namespace probe is also doubled
  in pipeline tests; this is not actual network isolation qualification.
- `test_prepare_worker.py` checks request/response contracts, private Store and
  engine cleanup with doubles, and a real CPU owner-death watchdog on Linux.
- `test_store_manager_preparation.py` checks preparation-before-registration,
  dtype aliases, actor cleanup and rank registration failures with Ray doubles.
- `test_vllm_store_plugin.py` checks registration and loader/Worker boundaries.
  Actual CPU Torch state is used where available; vLLM/native APIs are doubles.

Windows-only development environments without Click/Ray skip those dependency
smokes explicitly. The Linux suite with the locked CPU closure must run them;
an unavailable Ray dependency is not a passing Linux acceptance result.

## CPU Torch tensor profile

`requirements-tensor-cpu.in` and its generated hash-pinned `.txt` define a
separate Linux CPython 3.12 test environment. This uses Torch 2.13.0+cpu and must
never substitute for the serving lock or a CUDA/native ABI acceptance result.

```sh
uv venv /tmp/sllm-tensor-cpu --python 3.12.13
uv pip sync --python /tmp/sllm-tensor-cpu/bin/python --require-hashes --torch-backend cpu tests/baseline/requirements-tensor-cpu.txt
/tmp/sllm-tensor-cpu/bin/python -m pytest sllm_store/tests/python/test_tensor_layout.py tests/baseline/test_vllm_store_plugin.py tests/baseline/test_artifact.py -q
```

Regenerate with uv 0.13.4 `pip compile`, `--python-platform
x86_64-unknown-linux-gnu --python-version 3.12.13 --torch-backend cpu
--generate-hashes`, using the `.in` input. Both profiles have separate CI jobs.
The tensor profile covers scalar/empty/mixed dtype tensors, persistent buffers,
shared storage, transposed/sliced views and validation before binding. Real
vLLM construction, native Store IPC and GPU capacity still require approval and
separate runtime evidence.
