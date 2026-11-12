# Baseline CPU contract suite

Run the light suite with the Linux `cpu` dependency closure described in
[environment.md](../../docs/baseline/environment.md). This profile does not
install Torch, vLLM, or the Store native extension. Torch-dependent plugin
checks explicitly skip there and run in the separate tensor profile below.
Neither profile requests a GPU.

For a complete CPU review, prepare and check the required environments before
running tests, then run the light suite with the pinned vLLM source, the tensor
profile, and the Transformers cache profile against the same checkout. A light
suite result alone is not a complete CPU result. Its Torch-dependent skips must
be matched to passing cases in a fresh tensor-profile run; earlier results do
not close the current run. Missing required dependencies or pinned source are
environment failures to resolve before reporting acceptance, even when pytest
returns success with skips. Report each profile and any remaining skips.

The local Store Python source directory is added to the test interpreter and
CPU child processes by `conftest.py`, without loading its native extensions.
Standalone packaging tests remove this source-path environment before checking
their isolated package payloads and installed environments.

- `test_vllm_config.py` checks real configuration projection against a field-name
  snapshot from the pinned official engine source. This is not real constructor
  validation or model qualification.
- `test_import_boundary.py` blocks execution-dependency imports in a fresh
  interpreter, protecting the configuration-only boundary.
- `test_legacy_contract.py` imports the current checkout's backend and checks
  preserved or repaired legacy contracts with engine doubles. The fixture name
  `legacy` does not mean it executes the immutable upstream source. A genuine
  old/new comparison must identify and extract the fixed Git objects separately.
  Tests named `original` or `incomplete`
  characterize inherited defects; their passing is not evidence of correctness.
  When those components are fixed, replace the defect observations with the
  intended behavior assertions and preserve the historical evidence externally.
  Its downloader cases now check delegation to the transactional preparation
  pipeline; they no longer assert directory-only reuse or final-target deletion.
  They cover accepted legacy `pattern`/`max_size` arguments (keyword and
  positional), ignored-parameter diagnostics, preparation identity forwarding
  and unchanged result/error propagation.
- `test_cli_cpu.py` uses actual Click parsing with service callbacks replaced.
- `test_vllm_debug_history.py` checks completed-token diagnostic access separately
  from active migration snapshots, including copied values, disabled tracing,
  cancellation/failure and draining with retained history. Engine outputs are
  CPU doubles. It complements the public-contract cases in the adapted backend
  suite below with failure and cancellation histories.
- `test_store_export_cpu.py` exercises the Store save command and export example
  with actual Click/argparse parsing, preserving their distinct path priorities,
  method parameters and destinations. Hub resolution and preparation are doubles;
  it also checks error propagation, old-output preservation and the Transformers
  save branch. The public export adapter additionally runs through the real
  transaction in `test_preparation_pipeline.py` with child-process doubles.
- `test_store_packaging.py` builds the Python payload through the real Store
  setup configuration, replacing CUDA discovery and native compilation only.
  It checks direct wheels and sdist-to-wheel builds, lock provenance, no-dependency
  installation in fresh environments, main-package import isolation and Worker
  entry points. The build cases require setuptools and wheel, installed by the
  packaging command in `environment.md`; otherwise those cases explicitly skip.
  These CPU test wheels do not contain working native extensions.
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

## Original vLLM backend test adaptation

`tests/backend_test/vllm_backend_test.py` runs alongside this suite in the light
CPU workflow. It imports the actual backend in a private test module, replaces
only vLLM types and the engine, and executes the real `vllm_compat.engine_args`,
`create`, `abort` and `close` paths. The argument field snapshot is shared with
the configuration tests. Fixtures restore their module replacements per test;
collection does not import vLLM or Torch. No model or Store artifact is loaded.

```sh
python -m pytest tests/baseline tests/backend_test/vllm_backend_test.py -q
```

The original cases are mapped to functional assertions, not removed or skipped:

| Original case | Preserved contract and adapted assertion |
| --- | --- |
| Initialization | Preserve config and UNINITIALIZED state; construct through the adapter, mark RUNNING on success, avoid duplicate construction. |
| Generate before initialization | Return the existing error envelope without creating an engine. |
| Generate | Preserve request/model identity, sampling forwarding, final text, choices, logprobs, finish reason and usage; do not mutate input. |
| Shutdown | Close the engine once; abort an active request with its public ID. Completed debug records are retained, not aborted. |
| Stop | Reject new requests while draining admitted work, without aborting it; close once after completion. Assert STOPPING while pending and DELETING after close, rather than requiring an internal call to `backend.shutdown`. |
| Current/debug tokens | Active snapshots cover pre-output and partial-output tokens; completed diagnostics use the separately approved `get_debug_tokens` view. Check both enabled and disabled tracing. |
| KV replay | Run actual backend generation for each snapshot with one-token budget and distinct IDs; preserve the input snapshots. Passing does not establish KV cache hits or transparent continuation. |
| Encode | Use an explicit pooling task and v0.31 pooling-output conversion. A single input yields one embedding, two inputs yield two; cover text and token inputs, order, IDs and usage. |

The old embedding fixture yielded two finished outputs for a single input and
expected both as separate embeddings. The adapted case checks one result per
input, preserving batch functionality rather than counting iterator updates.
The old stop assertion depended on a removed internal call, and the original
debug test mixed finalized history into the active migration view. These
assertion changes follow the accepted lifecycle and diagnostic contracts;
they do not qualify real engine constructors, numerical output, Store loading,
GPU cleanup or migration. Production code is not replaced by these tests.

## CPU Torch tensor profile

`requirements-tensor-cpu.in` and its generated hash-pinned `.txt` define a
separate Linux CPython 3.12 test environment. This uses Torch 2.13.0+cpu and must
never substitute for the serving lock or a CUDA/native ABI acceptance result.

```sh
uv venv /tmp/sllm-tensor-cpu --python 3.12.13
uv pip sync --python /tmp/sllm-tensor-cpu/bin/python --require-hashes --torch-backend cpu tests/baseline/requirements-tensor-cpu.txt
/tmp/sllm-tensor-cpu/bin/python -c "import pytest, torch; assert torch.version.cuda is None; assert not torch.cuda.is_initialized()"
/tmp/sllm-tensor-cpu/bin/python -m pytest sllm_store/tests/python/test_tensor_layout.py tests/baseline/test_vllm_store_plugin.py tests/baseline/test_artifact.py -q
```

Regenerate with uv 0.13.4 `pip compile`, `--python-platform
x86_64-unknown-linux-gnu --python-version 3.12.13 --torch-backend cpu
--generate-hashes`, using the `.in` input. Both profiles have separate CI jobs.
The tensor profile covers scalar/empty/mixed dtype tensors, persistent buffers,
shared storage, transposed/sliced views and validation before binding. Real
vLLM construction, native Store IPC and GPU capacity still require approval and
separate runtime evidence.

## Transformers CPU cache profile

`tests/backend_test/transformers_cache_cpu_test.py` uses real Transformers
5.17.0 and Torch 2.13.0+cpu with an offline, randomly initialized tiny OPT model
and local tokenizer. It exercises the actual backend's generation/streamer,
original preemption envelope, initial and growing token-snapshot replay and
one cached continuation. Native Store loading and the hard-coded CUDA token
transfer are replaced; the model's `generate` and Cache implementations are not.
The backend retains the returned Cache object, omits the removed
`return_legacy_cache` option and uses `get_seq_length()` for its replay log.
The legacy model-backed suite's cache assertions read `layers[].keys/values`;
the rest of that suite still requires its original models and separate validation.
PEFT/datasets imports use their pinned real packages, but their features are not
qualified by these cache tests. No pretrained model is downloaded.

```sh
uv venv /tmp/sllm-transformers-cpu --python 3.12.13
uv pip sync --python /tmp/sllm-transformers-cpu/bin/python --require-hashes --torch-backend cpu tests/baseline/requirements-transformers-cpu.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/sllm-transformers-cpu/bin/python -c "import pytest, torch, transformers, peft, datasets; assert torch.version.cuda is None; assert not torch.cuda.is_initialized()"
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/sllm-transformers-cpu/bin/python -m pytest tests/backend_test/transformers_cache_cpu_test.py -q
```

Regenerate the hash-pinned profile with uv 0.13.4 from the serving lock's
registry versions, substituting only the explicitly separate CPU Torch build:

```sh
python - <<'PY'
import pathlib, tomllib
lock = tomllib.loads(pathlib.Path('uv.lock').read_text())
pins = sorted({p['name'] + '==' + p['version'] for p in lock['package']
               if p['name'] != 'torch' and 'registry' in p['source']})
pathlib.Path('/tmp/sllm-transformers-serving-constraints.txt').write_text('\n'.join(pins) + '\n')
PY
uv pip compile tests/baseline/requirements-transformers-cpu.in --constraint /tmp/sllm-transformers-serving-constraints.txt --python-platform x86_64-unknown-linux-gnu --python-version 3.12.13 --torch-backend cpu --generate-hashes --no-annotate --no-emit-index-url -o tests/baseline/requirements-transformers-cpu.txt
```

Continuation checks cover final source snapshots shorter than, equal to and
longer than speculative replay, plus changed prefixes. Only a verified strict
prefix cache is reused. The source sequence is authoritative, default and
explicit remaining budgets are bounded, zero budget does not generate, and
EOS at the budget boundary remains a stop. Prompt usage is counted once.
The real-model quality, LoRA, training, embedding, native Store, repeated
Transformers preemption and real Router/two-model migration remain separate
checks.

## Active vLLM request migration

The backend and Router tests exercise the versioned freeze/prepare/commit/
rollback protocol, including a single caller crossing the two actual Python
state machines. Engine execution is replaced by controlled iterators; these
tests cannot qualify GPU execution or cache hits. Dedicated races cover late
admission, partial freeze failure, repeated migration, cancellation, finalizer
ownership, and shutdown. Normal generation, input priority and response schema
continue to run in the original backend contract suite.

`test_vllm_migration_ray_cpu.py` starts an isolated local Ray runtime with zero
GPUs. It fills every data request slot and checks control RPC progress with the
production actor concurrency reserve. This is independent of in-process RPC
doubles and does not contact a shared Ray cluster.

`test_vllm_resume_adapter.py` executes selected classes/methods from hash-pinned
vLLM 0.31.0 source, including sampling parameters, token history, incremental
detokenization and logprobs. It replaces their surrounding runtime imports;
it does not import or run the full vLLM engine, scheduler or GPU sampler.
The source hashes match the locked wheel. Fetch only those small Python files
into a cache outside the repository and add the pinned msgspec dependency:

```sh
uv pip install --python /tmp/sllm-cpu/bin/python --require-hashes -r tests/baseline/requirements-vllm-source-cpu.txt
/tmp/sllm-cpu/bin/python tests/baseline/fetch_vllm_resume_source.py /tmp/sllm-vllm-source
CUDA_VISIBLE_DEVICES='' SLLM_VLLM_031_SOURCE=/tmp/sllm-vllm-source /tmp/sllm-cpu/bin/python -m pytest tests/baseline tests/backend_test/vllm_backend_test.py -q
```

CI provides this source explicitly. Without it, the source-execution cases
report a skip; that skip is not acceptance evidence. Real V2 sampler behavior,
structured output, multi-choice generation, full HTTP migration and resource
release still require the separately approved GPU qualification.

The manual `qualify_continuation.py` entry has separate CPU tests:

- `test_continuation_evidence.py` rejects mismatched PID/epoch/request IDs,
  missing or conflicting snapshots, token/text/logprob prefix loss, wrong
  budgets/usage and old abort or fresh-request results masquerading as resume.
- `test_continuation_qualification.py` checks import safety, mutually exclusive
  D1/continuation approval scopes, complete scenario/cleanup requirements,
  all-choice history, no-retry HTTP attempt accounting and debug-only evidence.
- `test_continuation_flow.py` exercises the actual scenario orchestration with
  controlled HTTP/actor/cache boundaries, including late allocation cleanup.

These tests neither call the resource-owning runner nor load model weights.
The continuation entry and the retained D1 entry share startup/cleanup code,
so both qualification test modules must run after changes to that boundary.
The CPU workflow watches baseline scripts and the selected `vllm-upgrade`
branch as well as the existing `codex/` branches.
