# vLLM 0.31.0 upgrade: code review map

This branch contains runtime integration and required build changes. Research
plans, qualification runners, test changes, dependency locks and detailed run
records are maintained separately and excluded from this branch's diff.
This separation does not establish new runtime or GPU qualification.

| Area | Files | Purpose |
| --- | --- | --- |
| Engine configuration | `sllm/backends/vllm_config.py`, `vllm_compat.py` | Map legacy configuration and adapt the pinned async engine API |
| Backend lifecycle | `sllm/backends/vllm_backend.py` | Generation, cancellation, drain and shutdown |
| Store preparation | `sllm_store/sllm_store/vllm_preparation.py`, `vllm_prepare_worker.py`, `vllm_export.py` | Export, validate and commit artifacts before registration |
| Store restore | `sllm_store/sllm_store/vllm_plugin.py`, `vllm_loader.py`, `artifact.py`, `tensor_layout.py` | Restore tensors, buffers and shared storage |
| Native I/O | `sllm_store/csrc/checkpoint/`, `csrc/sllm_store/partition_reader.h`, `model.cpp` | Writer completion and aligned partition reads |
| Control-plane compatibility | `sllm/controller.py`, `routers/`, `schedulers/`, `store_manager.py` | Request, actor and reservation ownership across load/migration/failure |
| Active-request continuation | `sllm/backends/vllm_migration.py`, `vllm_resume_adapter.py`, `vllm_resume_scheduler.py` | Freeze, snapshot, prepare, resume, commit and roll back migrating requests |
| Transformers compatibility | `sllm/backends/transformers_backend.py` | Adapt replay caches and continuation budgets/EOS to upgraded shared dependencies |
| Diagnostics | `sllm/backends/vllm_qualification.py` | Opt-in numerical inspection for external qualification runners |
| Packaging | Root/Store requirements, `pyproject.toml`, `setup.py`, `build_support.py` | Dependencies, plugin discovery and package provenance |

Continuation and shared-control-plane changes are broader than replacing vLLM
imports. They remain explicit review areas. The original token replay mechanism
is retained rather than transferring KV tensors between GPUs.

## Dependency identity

The original full lock is kept with the external qualification snapshot.
`sllm_store/sllm_store/preparation-lock.sha256` retains its canonical SHA256 so
source builds and installed wheels keep the same preparation identity without
requiring that lock in this branch. The receipt is an identity, not a dependency
resolver or a new validation result. Package requirements remain in this branch.

## Qualification limits

Prior bounded Store/service evidence covers Qwen3-0.6B BF16/TP1/eager.
Same-request continuation evidence covers only the explicitly tested setup,
including its opt-in batch-invariant numerical profile. EOS/stop continuation,
Transformers runtime regression, TP2, broader models, performance and the final
serving image must not be inferred from those passes. Original upstream tests
are unchanged here; upgrade-specific tests/runners remain in the external snapshot.
