# vLLM serving lifecycle

The fixed vLLM 0.31.0 boundary is `sllm.backends.vllm_compat`. Backend
construction projects configuration and validates committed Store files on CPU;
`init_backend` creates the engine and marks it running only after success.
The adapter explicitly shuts down partially constructed engines.

Admission, active request ownership and close state share one lock. Requests
are recorded before the engine yields its first output, reject duplicate active
IDs, copy caller data, and remove their records in `finally`. Debug history is
separate from active requests. Graceful `stop` closes admission and drains;
`shutdown` escalates to abort and joins the same shielded engine-close task.
The pinned engine shutdown API closes EngineCore and its IPC resources.

Round-robin allocation owns each instance slot before resolving the waiting
request. Cancellation and backend errors release both slot and request counts.
Router shutdown joins startup and teardown tasks, escalates existing drains,
and attempts every instance cleanup before reporting errors. Controller deletion
awaits router shutdown and always kills the owned router. Controller shutdown
attempts model cleanup and releases its scheduler, Store manager and job-store
actors. FCFS reservation publication and cancellation maintain resource ownership
without changing the inherited request ordering policy.

Serving Store configuration requires an explicit loopback `store_address` in
`model_loader_extra_config`. The backend derives an absolute `store_root` from
`STORAGE_PATH`, validates that the artifact is contained in it, and passes exact
committed identity to the loader. This lets Controller registration and restore
use the same `vllm/<model>/rank_<rank>` Store key. The isolated preparation
verifier may omit `store_root`; its existing artifact-parent root remains valid.
The artifact byte layout and format identities are unchanged.

`scripts/baseline/qualify_service.py` is a manual qualification executable,
requiring a human-approved, deadline-bounded external cgroup runner. It creates
an isolated network namespace's Store and a private two-node Ray cluster on one
host, then exercises actual HTTP registration, generation, concurrency, errors,
deletion, restart, graceful drain and forced shutdown. It never reserves GPUs
or authorizes another batch. CPU tests and source review do not qualify GPU
serving, Worker failure recovery, migration, TP2 or a general model matrix;
acceptance remains tied to separately archived execution and cleanup evidence.
