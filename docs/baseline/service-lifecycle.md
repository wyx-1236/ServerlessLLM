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
host. The Store listens on the namespace's interfaces only after independent
namespace and loopback-only checks pass. Readiness probes cover both backend
loopback and Ray-discovered worker addresses before Controller startup. It then exercises actual HTTP registration, generation, concurrency, errors,
deletion, restart, graceful drain and forced shutdown. Additional probes require
four simultaneously active backend requests, real Ray task cancellation, an
exactly identified owned Worker fault, and failed KV-cache initialization followed
by fresh registration. Worker signaling uses a pidfd after UUID, service cgroup
and process identity checks. Wrapped initialization errors require the pinned
KV-memory error in the failed actor's newly written logs. It never reserves GPUs
or authorizes another batch. CPU tests and source review do not qualify GPU
serving, Worker failure recovery, migration, TP2 or a general model matrix;
acceptance remains tied to separately archived execution and cleanup evidence.

## Debug history and active snapshots

`get_current_tokens()` and the `request_trace` compatibility view read only the
active registry, including requests admitted before their first engine output.
MigrationRouter, scheduler status and service qualification use this boundary.
Finishing, failing or cancelling a request removes it from that registry.

With `trace_debug=True`, finalized records are retained in `request_history`.
`await backend.get_debug_tokens()` exposes a copied `dict[request_id, list[int]]`
for diagnostic callers. A Ray caller uses `get_debug_tokens.remote()`. Each value
contains the observed prompt IDs followed by the first choice's output IDs,
matching the original token projection. Pooling records contain prompt IDs only;
records without an engine output have an empty list. Failed and cancelled
requests may have retained tokens, so presence in this view does not indicate
successful completion. Reading or changing the returned dictionary/lists does
not mutate backend records. Both snapshot methods use the request-state lock.

History collection is disabled by default. Retained records survive engine
closure for the lifetime of the backend object; they are neither counted by
draining nor passed to abort. As with the existing history dictionary, a reused
request ID replaces its previous record when that request finishes. The raw
dictionary remains available to the existing migration qualification probe,
which separately records prompt and output tokens.

The fixed upstream `get_current_tokens()` implementation also returned completed
records when debug tracing was enabled, contrary to its documented ongoing-only
contract. Its history-reading test now explicitly uses `get_debug_tokens()`.
Diagnostic callers relying on the old completed-token behavior must make the
same change. Active migration callers continue using `get_current_tokens()`.
The old shutdown test's expectation of aborting an already completed request is
also corrected. CPU coverage lives in `test_vllm_debug_history.py`; the legacy
backend test module still has other old-engine fixtures and is not qualified by
the lightweight CPU suite.

## Bounded runtime qualification

Executable commit `1478c0acd6d5f8ceca87d4cd1c8cafcb5bc8510a` passed
all 13 manual checks in batch `20261022-103635`: Qwen3-0.6B BF16, TP1,
eager, max length 512, one private Store and two Ray nodes on one host/GPU.
Four direct backend requests were simultaneously present in the active registry;
HTTP submission concurrency is recorded separately. A Worker fault returned an
error within 1.014 seconds; recovery used explicit deletion/re-registration.
The actual initialization failure used a one-byte KV budget and required its
own actor's new KV-memory error log before accepting cleanup/recovery.
The inherited HTTP error body/status convention remains unchanged.

Independent cleanup found an inactive unit, no owned processes, no GPU compute
applications and 0 MiB. See [evidence index](evidence-index.json#stage_c_runtime_qualification)
for configuration, wheel/artifact hashes, archive SHA256 and exclusions.
Stage C technical checks and user stage review passed. Original migration,
TP2, graph mode, multiple models, embedding GPU, performance and the final serving
image remain unqualified by this batch.

## D1 migration candidate (bounded checks passed; review pending)

The migration candidate uses the parent's canonical ready pool and request
ledger, accepts Controller's optional LoRA constructor arguments, and joins
request finalization even when cancellation arrives while returning a slot.
StorageAwareScheduler records ownership and capacity before publishing a
reservation, rechecks cancellation under the metadata lock, and does not
publish an allocation whose prerequisite migration failed. Migration RPCs run
outside ownership locks. Successful partial moves remain committed.

Before publication, target initialization/replay failure or cancellation owns
target shutdown, actor kill and reservation return. After atomic source/target
publication, source retirement attempts kill and resource return even when
shutdown fails. Router close joins owned migration work. Replay error bodies
now propagate through `VllmBackend.resume_kv_cache`. Optional `trace_debug`
records source snapshots and matching replay inputs for explicit qualification.

These changes retain the original token replay algorithm and delta threshold.
The final sampled sequence need not be replayed, and vLLM still lacks
`resume_generate`: abort/cancel is not transparent continuation. LoRA argument
compatibility does not qualify LoRA serving. See the
[scoped plan](migration-lifecycle-plan.md) and
[CPU evidence](evidence-index.json#stage_d1_cpu_candidate).

`scripts/baseline/qualify_migration.py` is a manual, separately approved
two-GPU runner. It requires real scheduler-triggered migration, paired token
traces, target cache counters, HTTP equality and real initialization/replay
rollback evidence, followed by cleanup. A zero-weight allocation probe exercises
the unchanged planner; it does not qualify a second serving model. CPU tests
of this runner do not establish any of those runtime outcomes.

Commit `54d099e1d9839637b82773e5ddebb13b2d0ae94a` passed all 13 checks in
batch `20261022-130523`. The actual scheduler moved Qwen3-0.6B from worker 1
(physical GPU 1) to worker 0 (physical GPU 0), retaining the original planner.
One paired replay iteration contained three actual sequences of 436, 338 and
214 tokens; target history recorded identical inputs and one output token each.
A controlled identical-prefix probe on target actor PID 2424357 observed
436 additional query tokens and 432 additional hit tokens. This proves reuse
for that probe, not a general migration hit ratio or performance result.

The target initialization fault used a one-byte KV budget and its own new
actor log marker. A second target with context length 16 rejected a real
448-token replay. Both returned target reservations and actor resources while
retaining source routing and successful fresh HTTP generation. Three source
requests in the successful move returned `finish_reason=abort`; no transparent
continuation is claimed. The original version's GPU terminal behavior was not
re-executed in this batch, so equivalence is not established by source review.

Independent release confirmed no owned PIDs, no compute applications and
0 MiB on physical GPU 0 and GPU 1. The first D1 attempt's strict UUID comparison
failed because Torch returned the bare UUID; its evidence is preserved, and
the passing commit only normalizes valid UUID representation. See
[runtime evidence](evidence-index.json#stage_d1_runtime_qualification) for
identities, resource peaks, archive hashes and exclusions. D1 review and full
baseline acceptance remain pending; TP2 belongs to a separate review gate.

## Same-request continuation qualification entry

`scripts/baseline/qualify_continuation.py --approval <approved-config.json>`
consumes a separate `stage-d2-continuation` approval. It reuses the private
Store/Ray/HTTP lifecycle and exact device/cgroup/deadline/deployment guards,
while retaining the original D1 entry and its historical acceptance meaning.
Importing it or using `--help` does not import an engine or request resources.

The bounded suite requires both greedy single-choice and explicitly seeded
two-choice requests to complete through the original HTTP call. Source freeze,
target preparation and resume events use the existing `trace_debug` setting.
Each event includes schema, backend PID and epoch; frozen/prepared events
contain full plain-data snapshots. They may include request text and logprobs,
like existing debug history, and are absent when debug tracing is disabled.

The final source snapshots, target histories, original HTTP attempt ledger,
full control outputs, request accounting and target-specific cache counters
are checked before success. An earlier active observation or a fresh HTTP
request alone cannot pass. See the [bounded scenarios and exclusions](migration-continuation-engine-plan.md#同一-http-请求的首批实机验收入口).
`continuation-result.json` does not claim externally verified GPU release.
An actual approved run and independently observed release remain required;
the presence of this entry point is not runtime qualification.
