# D1 original migration lifecycle implementation plan

This is the scoped execution delta for the already approved
[upgrade design](engine-upgrade.md). D1 and TP2 retain separate review gates.
It does not introduce transparent vLLM continuation, a KV connector, or a new
scheduling objective. Actual progress and results belong in the evidence index
and personal stage report, rather than this target document.

## Required invariants

1. MigrationRouter uses the parent's canonical inference instance pools and
   request/allocation ownership. All terminal, error and cancellation paths
   return each request count and slot exactly once. Its constructor accepts the
   same optional LoRA arguments Controller already supplies to both routers;
   this repairs construction compatibility without expanding LoRA qualification.
2. StorageAwareScheduler commits an instance-to-node reservation and decrements
   capacity before publishing an allocation. Its overridden control loop must
   participate in the ownership contract introduced in C. A failed migration
   cannot publish capacity that has not actually become free.
3. Target reservation, actor creation and initialization have one cleanup owner.
   Before publication, any failure or cancellation kills the target actor and
   returns its reservation; source routing remains available. Creation is
   shielded so a late actor handle cannot escape cleanup.
4. Migration retains the existing source-token sampling, delta threshold and
   target one-token replay mechanism. Remote sampling/replay are awaited rather
   than synchronously blocking the actor event loop. A replay error is a failure,
   including an error returned as a backend response body.
5. Target publication is atomic with source removal and migration mapping.
   Retiring the source always attempts shutdown, actor kill and reservation
   return, even if shutdown raises. Router close joins/cancels owned migration
   tasks so targets cannot appear after shutdown.
6. Preserve the original vLLM active-request terminal behavior: shutdown may
   abort/cancel the source request. Report this explicitly. The missing
   `resume_generate` capability remains an inherited qualification gap; adding
   transparent continuation needs a separately reviewed minimal proposal with
   remaining-token, EOS/stop, output/usage and snapshot ownership contracts.

## Execution and ownership

- Characterize current and reference behavior with real production methods and
  CPU Ray/backend doubles in `tests/baseline/test_migration_contract.py`.
  Do not count characterization of a defect as acceptance of that defect.
- Implement reservation ownership in
  `sllm/schedulers/storage_aware_scheduler.py`, with focused CPU failure tests.
  Preserve the latency-based plan selection algorithm and FCFS ordering.
- Implement migration/request lifecycle repair in
  `sllm/routers/migration_router.py`, reusing existing parent cleanup helpers
  where ownership matches. Cover late actor creation, replay/init failures,
  cancellation, close overlap, source loss and cleanup exceptions.
- In `sllm/backends/vllm_backend.py`, make `resume_kv_cache` propagate failed
  replay results instead of silently discarding a `generate` error body. Keep
  its successful return contract and one-token replay mechanism unchanged.
- Keep pinned engine interfaces in `sllm/backends/vllm_compat.py`. Do not add an
  alternate vLLM patch or modify the immutable prepared Store artifact.
- Obtain separate specification and quality reviews of each implementation
  batch, then run the focused and complete CPU baseline suite with GPUs hidden.
- Only after CPU correctness, freeze the executable, runtime, two-worker device
  mapping, prompts and fault cases. Prepare a bounded manual D1 runner and
  request the exact two GPU indices and lock mode before running it.
  A shared soft reservation does not imply immediate availability: poll both
  approved UUIDs read-only, start only when available, and stop only this task's
  cgroup if competing workloads invalidate the frozen resource boundary. Keep
  monitoring and retry within the explicitly approved window; do not kill
  competing processes or extend the window.

## Runtime evidence and limits

Actual D1 GPU evidence must show target loading, a real active source token
snapshot, identical target replay input, route publication, source termination
and both success/failure resource boundaries. Save prefix-cache configuration
and available reuse evidence; a successful `resume_kv_cache` call alone proves
neither a cache hit nor continuation. Retain source-request terminal response
even when aborted. Keep migration and TP2 results separate.

The manual runner must bind each logical Ray worker to one different approved
GPU UUID and record both logical-to-physical mappings. Use a private temporary
directory per worker: the inherited hardware collector uses a fixed temporary
disk benchmark filename, so two collectors on one host must not share it.
The two-node worker topology remains single-host qualification; do not describe
it as separate-server network or failure-domain qualification.

To exercise scheduler-triggered migration without qualifying a second serving
model, a runner may enqueue an explicitly identified zero-weight resource
allocation probe whose Store metadata is confined to the occupied source node.
Record that synthetic metadata fixture, use the unmodified latency planner and
control loop, and release the probe reservation after the target is published.
It is not a second model's loading or generation qualification. Direct invocation
of a migration plan may supplement this path, but cannot replace evidence that
the real scheduler invoked the Router.

Do not mark D1 fully accepted while required continuation or rollback evidence
is missing. Preserve failures, identify inherited defects versus regressions,
and present any behavior-expanding compatibility proposal explicitly for review.
