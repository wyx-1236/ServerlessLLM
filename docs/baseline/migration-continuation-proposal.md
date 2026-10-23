# Proposed vLLM active-request continuation compatibility

Status: proposal for user review after the bounded D1 report. Not implemented,
accepted architecture, or GPU-qualified behavior. The original migration-chain
checks passed, but three active source requests ended with `abort` in the real
batch. Full baseline acceptance must not treat those requests as continued.

## Scope and invariants

The user's binding requirement is no regression in the original ServerlessLLM
project's functions or rules. Preserving functionality is not an optional
scope choice. Establish compatibility against the fixed upstream project's
specifications, externally promised behavior, code and tests; distinguish
inherited implementation gaps from upgrade regressions without weakening
acceptance. Shared Router/Scheduler/Store changes require regression coverage
for affected existing paths, not only the new vLLM qualification cases.

Restricted early batches stage evidence collection; they do not reduce final
functional requirements. Marking a previously supported capability unsupported
or unverified does not satisfy the upgrade. This requirement clarification does
not approve the proposed handoff implementation or authorize a GPU run.

Retain the existing scheduler objective, target loading through native Store,
iterative source token sampling and one-token target prefix replay. Add an
explicit final request handoff within that mechanism; do not introduce direct
KV transfer, a KV connector, or a replacement scheduler. Keep TP2 separate.

The compatibility path must preserve request identity, output-token budget,
model/tokenizer identity, stop semantics, final text and usage. It must own each
request and instance reservation exactly once. A replay/cache hit alone is
never evidence that these requirements hold.

## Proposed ownership protocol

1. Prepare the target while the source continues serving, using the original
   warmup loop. Do not publish the target at this point.
2. Enter an explicit source handoff state: close new admission, mark each
   admitted request with a migration generation, abort/join its engine iterator,
   and capture its final observed output. Joining is necessary because sampling
   before abort can miss tokens subsequently returned by the iterator. Retain
   the original request RPC until continuation is settled.
3. Build an immutable, versioned per-request envelope containing request ID,
   source instance/generation, model/tokenizer revision, original prompt token
   IDs, committed generated token IDs, original sampling/stop configuration,
   original output budget, visible-output trimming metadata, and natural versus
   internal-abort terminal state. Do not pass live engine output objects as the
   cross-instance ownership contract.
4. Replay the final frozen token sequences on the target. Naturally completed
   source requests return their original result and are not resumed. For an
   unfinished request, remaining output budget is original `max_tokens` minus
   committed generated tokens. A zero remaining budget finalizes without another
   generation. The target receives the original request ID and handoff generation
   so repeated handoff delivery cannot generate or charge twice.
5. Publish the route and transfer pending request ownership only after final
   replay succeeds. Router's vLLM continuation branch calls a new backend
   `resume_generate` compatibility method using the envelope. Decode and merge
   token outputs with the original tokenizer; usage charges the original prompt
   once and the total committed completion tokens once. The replay's one-token
   warmup output is not user output or billed completion.
6. Retire the source after ownership transfer. Before transfer, target failure
   must leave a source-owned recoverable envelope rather than lose a request;
   source engine restart/replay may be required after it has been quiesced.
   After transfer, source cleanup errors must not roll back the live target.
   Cancellation and close settle the unique owner and return all slots/counts.

## Semantic constraints requiring explicit qualification

Replaying generated tokens as target prompt tokens does not automatically
preserve sampling semantics: random-number state, presence/frequency/repetition
penalties, minimum-token accounting, EOS and stop strings can depend on prior
generation. Stop strings may cross the handoff boundary, and visible text may
exclude tokens that still count toward the engine budget. These cases need an
explicit adapter or a capability restriction; merely forwarding SamplingParams
does not establish equivalence.

The first proposed qualification batch uses deterministic greedy generation,
neutral penalties and a finite output budget, then tests natural EOS, token
stops, boundary-spanning string stops, zero remaining budget, cancellation and
both sides of the transfer failure boundary. Each semantic capability receives
its own admission/result entry. Unsupported active-request semantics must keep
the original source request owned and prevent committing its handoff; they
cannot silently produce an altered request or be counted as passed. A policy
for unsupported requests is part of the reviewed implementation delta, not an
implicit change to the latency planner.

## Required evidence and next gate

CPU tests must prove immutable final snapshots, no missing/duplicate tokens,
budget/usage/stop handling, idempotency, rollback ownership and cancellation.
A separately approved two-GPU batch must compare a no-migration reference with
the migrated request's complete output token IDs, finish reason and usage,
while proving actual source/target device identity, final replay/cache evidence
and release. Timing and stochastic equivalence remain unqualified until tested.

Review this proposal before implementation. Do not promote D1's bounded
original-chain result to transparent continuation or final baseline release.
