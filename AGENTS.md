# Baseline upgrade instructions

This fork owns the independent ServerlessLLM-vLLM baseline. Read
`CONTRIBUTING.md` and `docs/baseline/README.md` before making changes.
`docs/baseline/engine-upgrade.md` owns the target design and stage boundaries.
Do not integrate SLLM components into CrossPool or silently change the original
baseline's scheduling, request, storage, or migration behavior.

## Incremental implementation reporting

For all vLLM engine-upgrade work, use the sequence: announce one bounded
change, implement it, run its applicable focused checks, report it in the
current chat, then proceed to the next bounded change within the approved scope.
Each change should address one reviewable purpose and its necessary companion
files/tests; do not accumulate an entire stage before reporting.

Every report must identify the actual checkout and branch, changed files and
key functions, previous and new behavior, the purpose and effect of the change,
its plan/stage, checks actually run and their results, remaining limitations,
and whether the changes are committed. Explain compatibility effects on the
original ServerlessLLM behavior and the next proposed bounded change. Report
failures and user-requested corrections before proceeding with unrelated work.

Reporting does not by itself require approval after every small edit. Existing
stage-review, design-change and GPU approval gates still require explicit user
approval; a report or silence does not satisfy those gates. If the user requests
review of a particular change before continuing, wait for that review.

Before editing, verify the checkout path, origin, branch and worktree state.
Use the user's currently selected Fork checkout for implementation. When it is
missing already committed upgrade work, inspect and reuse the existing upgrade
branch rather than reimplementing it or overwriting unrelated local changes.

## Stage review

Work only within the currently approved stage. After every stage, report the
changes, commit and artifact identities, validation evidence, missing coverage,
risks or design deviations, resource cleanup, and the proposed next scope.
Wait for the user's explicit review before entering the next stage. Address
review feedback within the current stage and report again. Do not continue in
background jobs or delegate around a review gate. Keep actual progress reports
outside the target design; use the user's research vault for durable reports.

## Server GPU approval

Before requesting, reserving, allocating, or using server GPUs, report the
server, every physical GPU index, lock mode, purpose, scope, and expected
duration, then wait for explicit user approval. This includes conversion,
imports that initialize CUDA, tests, and service startup. A general instruction
to implement or continue does not authorize GPU use.

Use `hard lock GPU 0,1` or `soft lock GPU 0,1` for requests, substituting all
actual indices, and `hard unlock GPU 0,1` or `soft unlock GPU 0,1` for releases.
Always identify the server. Additional resources or a scope change need new
approval. Idle GPUs and existing credentials do not grant approval.

At completion, failure, or cancellation, stop this task's GPU workloads, verify
cleanup, release its allocated resources, and report usage, results or failures,
and actual release status. Wait for explicit user confirmation before any
further project work. Do not report a successful unlock if release failed.
Do not stop other users' workloads, shared Store services, or MPS controllers.
No workflow, runner, script, or delegated agent may bypass these gates.

## Sources and publication

Preserve upstream history and licenses. Publish changes to the user's fork;
PRs target `codex/baseline-vllm`, not the official upstream. Keep hooks active
and use normal commits. Preserve unrelated files and machine-local caches;
do not apply or pop a stash without a user request.

Keep model weights, Store binaries, large results, dependency caches, and
credentials outside Git. Record source identities and evidence scopes in
`docs/baseline/provenance.json` and `docs/baseline/evidence-index.json`.
Historical prototype results do not qualify the current integration commit.
