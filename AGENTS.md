# Baseline upgrade instructions

This fork owns the independent ServerlessLLM-vLLM baseline. Read
`CONTRIBUTING.md` and `docs/baseline/README.md` before making changes.
`docs/baseline/engine-upgrade.md` owns the target design and stage boundaries.
Do not integrate SLLM components into CrossPool or silently change the original
baseline's scheduling, request, storage, or migration behavior.

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
