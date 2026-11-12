"""vLLM 0.31 scheduler extension that recomputes KV for committed history."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from sllm.backends.vllm_resume_adapter import RESUME_KEY, SCHEMA_VERSION


class ResumeScheduler(Scheduler):
    def __new__(cls, vllm_config, *args, **kwargs):
        # vLLM resolves async_scheduling=None during config construction.
        # Select only now so CLI/default compatibility stays upstream-owned.
        if (
            cls is ResumeScheduler
            and vllm_config.scheduler_config.async_scheduling
        ):
            return object.__new__(ResumeAsyncScheduler)
        return object.__new__(cls)

    def add_request(self, request):
        params = request.sampling_params
        marker = (getattr(params, "extra_args", None) or {}).get(RESUME_KEY)
        if marker is not None:
            if not self.use_v2_model_runner:
                raise ValueError(
                    "Token-history resume requires the V2 model runner"
                )
            if marker.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported token-history resume schema")
            tokens = marker["output_token_ids"]
            if request.num_computed_tokens or request.output_token_ids:
                raise ValueError("Resume requires a fresh scheduler request")
            if len(tokens) >= request.max_tokens:
                raise ValueError("Completed choices must not be resubmitted")
            request.append_output_token_ids(tokens)
            # append_output_token_ids updates all_token_ids and block hashes;
            # computed remains zero so no source replica's KV is assumed.
            if tokens:
                # This choice already completed its first prompt prefill.
                # A target-local cache hit may also cover restored output;
                # upstream first-prefill stats assume prompt-only hits.
                # Consume that one-shot state without changing cache lookup,
                # the prompt boundary, or the real preemption counter.
                request.take_prefill_stats()
            request._sllm_resume_grammar_pending = bool(tokens)
        super().add_request(request)

    def _handle_blocked_waiting_request(self, request):
        ready = super()._handle_blocked_waiting_request(request)
        if ready and getattr(request, "_sllm_resume_grammar_pending", False):
            if not self.structured_output_manager.accept_tokens(
                request, list(request.output_token_ids)
            ):
                raise ValueError(
                    "Resume prefix was rejected by the structured-output "
                    "grammar"
                )
            request._sllm_resume_grammar_pending = False
        return ready


class ResumeAsyncScheduler(ResumeScheduler, AsyncScheduler):
    """The original AsyncScheduler, with the same history restoration hook."""
