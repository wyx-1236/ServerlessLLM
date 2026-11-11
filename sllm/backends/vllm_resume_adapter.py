"""Token-history migration for the pinned vLLM 0.31 V2 model runner.

Only plain data crosses the replica boundary. Imports of vLLM are deliberately
local: control-plane validation of a snapshot does not initialize an engine.
"""

import asyncio
import copy
from types import MethodType

SCHEMA_VERSION = 1
RESUME_KEY = "_sllm_resume_v1"
SCHEDULER_CLS = "sllm.backends.vllm_resume_scheduler.ResumeScheduler"
ASYNC_SCHEDULER_CLS = "sllm.backends.vllm_resume_scheduler.ResumeAsyncScheduler"


def configure_args(args):
    """Keep the upstream resolved sync/async choice and custom schedulers."""
    if getattr(args, "scheduler_cls", None) is None:
        args.scheduler_cls = SCHEDULER_CLS


def validate_engine(engine):
    config = engine.vllm_config
    if not config.use_v2_model_runner:
        raise ValueError("Request migration requires the vLLM V2 model runner")
    scheduler = config.scheduler_config.scheduler_cls
    if scheduler not in (SCHEDULER_CLS, ASYNC_SCHEDULER_CLS):
        raise ValueError("Request migration requires the resume scheduler")
    if not hasattr(engine, "_sllm_resume_adapter"):
        raise ValueError("Request migration frontend is not installed")


def _encode_sampling(params):
    import msgspec

    # to_builtins(params) omits defaults. Preserve *all* effective fields,
    # including tokenizer-derived EOS, stop IDs and bad-word tokenization.
    values = {
        field.name: getattr(params, field.name)
        for field in msgspec.structs.fields(type(params))
    }
    values["extra_args"] = dict(values.get("extra_args") or {})
    values["extra_args"].pop(RESUME_KEY, None)
    return msgspec.to_builtins(values)


def _decode_sampling(values):
    import msgspec
    from vllm.sampling_params import SamplingParams

    return msgspec.convert(copy.deepcopy(values), type=SamplingParams)


def _metadata(request, prompt=None, params=None, request_id=None):
    import msgspec

    params = params or request.sampling_params
    capture_error = None
    request_options = {}
    try:
        sampling = _encode_sampling(params)
        request_options = msgspec.to_builtins(
            {
                name: getattr(request, name, None)
                for name in (
                    "reasoning_ended",
                    "reasoning_parser_kwargs",
                    "cache_salt",
                )
            }
        )
    except (TypeError, ValueError) as error:
        # Encoding is an extra migration constraint, not an admission rule.
        capture_error = f"Sampling parameters cannot be migrated: {error}"
        output_kind = getattr(params, "output_kind", None)
        sampling = {
            key: getattr(params, key, default)
            for key, default in (
                ("n", 1),
                ("max_tokens", None),
                ("min_tokens", 0),
                ("output_text_buffer_length", 0),
            )
        }
        sampling["output_kind"] = getattr(output_kind, "value", output_kind)
    for field in ("mm_features", "prompt_embeds", "lora_request", "resumable"):
        value = getattr(request, field, None)
        if (
            value is not None
            and value is not False
            and not (isinstance(value, list) and not value)
        ):
            capture_error = f"Request migration does not encode {field}"
    metadata = {
        "request_id": request_id
        or request.external_req_id
        or request.request_id,
        "prompt": prompt,
        "prompt_token_ids": list(request.prompt_token_ids or []),
        "sampling_params": sampling,
        "request_options": request_options,
    }
    if capture_error is not None:
        metadata["capture_error"] = capture_error
    return metadata


def validate_metadata(metadata):
    if metadata.get("capture_error"):
        raise ValueError(metadata["capture_error"])


async def prepare_request(engine, inputs, sampling_params, request_id):
    """Normalize once using the same processor as AsyncLLM.add_request.

    The returned EngineCoreRequest is accepted directly by pinned AsyncLLM;
    its exact token IDs and generation defaults are known before admission.
    """
    request = await engine.input_processor.process_inputs_async(
        request_id,
        inputs,
        sampling_params,
        supported_tasks=await engine.get_supported_tasks(),
    )
    prompt = inputs if isinstance(inputs, str) else inputs.get("prompt")
    return request, _metadata(request, prompt)


def request_metadata(engine, request_id):
    value = engine._sllm_resume_adapter["metadata"].get(request_id)
    return copy.deepcopy(value)


def forget_request(engine, request_id):
    engine._sllm_resume_adapter["metadata"].pop(request_id, None)
    engine._sllm_resume_adapter["pending"].pop(request_id, None)
    engine._sllm_resume_adapter["registration"].pop(request_id, None)


def _registration(adapter, request_id):
    return adapter["registration"].setdefault(
        request_id, dict(event=asyncio.Event(), submitted=set(), expected=None)
    )


async def wait_registered(engine, request_id):
    """Wait for all children to reach core, not merely frontend registration.

    The caller must race this against its generation task: failed submissions
    intentionally do not signal success.
    """
    registration = _registration(engine._sllm_resume_adapter, request_id)
    await registration["event"].wait()


def _encode_logprobs(rows):
    if rows is None:
        return None
    return [
        None
        if row is None
        else {
            int(token): {
                "logprob": value.logprob,
                "rank": value.rank,
                "decoded_token": value.decoded_token,
            }
            for token, value in row.items()
        }
        for row in rows
    ]


def _decode_logprobs(rows, flat=False):
    if rows is None:
        return None
    from vllm.logprobs import FlatLogprobs, Logprob

    result = FlatLogprobs() if flat else []
    result.extend(
        None
        if row is None
        else {int(token): Logprob(**value) for token, value in row.items()}
        for row in rows
    )
    return result


def snapshot_request(metadata, latest_output, *, model, identity):
    """Capture cumulative output; missing choices have empty history."""
    validate_metadata(metadata)
    snapshot = copy.deepcopy(metadata)
    snapshot.update(
        schema_version=SCHEMA_VERSION,
        model=model,
        identity=copy.deepcopy(identity),
        choices=[],
        prompt_logprobs=None,
        prompt_token_id_logprobs=None,
    )
    outputs = (
        {}
        if latest_output is None
        else {output.index: output for output in latest_output.outputs}
    )
    for index in range(snapshot["sampling_params"].get("n", 1)):
        output = outputs.get(index)
        snapshot["choices"].append(
            dict(
                index=index,
                token_ids=[] if output is None else list(output.token_ids),
                text="" if output is None else output.text,
                logprobs=None
                if output is None
                else _encode_logprobs(output.logprobs),
                cumulative_logprob=None
                if output is None
                else output.cumulative_logprob,
                finish_reason=None if output is None else output.finish_reason,
                stop_reason=None if output is None else output.stop_reason,
            )
        )
    if latest_output is not None:
        snapshot["prompt_logprobs"] = _encode_logprobs(
            latest_output.prompt_logprobs
        )
        token_logprobs = getattr(
            latest_output, "prompt_token_id_logprobs", None
        )
        if token_logprobs is not None:
            snapshot["prompt_token_id_logprobs"] = token_logprobs.tolist()
    _validate_snapshot(snapshot)
    return snapshot


def normalize_abort_output(output, metadata):
    """Undo the abort-only stop-buffer flush before migration captures it.

    A normal client cancellation retains upstream ABORT semantics. Call this
    only when the backend intentionally aborts an engine for migration.
    """
    output = copy.deepcopy(output)
    params = metadata["sampling_params"]
    stop = params.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    # Match BaseIncrementalDetokenizer itself. SamplingParams exposes a
    # derived output_text_buffer_length field, but callers can set that field
    # directly; the detokenizer recomputes its own buffer from stop strings.
    buffer_length = (
        max(len(text) for text in stop) - 1
        if stop and not params.get("include_stop_str_in_output", False)
        else 0
    )
    for choice in output.outputs:
        if choice.finish_reason == "abort":
            if buffer_length:
                choice.text = choice.text[:-buffer_length]
            choice.finish_reason = None
            choice.stop_reason = None
    output.finished = all(
        choice.finish_reason is not None for choice in output.outputs
    )
    return output


def _validate_snapshot(snapshot):
    validate_metadata(snapshot)
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported vLLM resume snapshot version")
    for key in (
        "request_id",
        "model",
        "identity",
        "prompt_token_ids",
        "sampling_params",
        "choices",
        "prompt_logprobs",
    ):
        if key not in snapshot:
            raise ValueError(f"Incomplete vLLM resume snapshot: {key}")
    params = snapshot["sampling_params"]
    max_tokens = params.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("Resume snapshot requires resolved max_tokens")
    choices = snapshot["choices"]
    if [choice["index"] for choice in choices] != list(
        range(params.get("n", 1))
    ):
        raise ValueError(
            "Resume snapshot must contain every choice in index order"
        )
    for ids in [snapshot["prompt_token_ids"]] + [
        choice["token_ids"] for choice in choices
    ]:
        if not isinstance(ids, list) or any(
            type(token) is not int or token < 0 for token in ids
        ):
            raise ValueError("Resume snapshot contains invalid token IDs")
    if not snapshot["prompt_token_ids"]:
        raise ValueError(
            "Resume snapshot requires the exact nonempty original prompt"
        )
    for choice in choices:
        if len(choice["token_ids"]) > max_tokens:
            raise ValueError(
                "Resume output history exceeds the original token budget"
            )
        rows = choice["logprobs"]
        if rows is not None and len(rows) != len(choice["token_ids"]):
            raise ValueError(
                "Resume logprobs must cover the committed output prefix"
            )
        if (
            choice["token_ids"]
            and (
                params.get("logprobs") is not None
                or params.get("logprob_token_ids")
            )
            and rows is None
        ):
            raise ValueError("Resume snapshot lost committed output logprobs")
        if (
            choice["token_ids"]
            and params.get("prompt_logprobs") is not None
            and snapshot["prompt_logprobs"] is None
        ):
            raise ValueError("Resume snapshot lost original prompt logprobs")


def validate_snapshot(engine, snapshot):
    validate_engine(engine)
    _validate_snapshot(snapshot)
    if len(snapshot["prompt_token_ids"]) >= engine.model_config.max_model_len:
        raise ValueError("Resume prompt exceeds the destination model length")


def _unfinished(snapshot):
    limit = snapshot["sampling_params"]["max_tokens"]
    return [
        choice
        for choice in snapshot["choices"]
        if choice["finish_reason"] is None and len(choice["token_ids"]) < limit
    ]


def child_request_ids(snapshot, internal_root):
    return [
        f"{internal_root}:choice:{choice['index']}"
        for choice in _unfinished(snapshot)
    ]


def _restore_frontend(state, snapshot, choice):
    # Replaying through the actual incremental detokenizer restores byte/token
    # offsets and withheld stop-string characters, not merely visible text.
    for token in choice["token_ids"]:
        if state.detokenizer.update([token], False) is not None:
            raise ValueError(
                "Unfinished resume prefix already contains a stop string"
            )
    if state.detokenizer.get_next_output_text(False, False) != choice["text"]:
        raise ValueError("Resume detokenizer does not reproduce committed text")
    state.sent_tokens_offset = len(choice["token_ids"])
    processor = state.logprobs_processor
    flat = snapshot["sampling_params"].get("flat_logprobs", False)
    if choice["logprobs"] is not None:
        processor.logprobs = _decode_logprobs(choice["logprobs"], flat)
        processor.cumulative_logprob = choice["cumulative_logprob"]
    # V2 skips prompt scoring when prefill contains output history. Empty
    # choices execute a normal prefill and must not append to copied scores.
    if choice["token_ids"] and snapshot["prompt_logprobs"] is not None:
        processor.prompt_logprobs = _decode_logprobs(
            snapshot["prompt_logprobs"], flat
        )
    token_logprobs = snapshot.get("prompt_token_id_logprobs")
    if choice["token_ids"] and token_logprobs is not None:
        import numpy as np

        processor.prompt_token_id_logprobs = np.asarray(token_logprobs)


def install(engine):
    """Wrap one engine's processor; registration precedes core submit."""
    if hasattr(engine, "_sllm_resume_adapter"):
        return
    adapter = engine._sllm_resume_adapter = {
        "metadata": {},
        "pending": {},
        "registration": {},
    }
    processor = engine.output_processor
    original_add_request = processor.add_request

    def add_request(
        self, request, prompt, parent_req=None, request_index=0, queue=None
    ):
        external_id = (
            parent_req.external_req_id
            if parent_req
            else request.external_req_id
        )
        params = (
            parent_req.sampling_params
            if parent_req
            else request.sampling_params
        )
        if params is not None:
            adapter["metadata"][external_id] = _metadata(
                request, prompt, params, external_id
            )
        _registration(adapter, external_id)["expected"] = (
            parent_req.n if parent_req else 1
        )
        pending = adapter["pending"].get(external_id)
        marker = (getattr(params, "extra_args", None) or {}).get(RESUME_KEY)
        if marker is not None and pending is None:
            raise ValueError("Resume frontend state is missing")
        if pending is not None:
            validate_metadata(adapter["metadata"][external_id])
        original_add_request(request, prompt, parent_req, request_index, queue)
        if pending is not None:
            snapshot, choice = pending
            _restore_frontend(
                self.request_states[request.request_id], snapshot, choice
            )

    processor.add_request = MethodType(add_request, processor)
    core = engine.engine_core
    original_submit = core.add_request_async

    async def submit(self, request):
        await original_submit(request)
        registration = _registration(adapter, request.external_req_id)
        registration["submitted"].add(request.request_id)
        if len(registration["submitted"]) == registration["expected"]:
            registration["event"].set()

    core.add_request_async = MethodType(submit, core)


def _request_output(snapshot):
    from vllm.outputs import CompletionOutput, RequestOutput

    flat = snapshot["sampling_params"].get("flat_logprobs", False)
    outputs = []
    for choice in snapshot["choices"]:
        values = copy.deepcopy(choice)
        if (
            values["finish_reason"] is None
            and len(values["token_ids"])
            >= snapshot["sampling_params"]["max_tokens"]
        ):
            values["finish_reason"] = "length"
        values["logprobs"] = _decode_logprobs(values["logprobs"], flat)
        outputs.append(CompletionOutput(**values))
    result = RequestOutput(
        request_id=snapshot["request_id"],
        prompt=snapshot.get("prompt"),
        prompt_token_ids=list(snapshot["prompt_token_ids"]),
        prompt_logprobs=_decode_logprobs(snapshot["prompt_logprobs"], flat),
        outputs=outputs,
        finished=all(output.finish_reason is not None for output in outputs),
    )
    if snapshot.get("prompt_token_id_logprobs") is not None:
        import numpy as np

        result.prompt_token_id_logprobs = np.asarray(
            snapshot["prompt_token_id_logprobs"]
        )
    return result


async def resume(engine, snapshot, internal_root):
    """Resume unfinished choices and yield cumulative RequestOutputs."""
    validate_snapshot(engine, snapshot)
    from vllm.sampling_params import RequestOutputKind

    current = _request_output(snapshot)
    if current.finished:
        yield current
        return
    events = asyncio.Queue()
    child_ids = child_request_ids(snapshot, internal_root)

    async def run_choice(choice, request_id):
        try:
            params = _decode_sampling(snapshot["sampling_params"])
            params.n = 1
            if params.seed is not None:
                params.seed += choice["index"]
            params.output_kind = RequestOutputKind.CUMULATIVE
            params.extra_args = dict(params.extra_args or {})
            params.extra_args[RESUME_KEY] = {
                "schema_version": SCHEMA_VERSION,
                "output_token_ids": list(choice["token_ids"]),
            }
            request, _ = await prepare_request(
                engine,
                {"prompt_token_ids": snapshot["prompt_token_ids"]},
                params,
                request_id,
            )
            # Validation/token preprocessing ran above, but source effective
            # generation defaults and derived stop IDs are authoritative.
            request.sampling_params = params
            for name, value in snapshot.get("request_options", {}).items():
                setattr(request, name, copy.deepcopy(value))
            engine._sllm_resume_adapter["pending"][request_id] = (
                snapshot,
                choice,
            )
            async for output in engine.generate(request, params, request_id):
                await events.put((choice["index"], output))
        except Exception as error:
            await events.put((choice["index"], error))
        finally:
            await events.put((choice["index"], None))

    tasks = [
        asyncio.create_task(run_choice(choice, child_id))
        for choice, child_id in zip(_unfinished(snapshot), child_ids)
    ]
    try:
        remaining = len(tasks)
        while remaining:
            index, output = await events.get()
            if output is None:
                remaining -= 1
                continue
            if isinstance(output, Exception):
                raise output
            completion = copy.deepcopy(output.outputs[0])
            completion.index = index
            current.outputs[index] = completion
            if output.prompt_logprobs is not None:
                current.prompt_logprobs = output.prompt_logprobs
            if getattr(output, "prompt_token_id_logprobs", None) is not None:
                current.prompt_token_id_logprobs = (
                    output.prompt_token_id_logprobs
                )
            current.finished = all(
                item.finish_reason is not None for item in current.outputs
            )
            yield copy.deepcopy(current)
        if not current.finished:
            raise RuntimeError(
                "Resumed generation ended without a terminal output"
            )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(engine.abort(child_id) for child_id in child_ids),
            return_exceptions=True,
        )
        for child_id in child_ids:
            forget_request(engine, child_id)
