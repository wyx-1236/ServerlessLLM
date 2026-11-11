"""Owned request handoff for the fixed vLLM backend.

The caller stays admitted while its engine iterator is frozen. Only a router
commit transfers the response; ordinary cancellation never resumes a request.
"""

import asyncio
import copy
import hashlib
import json
import logging
import os

from sllm.backends import vllm_resume_adapter as resume_adapter
from sllm.backends.backend_utils import BackendStatus


def merge_output(previous, output, *, freezing=False, delta=False):
    """vLLM cumulative updates may contain only some parallel choices."""
    result = copy.copy(output)
    choices = {
        getattr(choice, "index", index): copy.copy(choice)
        for index, choice in enumerate(getattr(previous, "outputs", []))
    }
    for index, choice in enumerate(output.outputs):
        choice = copy.deepcopy(choice)
        choice_index = getattr(choice, "index", index)
        old = choices.get(choice_index)
        if delta and old is not None:
            choice.token_ids = list(old.token_ids) + list(choice.token_ids)
            choice.text = old.text + choice.text
            if old.logprobs is not None:
                rows = copy.deepcopy(old.logprobs)
                rows.extend(choice.logprobs or [])
                choice.logprobs = rows
        if freezing and choice.finish_reason == "abort":
            choice.finish_reason = None
            choice.stop_reason = None
        choices[choice_index] = choice
    result.outputs = [choices[index] for index in sorted(choices)]
    if (
        getattr(result, "prompt_logprobs", None) is None
        and previous is not None
    ):
        result.prompt_logprobs = getattr(previous, "prompt_logprobs", None)
    return result


def naturally_finished(record):
    output = record["output"]
    if output is None:
        return False
    expected = (
        (record.get("metadata") or {}).get("sampling_params", {}).get("n", 1)
    )
    return len(output.outputs) == expected and all(
        choice.finish_reason in {"stop", "length"} for choice in output.outputs
    )


class VllmMigration:
    def _trace_migration(self, event, epoch, **data):
        if self.trace_debug:
            logging.getLogger("ray").info(
                "SLLM migration evidence: %s",
                json.dumps(
                    {
                        "schema_version": 1,
                        "event": event,
                        "pid": os.getpid(),
                        "epoch": epoch,
                        **data,
                    }
                ),
            )

    def _init_migration(self, model):
        self._logical_model = model
        self._migration_condition = asyncio.Condition(self.status_lock)
        self._migration = None
        self._migration_epochs = {}
        self._migration_prepared = {}
        self._migration_preparing = {}
        self._migration_cancelled = set()
        self._migration_tracked = set()
        self._migration_forget_pending = set()
        self._request_terminal = set()
        self._identity = None

    def _migration_identity(self):
        if self._identity is not None:
            return copy.deepcopy(self._identity)
        extra = (
            getattr(self.engine_args, "model_loader_extra_config", None) or {}
        )
        identity = {
            "model": self._logical_model,
            "artifact": extra.get("identity"),
        }
        if identity["artifact"] is None:
            identity["source"] = {
                key: getattr(self.engine_args, key, None)
                for key in ("model", "revision", "dtype", "load_format")
            }
        tokenizer = self.engine.get_tokenizer()
        native = getattr(tokenizer, "backend_tokenizer", None)
        if native is not None and hasattr(native, "to_str"):
            tokenizer_state = native.to_str()
        else:
            # This is deliberately bounded to immutable tokenizer properties;
            # model paths on different workers are not an identity.
            tokenizer_state = json.dumps(
                {
                    "class": type(tokenizer).__qualname__,
                    "vocab": tokenizer.get_vocab(),
                    "special_tokens": getattr(
                        tokenizer, "all_special_tokens", []
                    ),
                    "special_ids": getattr(tokenizer, "all_special_ids", []),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        identity["tokenizer"] = hashlib.sha256(
            tokenizer_state.encode("utf-8")
        ).hexdigest()
        self._identity = identity
        return copy.deepcopy(identity)

    def _snapshot(self, record):
        if record.get("metadata") is None:
            raise ValueError("Only token generation requests support migration")
        if record["output"] is None and "snapshot" in record:
            return copy.deepcopy(record["snapshot"])
        return resume_adapter.snapshot_request(
            record["metadata"],
            record["output"],
            model=record["model_name"],
            identity=self._migration_identity(),
        )

    async def _mark_terminal(self, request_id):
        async with self._migration_condition:
            self._remember_terminal(request_id)
            self._migration_condition.notify_all()

    def _remember_terminal(self, request_id):
        if request_id in self._migration_tracked or (
            self._migration is not None
            and request_id in self._migration["expected"]
        ):
            self._request_terminal.add(request_id)

    async def _collect_generation(self, request_id, iterator):
        async for output in iterator:
            async with self.status_lock:
                record = self.active_requests[request_id]
                if record.get("migration") is not None:
                    output = resume_adapter.normalize_abort_output(
                        output, record["metadata"]
                    )
                record["output"] = merge_output(
                    record["output"],
                    output,
                    freezing=record.get("migration") is not None,
                    delta=(
                        not record.get("resuming", False)
                        and (record.get("metadata") or {})
                        .get("sampling_params", {})
                        .get("output_kind")
                        == 1
                    ),
                )

    async def _run_generation(
        self, request_id, iterator_factory, process_output
    ):
        """Run or freeze one iterator, retaining the original response owner."""
        while True:
            async with self.status_lock:
                record = self.active_requests[request_id]
                context = record.get("migration")
                # Expected late admissions freeze before sending anything to
                # EngineCore. Their exact prompt/params are already normalized.
                if context is None:
                    record["run_task"] = asyncio.create_task(
                        self._collect_generation(request_id, iterator_factory())
                    )
                run_task = record["run_task"]
            if run_task is not None:
                await asyncio.shield(run_task)
            async with self.status_lock:
                context = record.get("migration")
                if context is None or naturally_finished(record):
                    output = record["output"]
                    if output is None:
                        return {"error": "Engine returned no output"}
                    return process_output(output, record["model_name"])
                record["paused"].set()
            decision = await asyncio.shield(context["decision"])
            if decision == "commit":
                return {
                    "_sllm_migration": {
                        "epoch": context["epoch"],
                        "request_id": request_id,
                    }
                }
            async with self.status_lock:
                if request_id in self._migration_cancelled:
                    raise asyncio.CancelledError
                if record["run_task"] is None:
                    # The router froze this late admission before its original
                    # iterator started. Rollback simply opens that iterator.
                    if record.get("migration") is context:
                        record["migration"] = None
                    continue
                snapshot = record["snapshot"]
                internal_root = f"{request_id}:rollback:{context['epoch']}"
                record["engine_ids"] = resume_adapter.child_request_ids(
                    snapshot, internal_root
                )
                record["run_task"] = None
                if record.get("migration") is context:
                    record["migration"] = None
                record["aborting"] = False
                record["resuming"] = True
                record["paused"].clear()

            def iterator_factory():
                return resume_adapter.resume(
                    self.engine, snapshot, internal_root
                )

    async def _freeze_iterator(self, record):
        task = record["run_task"]
        if task is None or task.done():
            return

        # Frontend registration alone is too early: a late EngineCore add
        # after abort could resurrect a frozen request. Wait for the actual
        # add RPC, or an iterator failure, before issuing the abort.
        async def registered():
            await asyncio.gather(
                *(
                    resume_adapter.wait_registered(self.engine, request_id)
                    for request_id in record["engine_ids"]
                )
            )

        registration = asyncio.create_task(registered())
        try:
            await asyncio.wait(
                (registration, task), return_when=asyncio.FIRST_COMPLETED
            )
            if not task.done():
                await registration
                await asyncio.gather(
                    *(
                        self.engine.abort(request_id)
                        for request_id in record["engine_ids"]
                    )
                )
            # Consume the final frontend output, including choices that have
            # naturally finished while freeze was being established.
            await asyncio.shield(task)
        finally:
            registration.cancel()
            await asyncio.gather(registration, return_exceptions=True)

    async def freeze_migration(self, epoch, request_ids=None):
        resume_adapter.validate_engine(self.engine)
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("Migration epoch must be a nonempty string")
        async with self._migration_condition:
            prior = self._migration_epochs.get(epoch)
            if prior is not None:
                task = prior["freeze_task"]
            else:
                if (
                    self.status != BackendStatus.RUNNING
                    or self._migration is not None
                ):
                    raise RuntimeError(
                        "Instance cannot begin another migration"
                    )
                # Validate every currently admitted request before aborting any.
                for record in self.active_requests.values():
                    resume_adapter.validate_snapshot(
                        self.engine, self._snapshot(record)
                    )
                context = {
                    "epoch": epoch,
                    "expected": set(request_ids or ())
                    | set(self.active_requests),
                    "decision": asyncio.get_running_loop().create_future(),
                    "phase": "freezing",
                }
                self._migration = context
                self._migration_epochs[epoch] = context
                task = context["freeze_task"] = asyncio.create_task(
                    self._freeze(context)
                )
            self._migration_condition.notify_all()
        return copy.deepcopy(await asyncio.shield(task))

    async def _freeze(self, context):
        async with self._migration_condition:
            await self._migration_condition.wait_for(
                lambda: context["expected"]
                <= (
                    set(self.active_requests)
                    | self._request_terminal
                    | self._migration_cancelled
                )
                or self.status == BackendStatus.DELETING
            )
            if self.status == BackendStatus.DELETING:
                raise RuntimeError("Instance shut down while freezing")
            records = {
                request_id: self.active_requests[request_id]
                for request_id in context["expected"]
                if request_id in self.active_requests
                and request_id not in self._migration_cancelled
            }
            for record in records.values():
                resume_adapter.validate_snapshot(
                    self.engine, self._snapshot(record)
                )
            for record in records.values():
                record["migration"] = context
        results = await asyncio.gather(
            *(self._freeze_iterator(record) for record in records.values()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                raise result
        async with self.status_lock:
            snapshots = []
            for request_id, record in records.items():
                if (
                    request_id not in self.active_requests
                    or request_id in self._migration_cancelled
                    or naturally_finished(record)
                    or (
                        record["run_task"] is not None
                        and record["run_task"].cancelled()
                    )
                ):
                    continue
                snapshot = self._snapshot(record)
                resume_adapter.validate_snapshot(self.engine, snapshot)
                record["snapshot"] = snapshot
                snapshots.append(snapshot)
            context["phase"] = "frozen"
            self._trace_migration(
                "frozen", context["epoch"], requests=snapshots
            )
            return {
                "schema_version": 1,
                "epoch": context["epoch"],
                "requests": snapshots,
            }

    async def commit_migration(self, epoch):
        context = self._migration_epochs.get(epoch)
        if context is None:
            raise ValueError("Unknown migration epoch")
        await asyncio.shield(context["freeze_task"])
        async with self.status_lock:
            if context["phase"] == "rolled_back":
                raise ValueError("Cannot commit a rolled back migration")
            context["phase"] = "committed"
            if not context["decision"].done():
                context["decision"].set_result("commit")
            self._flush_forgotten_requests()

    async def rollback_migration(self, epoch):
        context = self._migration_epochs.get(epoch)
        if context is None:
            return
        # Observe freeze even when its RPC caller was cancelled. A failed
        # freeze can still have paused some iterators.
        await asyncio.gather(
            asyncio.shield(context["freeze_task"]), return_exceptions=True
        )
        async with self._migration_condition:
            if context["phase"] == "committed":
                raise ValueError("Cannot roll back a committed migration")
            if context["phase"] == "rolled_back":
                return
            for record in self.active_requests.values():
                if record.get("migration") is not context:
                    continue
                task = record["run_task"]
                if task is not None and not task.done():
                    # A failed abort left this original iterator running.
                    record["migration"] = None
                elif task is not None:
                    record["snapshot"] = self._snapshot(record)
            context["phase"] = "rolled_back"
            self._migration = None
            if not context["decision"].done():
                context["decision"].set_result("rollback")
            self._flush_forgotten_requests()
            self._migration_condition.notify_all()

    async def prepare_migration(self, epoch, envelopes):
        resume_adapter.validate_engine(self.engine)
        identity = self._migration_identity()
        snapshots = copy.deepcopy(envelopes)
        ids = [snapshot["request_id"] for snapshot in snapshots]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate migration request id")
        for snapshot in snapshots:
            if snapshot["identity"] != identity:
                raise ValueError("Migration model/tokenizer identity mismatch")
            resume_adapter.validate_snapshot(self.engine, snapshot)
        async with self.status_lock:
            if (
                self.status != BackendStatus.RUNNING
                or self._migration is not None
            ):
                raise RuntimeError("Target is not accepting migration")
            existing = self._migration_prepared.get(epoch)
            if existing is not None:
                if [
                    entry["snapshot"] for entry in existing.values()
                ] != snapshots:
                    raise ValueError("Conflicting migration preparation")
                task = self._migration_preparing[epoch]
            else:
                if any(
                    request_id in self.active_requests for request_id in ids
                ):
                    raise ValueError("Migration request is already active")
                self._migration_prepared[epoch] = {
                    snapshot["request_id"]: {"snapshot": snapshot, "task": None}
                    for snapshot in snapshots
                }
                task = self._migration_preparing[epoch] = asyncio.create_task(
                    self._prewarm_migration(epoch, snapshots)
                )
        await asyncio.shield(task)
        return ids

    async def _prewarm_migration(self, epoch, snapshots):
        # Preserve the iterative prewarm, and add the final frozen suffixes.
        # These speculative single tokens never enter the user snapshot.
        try:
            await self.resume_kv_cache(
                [
                    snapshot["prompt_token_ids"] + choice["token_ids"]
                    for snapshot in snapshots
                    for choice in snapshot["choices"]
                    if choice["finish_reason"] is None
                ]
            )
        except BaseException:
            async with self.status_lock:
                self._migration_prepared.pop(epoch, None)
                self._migration_preparing.pop(epoch, None)
            raise
        self._trace_migration("prepared", epoch, requests=snapshots)

    async def resume_migration(self, epoch, request_id):
        preparation = self._migration_preparing.get(epoch)
        if preparation is not None:
            await asyncio.shield(preparation)
        async with self.status_lock:
            self._migration_tracked.add(request_id)
            if request_id in self._migration_cancelled:
                raise asyncio.CancelledError
            entry = self._migration_prepared.get(epoch, {}).get(request_id)
            if entry is None:
                raise ValueError("Request was not prepared for this epoch")
            if entry["task"] is None:
                entry["task"] = asyncio.create_task(
                    self._resume_prepared(epoch, entry["snapshot"])
                )
            task = entry["task"]
        return await asyncio.shield(task)

    async def _resume_prepared(self, epoch, snapshot):
        from sllm.backends.vllm_backend import process_output

        request_id = snapshot["request_id"]
        metadata = {
            key: copy.deepcopy(snapshot.get(key))
            for key in (
                "request_id",
                "prompt",
                "prompt_token_ids",
                "sampling_params",
                "request_options",
            )
        }
        metadata["request_options"] = copy.deepcopy(
            snapshot.get("request_options") or {}
        )
        error = await self._admit(
            request_id,
            snapshot["prompt_token_ids"],
            metadata=metadata,
            model_name=snapshot["model"],
        )
        if error:
            await self._mark_terminal(request_id)
            return {"error": error}
        internal_root = f"{request_id}:resume:{epoch}"
        try:
            async with self.status_lock:
                record = self.active_requests[request_id]
                record["snapshot"] = snapshot
                record["resuming"] = True
                record["engine_ids"] = resume_adapter.child_request_ids(
                    snapshot, internal_root
                )
                self._trace_migration("resume", epoch, request_id=request_id)
            return await self._run_generation(
                request_id,
                lambda: resume_adapter.resume(
                    self.engine, snapshot, internal_root
                ),
                process_output,
            )
        except asyncio.CancelledError:
            await self._abort_request(request_id)
            raise
        except Exception as error:
            await self._abort_request(request_id)
            return {"error": str(error)}
        finally:
            await self._finish(request_id)

    async def cancel_migration_request(self, epoch, request_id):
        async with self._migration_condition:
            self._migration_cancelled.add(request_id)
            record = self.active_requests.get(request_id)
            task = record["task"] if record is not None else None
            self._migration_condition.notify_all()
        if record is not None:
            await self._abort_request(request_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def forget_migration_request(self, request_id):
        """Router calls only after all dispatched RPCs have actually settled."""
        async with self.status_lock:
            if request_id in self.active_requests:
                return
            context = self._migration
            if (
                context is not None
                and context["phase"] in {"freezing", "frozen"}
                and request_id in context["expected"]
            ):
                self._migration_forget_pending.add(request_id)
            else:
                self._forget_request(request_id)

    def _forget_request(self, request_id):
        self._request_terminal.discard(request_id)
        self._migration_tracked.discard(request_id)
        self._migration_cancelled.discard(request_id)
        for entries in self._migration_prepared.values():
            entries.pop(request_id, None)

    def _flush_forgotten_requests(self):
        for request_id in self._migration_forget_pending:
            self._forget_request(request_id)
        self._migration_forget_pending.clear()
