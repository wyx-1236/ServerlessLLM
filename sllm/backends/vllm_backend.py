# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import time
import uuid
from itertools import count
from typing import Any, Dict, List, Optional

from vllm import (
    EmbeddingRequestOutput,
    PoolingParams,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt

from sllm.backends import vllm_compat
from sllm.backends import vllm_resume_adapter as resume_adapter
from sllm.backends.backend_utils import BackendStatus, SllmBackend
from sllm.backends.vllm_migration import VllmMigration
from sllm.backends.vllm_qualification import NumericalQualificationMixin

logger = logging.getLogger("ray")


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


def _request_tokens(output, input_tokens=None):
    tokens = list(input_tokens or [])
    if output is not None:
        tokens = list(output.prompt_token_ids or [])
        if isinstance(output.outputs, list) and output.outputs:
            tokens += list(output.outputs[0].token_ids)
    return tokens


class LLMEngineStatusDict:
    """Compatibility view of active requests, separate from debug history."""

    def __init__(self, active, lock):
        self.status_dict = active
        self.lock = lock

    async def return_all_results(self):
        async with self.lock:
            return [record["output"] for record in self.status_dict.values()]

    async def return_all_request_ids(self):
        async with self.lock:
            return list(self.status_dict)

    async def request_count(self):
        async with self.lock:
            return len(self.status_dict)


class VllmBackend(NumericalQualificationMixin, VllmMigration, SllmBackend):
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ):
        if backend_config is None:
            raise ValueError("Backend config is missing")
        self.status = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = dict(backend_config)
        self.trace_debug = backend_config.get("trace_debug", False)
        self.engine_args = vllm_compat.engine_args(model, backend_config)
        self.engine = None
        self.active_requests = {}
        self.request_history = {}
        self.request_trace = LLMEngineStatusDict(
            self.active_requests, self.status_lock
        )
        self._drained = asyncio.Event()
        self._drained.set()
        self._close_task = None
        self._abort_task = None
        self._request_counter = count()
        self._init_migration(model)

    async def init_backend(self):
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            try:
                self.engine = vllm_compat.create(self.engine_args)
            except BaseException:
                await vllm_compat.close(self.engine)
                self.engine = None
                raise
            self.status = BackendStatus.RUNNING

    async def _admit(
        self, request_id, input_tokens, *, metadata=None, model_name=None
    ):
        async with self._migration_condition:
            if self.status != BackendStatus.RUNNING:
                return "Engine is not running"
            if request_id in self._migration_cancelled:
                return "Request cancelled"
            context = self._migration
            if context is not None and (
                context["phase"] != "freezing"
                or request_id not in context["expected"]
            ):
                return "Engine is frozen for migration"
            if request_id in self.active_requests:
                return f"Duplicate request id: {request_id}"
            self.active_requests[request_id] = {
                "tokens": list(input_tokens or []),
                "output": None,
                "task": asyncio.current_task(),
                "terminal": False,
                "aborting": False,
                "metadata": metadata,
                "model_name": model_name,
                "engine_ids": [request_id],
                "run_task": None,
                "paused": asyncio.Event(),
                "migration": context,
            }
            self._drained.clear()
            self._migration_condition.notify_all()
        return None

    async def _abort_request(self, request_id):
        async with self.status_lock:
            record = self.active_requests.get(request_id)
            if record is None or record["aborting"]:
                return
            record["aborting"] = True
            engine = self.engine
            engine_ids = list(record["engine_ids"])
            run_task = record["run_task"]
        try:
            # User cancellation discards unobserved output. Migration uses
            # _freeze_iterator instead and explicitly drains the abort output.
            if run_task is not None and not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
            await asyncio.gather(
                *(
                    vllm_compat.abort(engine, engine_id)
                    for engine_id in engine_ids
                )
            )
        except Exception:
            logger.exception("Engine abort failed for request %s", request_id)

    async def _finish(self, request_id):
        async with self._migration_condition:
            record = self.active_requests.pop(request_id)
            record["terminal"] = True
            self._remember_terminal(request_id)
            if self.trace_debug:
                self.request_history[request_id] = record["output"]
            if not self.active_requests:
                self._drained.set()
            self._migration_condition.notify_all()
        if hasattr(self.engine, "_sllm_resume_adapter"):
            for engine_id in record["engine_ids"]:
                resume_adapter.forget_request(self.engine, engine_id)

    async def generate(self, request_data, *, migration_tracked=False):
        if request_data is None:
            return {"error": "Request data is missing"}
        data = dict(request_data)
        model_name = data.pop("model", "vllm-model")
        messages = data.pop("messages", [])
        prompt = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
            if "content" in message
        )
        inputs = data.pop("prompt", prompt)
        input_tokens = data.pop("input_tokens", None)
        if input_tokens is not None:
            inputs = TokensPrompt(prompt_token_ids=list(input_tokens))
        request_id = data.pop("request_id", f"chatcmpl-{uuid.uuid4()}")
        if migration_tracked:
            async with self.status_lock:
                self._migration_tracked.add(request_id)
        try:
            sampling_params = SamplingParams(**data)
        except Exception as error:
            await self._mark_terminal(request_id)
            return {"error": f"Invalid sampling parameters: {error}"}
        metadata = None
        try:
            if hasattr(self.engine, "_sllm_resume_adapter"):
                inputs, metadata = await resume_adapter.prepare_request(
                    self.engine, inputs, sampling_params, request_id
                )
                input_tokens = metadata["prompt_token_ids"]
        except asyncio.CancelledError:
            await self._mark_terminal(request_id)
            raise
        except Exception as error:
            await self._mark_terminal(request_id)
            return {"error": str(error)}
        error = await self._admit(
            request_id, input_tokens, metadata=metadata, model_name=model_name
        )
        if error:
            await self._mark_terminal(request_id)
            return {"error": error}
        try:
            return await self._run_generation(
                request_id,
                lambda: self.engine.generate(
                    inputs, sampling_params, request_id
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

    async def _close(self):
        owned = self._owned_migration_tasks()
        if owned:
            await asyncio.gather(*owned, return_exceptions=True)
        await self._drained.wait()
        if self._abort_task is not None:
            await asyncio.shield(self._abort_task)
        try:
            await vllm_compat.close(self.engine)
        finally:
            self.engine = None
            async with self.status_lock:
                self.status = BackendStatus.DELETING

    async def _abort_active(self, requests):
        try:
            await asyncio.gather(
                *(self._abort_request(request_id) for request_id in requests)
            )
        finally:
            # Abort terminates the iterator; cancellation also makes bounded
            # progress when a failed engine no longer yields a terminal.
            for task in requests.values():
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in requests.values() if task is not None),
                return_exceptions=True,
            )

    def _owned_migration_tasks(self):
        tasks = set(self._migration_preparing.values())
        tasks.update(
            entry["task"]
            for entries in self._migration_prepared.values()
            for entry in entries.values()
            if entry["task"] is not None
        )
        return [task for task in tasks if not task.done()]

    async def _begin_close(self, force):
        async with self.status_lock:
            if self._close_task is None:
                self.status = (
                    BackendStatus.DELETING if force else BackendStatus.STOPPING
                )
                self._close_task = asyncio.create_task(self._close())
            if force and self._abort_task is None:
                self.status = BackendStatus.DELETING
                requests = {
                    key: value["task"]
                    for key, value in self.active_requests.items()
                }
                for index, task in enumerate(self._owned_migration_tasks()):
                    requests[("internal-migration", index)] = task
                self._abort_task = asyncio.create_task(
                    self._abort_active(requests)
                )
            close_task, abort_task = self._close_task, self._abort_task
            self._migration_condition.notify_all()
        try:
            if abort_task is not None:
                await asyncio.shield(abort_task)
        finally:
            await asyncio.shield(close_task)

    async def stop(self):
        """Close admission, drain admitted work, and join one engine close."""
        await self._begin_close(False)

    async def shutdown(self):
        """Escalate graceful draining to abort and join the same close task."""
        await self._begin_close(True)

    async def get_current_tokens(self):
        """Copy active request tokens for migration, excluding debug history."""
        async with self.status_lock:
            return [
                _request_tokens(record["output"], record["tokens"])
                for record in self.active_requests.values()
            ]

    async def get_debug_tokens(self) -> Dict[str, List[int]]:
        """Copy finalized trace_debug token history, keyed by request ID.

        Values contain prompt tokens and the first choice's observed output.
        Requests ending before any output have an empty list. Retained records
        include failed/cancelled requests and do not imply successful
        completion.
        """
        async with self.status_lock:
            return {
                request_id: _request_tokens(output)
                for request_id, output in self.request_history.items()
            }

    async def resume_kv_cache(self, request_datas):
        results = await asyncio.gather(
            *(
                self.generate({"input_tokens": tokens, "max_tokens": 1})
                for tokens in request_datas
            )
        )
        for result in results:
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"KV cache replay failed: {result['error']}")

    async def encode(self, request_data):
        if not request_data:
            return {"error": "Request data is missing"}
        if (
            self.backend_config.get("task", "auto") == "generate"
            or getattr(self.engine_args, "runner", "auto") == "generate"
            or not hasattr(self.engine, "encode")
        ):
            return {"error": "Model is not configured for pooling/embedding"}
        data = dict(request_data)
        query = data.get("input", [])
        if not query:
            return {"error": "No inputs provided"}
        if isinstance(query, str) or (
            isinstance(query, list) and isinstance(query[0], int)
        ):
            query = [query]

        async def process_input(inputs):
            request_id = f"embed-{uuid.uuid4()}-{next(self._request_counter)}"
            tokens = inputs if isinstance(inputs, list) else []
            error = await self._admit(request_id, tokens)
            if error:
                return {"error": error}
            try:
                output = None
                if isinstance(inputs, list) and all(
                    isinstance(token, int) for token in inputs
                ):
                    inputs = TokensPrompt(prompt_token_ids=list(inputs))
                async for output in self.engine.encode(
                    inputs, PoolingParams(), request_id
                ):
                    async with self.status_lock:
                        self.active_requests[request_id]["output"] = output
                if output is not None and not hasattr(
                    output.outputs, "embedding"
                ):
                    output = EmbeddingRequestOutput.from_base(output)
                return output
            except asyncio.CancelledError:
                await self._abort_request(request_id)
                raise
            except Exception as error:
                await self._abort_request(request_id)
                return {"error": f"Pooling/embedding request failed: {error}"}
            finally:
                await self._finish(request_id)

        outputs = await asyncio.gather(*(process_input(item) for item in query))
        valid = [
            item
            for item in outputs
            if item is not None and not isinstance(item, dict)
        ]
        if not valid:
            return next(
                (item for item in outputs if isinstance(item, dict)),
                {"error": "All inputs failed"},
            )
        if any(not hasattr(item.outputs, "embedding") for item in valid):
            return {"error": "Pooling model does not provide embedding outputs"}
        return process_embedding_output(valid, data.get("model", "vllm-model"))
