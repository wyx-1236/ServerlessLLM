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
from sllm.backends.backend_utils import BackendStatus, SllmBackend

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


class VllmBackend(SllmBackend):
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

    async def _admit(self, request_id, input_tokens):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return "Engine is not running"
            if request_id in self.active_requests:
                return f"Duplicate request id: {request_id}"
            self.active_requests[request_id] = {
                "tokens": list(input_tokens or []),
                "output": None,
                "task": asyncio.current_task(),
                "terminal": False,
                "aborting": False,
            }
            self._drained.clear()
        return None

    async def _abort_request(self, request_id):
        async with self.status_lock:
            record = self.active_requests.get(request_id)
            if record is None or record["aborting"]:
                return
            record["aborting"] = True
            engine = self.engine
        try:
            await vllm_compat.abort(engine, request_id)
        except Exception:
            logger.exception("Engine abort failed for request %s", request_id)

    async def _finish(self, request_id):
        async with self.status_lock:
            record = self.active_requests.pop(request_id)
            record["terminal"] = True
            if self.trace_debug:
                self.request_history[request_id] = record["output"]
            if not self.active_requests:
                self._drained.set()

    async def generate(self, request_data):
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
        try:
            sampling_params = SamplingParams(**data)
        except Exception as error:
            return {"error": f"Invalid sampling parameters: {error}"}
        error = await self._admit(request_id, input_tokens)
        if error:
            return {"error": error}
        final_output = None
        try:
            async for output in self.engine.generate(
                inputs, sampling_params, request_id
            ):
                final_output = output
                async with self.status_lock:
                    self.active_requests[request_id]["output"] = output
            if final_output is None:
                return {"error": "Engine returned no output"}
            return process_output(final_output, model_name)
        except asyncio.CancelledError:
            await self._abort_request(request_id)
            raise
        except Exception as error:
            await self._abort_request(request_id)
            return {"error": str(error)}
        finally:
            await self._finish(request_id)

    async def _close(self):
        await self._drained.wait()
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
            # Abort should terminate the engine iterator; cancellation also makes
            # bounded progress when a failed engine no longer yields a terminal.
            for task in requests.values():
                if task is not None and not task.done():
                    task.cancel()

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
                self._abort_task = asyncio.create_task(
                    self._abort_active(requests)
                )
            close_task, abort_task = self._close_task, self._abort_task
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
        async with self.status_lock:
            snapshots = []
            for record in self.active_requests.values():
                output = record["output"]
                tokens = list(record["tokens"])
                if output is not None:
                    tokens = list(output.prompt_token_ids or [])
                    if isinstance(output.outputs, list) and output.outputs:
                        tokens += list(output.outputs[0].token_ids)
                snapshots.append(tokens)
            return snapshots

    async def resume_kv_cache(self, request_datas):
        await asyncio.gather(
            *(
                self.generate({"input_tokens": tokens, "max_tokens": 1})
                for tokens in request_datas
            )
        )

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
