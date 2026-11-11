"""Run the continuation qualification flow with CPU HTTP/actor substitutes."""

import asyncio
import builtins
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "baseline"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActorGone(Exception):
    pass


class Actor:
    def __init__(self, world, kind, **state):
        self.world, self.kind = world, kind
        self.__dict__.update(state)
        self.__ray_call__ = SimpleNamespace(remote=self.call)

    async def call(self, function, *args):
        world = self.world
        if self.kind == "source" and world.migrated:
            raise ActorGone("Source actor has retired")
        if function is world.base.actor_identity:
            return {
                "pid": 101 if self.kind == "source" else 202,
                "cgroup": "0::/cpu-continuation-test\n",
                "engine_model": "model",
            }
        if self.kind == "router" and world.migrated:
            world.cutover_observed.set()
        if function is world.module.backend_active and world.fail_observation:
            raise RuntimeError("Controlled observation failure")
        return function(self, *args)


class Flow:
    def __init__(self, module, base, evidence, output, mode="normal"):
        self.module, self.base, self.evidence = module, base, evidence
        self.output, self.mode = output, mode
        self.posts, self.requests, self.events = [], {}, []
        self.deallocated, self.cache_calls = [], []
        self.reservations = {}
        self.migrated = self.fail_observation = False
        self.all_active, self.release_http = asyncio.Event(), asyncio.Event()
        self.cutover_observed = asyncio.Event()
        self.allocation_started, self.late_allocation = (
            asyncio.Event(),
            asyncio.Event(),
        )
        self.pumps_drained = asyncio.Event()
        self.metrics = {"queries": 20.0, "hits": 10.0}
        self.source = Actor(
            self, "source", request_history={}, active_requests={}
        )
        self.target = Actor(
            self, "target", request_history={}, active_requests={}
        )
        self.target.generate = SimpleNamespace(remote=self.cache_probe)
        self.source_handle = SimpleNamespace(
            node_id="0", backend_instance=self.source, concurrency=0
        )
        self.target_handle = SimpleNamespace(
            node_id="1", backend_instance=self.target, concurrency=0
        )
        self.router = Actor(
            self,
            "router",
            ready_inference_instances={"source": self.source_handle},
            migration_record={},
            request_count=0,
            pending_allocations=set(),
            allocated_instances={},
            migration_requests={},
        )
        self.scheduler = Actor(
            self,
            "scheduler",
            model_instance={"model": {"source": "0"}},
            worker_nodes={"0": {"free_gpu": 0}, "1": {"free_gpu": 1}},
        )
        self.scheduler.allocate_resource = SimpleNamespace(remote=self.allocate)
        self.scheduler.deallocate_resource = SimpleNamespace(
            remote=self.deallocate
        )
        self.manager = Actor(
            self,
            "manager",
            model_info={},
            local_servers={"0": SimpleNamespace(disk_models={})},
        )
        self.counter_reader = SimpleNamespace(
            read_all_values_from_file=self.counter_rows
        )
        self.log_reads = 0

    @staticmethod
    def choice(index, count, reason=None):
        tokens = list(range(1000 + index * 1000, 1000 + index * 1000 + count))
        return {
            "index": index,
            "token_ids": tokens,
            "text": "".join(f"{token}," for token in tokens),
            "logprobs": [
                {
                    token: {
                        "logprob": -0.1,
                        "rank": 1,
                        "decoded_token": str(token),
                    }
                }
                for token in tokens
            ],
            "cumulative_logprob": -0.1 * count,
            "finish_reason": reason,
            "stop_reason": None,
        }

    def history(self, request, *, frozen=False):
        count = (
            (0 if self.mode == "zero-prefix" else 2)
            if frozen
            else request["max_tokens"]
        )
        return {
            "request_id": request["request_id"],
            "prompt_token_ids": list(request["input_tokens"]),
            "choices": [
                self.choice(index, count, None if frozen else "length")
                for index in range(request["n"])
            ],
        }

    @staticmethod
    def engine_output(history):
        choices = []
        for item in history["choices"]:
            item = copy.deepcopy(item)
            item["logprobs"] = [
                {
                    token: SimpleNamespace(**value)
                    for token, value in row.items()
                }
                for row in item["logprobs"]
            ]
            choices.append(SimpleNamespace(**item))
        return SimpleNamespace(
            prompt_token_ids=list(history["prompt_token_ids"]), outputs=choices
        )

    @staticmethod
    def response(request, history):
        completion = sum(len(item["token_ids"]) for item in history["choices"])
        prompt = len(history["prompt_token_ids"])
        body = {
            "id": request["request_id"],
            "model": request["model"],
            "object": "chat.completion",
            "choices": [
                {
                    "index": item["index"],
                    "message": {"role": "assistant", "content": item["text"]},
                    "finish_reason": item["finish_reason"],
                    "logprobs": item["logprobs"],
                }
                for item in history["choices"]
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
        # HTTP JSON, unlike the actor history, has string token-map keys.
        body = json.loads(json.dumps(body))
        return SimpleNamespace(
            status_code=200, json=lambda: copy.deepcopy(body)
        )

    async def post(self, url, *, json):
        assert url == "/v1/chat/completions"
        request = copy.deepcopy(json)
        request_id = request["request_id"]
        assert request_id not in self.posts, "The HTTP request was resent"
        self.posts.append(request_id)
        history = self.history(request)
        if request_id.endswith("-reference"):
            self.source.request_history[request_id] = self.engine_output(
                history
            )
            return self.response(request, history)
        self.requests[request_id] = request
        self.router.request_count += 1
        self.source_handle.concurrency += 1
        self.router.allocated_instances[request_id] = self.source_handle
        self.router.migration_requests[request_id] = request
        active = self.history(request, frozen=True)
        # An earlier observation can have output even when the supplied final
        # evidence is defective; the final evidence still has to reject it.
        if self.mode == "zero-prefix":
            active["choices"] = [
                self.choice(index, 1) for index in range(request["n"])
            ]
        self.source.active_requests[request_id] = {
            "output": self.engine_output(active)
        }
        if len(self.requests) == 3:
            self.all_active.set()
        try:
            await self.release_http.wait()
            self.target.request_history[request_id] = self.engine_output(
                history
            )
            return self.response(request, history)
        finally:
            self.source.active_requests.pop(request_id, None)
            owner = self.router.allocated_instances.pop(request_id)
            owner.concurrency -= 1
            self.router.migration_requests.pop(request_id)
            self.router.request_count -= 1
            if self.router.request_count == 0:
                self.pumps_drained.set()

    def snapshots(self):
        snapshots = []
        for request in self.requests.values():
            history = self.history(request, frozen=True)
            snapshots.append(
                {
                    **history,
                    "schema_version": 1,
                    "model": request["model"],
                    "identity": {
                        "artifact": "cpu-artifact",
                        "tokenizer": "cpu-tokenizer",
                    },
                    "prompt_logprobs": None,
                    "sampling_params": {
                        key: value
                        for key, value in request.items()
                        if key not in {"request_id", "model", "input_tokens"}
                    },
                }
            )
        return [] if self.mode == "empty-freeze" else snapshots

    async def allocate(self, probe, instance, resources):
        assert resources == {"num_gpus": 1}
        self.allocation_started.set()
        await self.all_active.wait()
        if self.mode == "late-allocation":
            self.fail_observation = True
            await self.late_allocation.wait()
            self.reservations[probe] = instance
            return "0"
        snapshots = self.snapshots()
        epoch = "cpu-epoch"
        for kind, pid in (("frozen", 101), ("prepared", 202)):
            if kind == "frozen" and self.mode == "missing-freeze":
                continue
            self.events.append(
                {
                    "schema_version": 1,
                    "event": kind,
                    "pid": pid,
                    "epoch": epoch,
                    "requests": copy.deepcopy(snapshots),
                }
            )
        self.events.extend(
            {
                "schema_version": 1,
                "event": "resume",
                "pid": 202,
                "epoch": epoch,
                "request_id": request_id,
            }
            for request_id in self.requests
        )
        self.router.ready_inference_instances = {"target": self.target_handle}
        self.router.migration_record["source"] = "target"
        self.target_handle.concurrency = self.source_handle.concurrency
        self.source_handle.concurrency = 0
        for request_id in self.requests:
            self.router.allocated_instances[request_id] = self.target_handle
        self.scheduler.model_instance["model"] = {"target": "1"}
        self.scheduler.worker_nodes = {
            "0": {"free_gpu": 1},
            "1": {"free_gpu": 0},
        }
        self.migrated = True
        self.reservations[probe] = instance
        await self.cutover_observed.wait()
        self.release_http.set()
        return "0"

    async def deallocate(self, probe, instance, resources):
        assert resources == {"num_gpus": 1}
        assert self.reservations.pop(probe) == instance
        self.deallocated.append((probe, instance))

    async def cache_probe(self, request):
        self.cache_calls.append(copy.deepcopy(request))
        self.metrics["queries"] += len(request["input_tokens"])
        self.metrics["hits"] += len(request["input_tokens"])
        return {"done": True}

    def counter_rows(self, filename):
        assert Path(filename).name == "counter_202.db"
        return [
            (
                json.dumps(
                    [
                        "metric",
                        "vllm:prefix_cache_" + key,
                        {
                            "model_name": "model",
                            "engine": "0",
                        },
                        "help",
                    ]
                ),
                value,
            )
            for key, value in self.metrics.items()
        ]

    def logs(self, root, _):
        self.log_reads += 1
        return "\n".join(
            self.evidence.MARKER + json.dumps(event) for event in self.events
        )

    async def run(self, scenario, result):
        return await self.module.exercise_scenario(
            scenario,
            {},
            self.output,
            result,
            self.output,
            self.output,
            "model",
            [10, 11],
            self,
            self.router,
            self.scheduler,
            self.manager,
            self.base,
            self.counter_reader,
            self.evidence.find_handoff,
            self.evidence.parse_events,
            self.evidence.validate_case,
        )


@pytest.fixture
def flow_factory(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module, base, evidence = (
        load("qualify_continuation"),
        load("qualify_migration"),
        load("continuation_evidence"),
    )
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "vllm", "transformers"}:
            raise AssertionError(
                "GPU/model dependency imported in CPU flow: " + name
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(exceptions=SimpleNamespace(RayActorError=ActorGone)),
    )
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::/cpu-continuation-test\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    real_sleep = asyncio.sleep

    async def flush_sleep(seconds):
        assert seconds == 10
        await real_sleep(0)

    # Replace this module's asyncio binding, never asyncio.sleep globally.
    local_asyncio = SimpleNamespace(**vars(asyncio))
    local_asyncio.sleep = flush_sleep
    module.asyncio = local_asyncio

    def create(mode="normal"):
        output = tmp_path / str(len(list(tmp_path.iterdir())))
        output.mkdir()
        (output / "counter_202.db").touch()
        flow = Flow(module, base, evidence, output, mode)
        monkeypatch.setattr(base, "read_run_logs", flow.logs)
        return flow

    return create


def result_state():
    return {"checks": {}, "cases": [], "scenarios": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,count", [("greedy", 1), ("seeded-multi", 2)])
async def test_real_flow_completes_original_http_requests_and_ledgers(
    flow_factory, scenario, count
):
    flow, result = flow_factory(), result_state()
    await asyncio.wait_for(flow.run(scenario, result), 5)
    assert len(flow.posts) == len(set(flow.posts)) == 4
    assert len(result["cases"]) == 3
    assert all(case["validated"] for case in result["cases"])
    assert all(
        case["validation"]["choice_count"] == count for case in result["cases"]
    )
    assert all(
        case["validation"]["full_tokens_equal"] for case in result["cases"]
    )
    assert all(
        case["validation"]["source_had_output"] for case in result["cases"]
    )
    scenario_evidence = result["scenarios"][0]
    assert len(scenario_evidence["attempts"]) == 3
    assert len(scenario_evidence["active_observations"]) == 3
    assert flow.log_reads == 1
    assert flow.cache_calls == [
        {
            "input_tokens": [10, 11, 1000, 1001],
            "max_tokens": 1,
            "temperature": 0,
        }
    ]
    assert scenario_evidence["target_cache"]["delta"] == {
        "queries": 4,
        "hits": 4,
    }
    assert scenario_evidence["settled_router"]["request_count"] == 0
    assert (
        flow.router.allocated_instances == flow.router.migration_requests == {}
    )
    assert flow.target_handle.concurrency == 0
    assert len(flow.deallocated) == 1 and not flow.reservations
    saved = json.loads((flow.output / "continuation-result.json").read_text())
    assert saved == result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing-freeze", "empty-freeze"])
async def test_no_final_frozen_evidence_cannot_pass_flow(flow_factory, mode):
    flow, result = flow_factory(mode), result_state()
    with pytest.raises(
        (RuntimeError, ValueError), match="freeze|handoff|resume"
    ):
        await asyncio.wait_for(flow.run("greedy", result), 5)
    assert not result["cases"]
    assert not result["checks"].get("same_http_continuation")
    assert len(flow.deallocated) == 1 and not flow.reservations
    assert flow.router.request_count == flow.target_handle.concurrency == 0


@pytest.mark.asyncio
async def test_zero_generated_prefix_cannot_qualify_overall(flow_factory):
    result = result_state()
    for scenario in ("greedy", "seeded-multi"):
        flow = flow_factory("zero-prefix")
        await asyncio.wait_for(flow.run(scenario, result), 5)
    assert len(result["cases"]) == 6
    assert all(
        not case["validation"]["source_had_output"] for case in result["cases"]
    )
    # Supply only the unrelated outer-runner prerequisites, so the actual
    # runtime gate must reject the weak cases produced by exercise_scenario.
    result["checks"].update(
        dict.fromkeys(
            (
                "private_store",
                "worker_mapping",
                "http_baseline",
                "controller_shutdown",
            ),
            True,
        )
    )
    result.update(
        cleanup_completed=True, cleanup_errors=[], transparent_continuation=True
    )
    assert not flow.module.runtime_passed(result)


@pytest.mark.asyncio
async def test_failure_observes_and_releases_late_allocation(flow_factory):
    flow, result = flow_factory("late-allocation"), result_state()
    task = asyncio.create_task(flow.run("greedy", result))
    try:
        await asyncio.wait_for(flow.allocation_started.wait(), 2)
        await asyncio.wait_for(flow.pumps_drained.wait(), 2)
        assert not task.done(), "Cleanup abandoned the pending allocation"
        assert not flow.reservations
        flow.late_allocation.set()
        with pytest.raises(
            RuntimeError, match="Controlled observation failure"
        ):
            await asyncio.wait_for(task, 3)
        assert len(flow.deallocated) == 1 and not flow.reservations
        assert flow.router.request_count == flow.source_handle.concurrency == 0
        assert not flow.router.allocated_instances
        assert not result["cases"]
        assert (flow.output / "continuation-result.json").is_file()
    finally:
        flow.late_allocation.set()
        await asyncio.gather(task, return_exceptions=True)
