"""Separately approved EOS and cross-freeze stop-string qualification."""

import json
import sys
from pathlib import Path

import qualify_continuation as continuation
import stopping_evidence as evidence

SCENARIOS = evidence.SCENARIOS
SCOPE = "stage-d2-stopping"
RESULT_NAME = "stopping-result.json"
LIMITATIONS = (
    "single-host Qwen3-0.6B BF16 TP1 eager V2 FLASH_ATTN batch-invariant-v1",
    "greedy single-choice EOS and stop strings only; no default numerical profile qualification",
    "no rollback/cancellation, repeated request migration, Transformers or TP2 GPU qualification",
    "external runner owns complete evidence archives and resource release proof",
)


def validate_approval(config, environ, now, cgroup):
    from continuation_profile import validate_environment
    from qualify_migration import validate_approval as validate

    validate(config, environ, now, cgroup, scope=SCOPE, run_prefix="d2s")
    evidence.require(
        config.get("numerical_profile") == "batch-invariant-v1",
        "Stopping qualification requires explicit batch-invariant-v1",
    )
    validate_environment(config, environ)


def runtime_passed(result):
    try:
        if (
            result.get("suite") != SCOPE
            or result["approval"]["scope"] != SCOPE
            or result["approval"].get("numerical_profile")
            != "batch-invariant-v1"
            or not continuation.runtime_passed(result, SCENARIOS)
        ):
            return False
        scenarios = result["scenarios"]
        if len(scenarios) != len(SCENARIOS) or {
            s["scenario"] for s in scenarios
        } != set(SCENARIOS):
            return False
        case_ids = [(c["scenario"], c["request_id"]) for c in result["cases"]]
        if len(case_ids) != len(set(case_ids)):
            return False
        expected = set()
        for scenario in scenarios:
            if scenario.get("preparation_error") or scenario.get(
                "stopping_error"
            ):
                return False
            audit = evidence.audit_scenario(scenario)
            if audit != scenario["stopping_validation"]:
                return False
            expected.update(
                (scenario["scenario"], r["request_id"])
                for r in scenario["frozen"]["requests"]
            )
        return expected == set(case_ids)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


async def prepare_reference(
    scenario, model, tokens, tokenizer, config, source, client, record, save
):
    from continuation_evidence import _canonical, _response

    state = record["stopping"] = {"probes": [], "decoded_prefixes": {}}
    save()

    def decode(ids):
        return tokenizer.decode(ids, **evidence.DECODE_OPTIONS)

    async def request(body, request_id, receipt):
        receipt["request"] = {**body, "request_id": request_id}
        save()
        try:
            response = await continuation.send_once(
                client, receipt["request"], {}
            )
            receipt["response"] = response
            save()
            histories = await source.__ray_call__.remote(
                continuation.backend_history
            )
            matches = [h for h in histories if h["request_id"] == request_id]
            evidence.require(
                len(matches) == 1, "Missing or ambiguous control history"
            )
            receipt["history"] = matches[0]
            save()
            _response(response, matches[0], model, 1, request_id)
            return response, matches[0]
        except BaseException as error:
            receipt["error"] = str(error)
            save()
            raise

    params = evidence.parameters(scenario)
    if scenario == "eos":
        path = (
            Path(config["source_snapshot"]["path"]) / "generation_config.json"
        )
        generation = json.loads(path.read_text()) if path.exists() else {}
        identity = evidence.eos_identity(tokenizer.eos_token_id, generation)
        state["eos_source"] = dict(
            path=str(path),
            generation_config=generation,
            tokenizer_eos_token_id=tokenizer.eos_token_id,
        )
        spec = {"scenario": scenario, "eos": identity}
        selected = None
        for index, prompt in enumerate(evidence.EOS_PROMPTS):
            probe = {"prompt": prompt}
            state["probes"].append(probe)
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            probe["prompt_token_ids"] = prompt_ids
            save()
            if (
                not isinstance(prompt_ids, list)
                or not 0 < len(prompt_ids) <= 128
            ):
                probe["rejection"] = "Prompt outside the bounded token range"
                save()
                continue
            body = dict(model=model, input_tokens=prompt_ids, **params)
            _, history = await request(
                body, f"d2s-eos-probe-{index}-reference", probe
            )
            try:
                evidence.validate_terminal(history, spec)
            except ValueError as error:
                probe["rejection"] = str(error)
                save()
                continue
            selected = probe
            break
        evidence.require(
            selected is not None,
            "No qualified EOS probe in the bounded prompt set",
        )
        state["selected_probe"] = selected["request"]["request_id"]
    else:
        body = dict(model=model, input_tokens=tokens, **params)
        probe = {}
        state["probes"].append(probe)
        _, history = await request(
            body, f"d2s-{scenario}-probe-reference", probe
        )
        spec = evidence.stop_spec(scenario, history, decode)
        body["stop"] = [spec["stop"]]
        state["selected_probe"] = probe["request"]["request_id"]
    state["spec"] = spec
    reference_record = state["control"] = {}
    save()
    response, history = await request(
        body, f"d2s-{scenario}-reference", reference_record
    )
    evidence.validate_terminal(history, spec)
    if scenario == "eos":
        evidence.require(
            _canonical(history["choices"])
            == _canonical(selected["history"]["choices"]),
            "Independent EOS controls differ",
        )
    else:
        raw = decode(evidence.one(history)["token_ids"])
        evidence.require(
            raw.startswith(spec["probe_text"][: spec["end"]])
            and raw.find(spec["stop"]) == spec["start"],
            "Stop control does not reach the selected first match",
        )
    save()
    return body, response, history


def validate_scenario(record, tokenizer):
    if record["scenario"] != "eos":
        record["stopping"]["decoded_prefixes"] = {
            item["request_id"]: {
                "token_ids": list(evidence.one(item)["token_ids"]),
                "text": tokenizer.decode(
                    evidence.one(item)["token_ids"], **evidence.DECODE_OPTIONS
                ),
            }
            for item in record["frozen"]["requests"]
        }
    return evidence.audit_scenario(record)


def main():
    return continuation.main(suite=sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
