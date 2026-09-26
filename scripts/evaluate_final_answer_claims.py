"""Select existing final-answer claims and compare them with gold answers using gpt-oss-120b."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factcheck_answers import now, run_stage
from generate_answers import read_jsonl, write_atomic, write_json

from clients import APIConfig, get_client
from utils import parse_json_response

MODEL = "gpt-oss-120b"
PROMPTS = {
    "selection": ROOT / "prompts/final_answer_selection.txt",
    "correctness": ROOT / "prompts/final_answer_correctness.txt",
}
SCHEMAS = {
    "selection": {
        "status": {"type": "string", "enum": ["selected", "no_matching_claim"]},
        "claim_index": {"type": ["integer", "null"]},
        "rationale": {"type": "string"},
    },
    "correctness": {
        "label": {"type": "string", "enum": ["match", "mismatch", "undetermined", "no_claim"]},
        "answer_text": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_input(doc, stage):
    if stage == "selection":
        return {
            "question": doc["question"],
            "response": doc["response"],
            "claims": [{"index": i, "claim": text} for i, text in enumerate(doc["claims"])],
        }
    return {
        "question": doc["question"],
        "selected_claim": doc["selection"]["claim"],
        "reference_answers": doc["reference_answers"],
    }


def validate(payload, doc, stage):
    if set(payload) != set(SCHEMAS[stage]):
        raise ValueError(f"Unexpected fields in {stage} response.")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("Expected a nonempty rationale.")
    if stage == "selection":
        index = payload["claim_index"]
        if payload["status"] == "selected":
            if type(index) is not int or not 0 <= index < len(doc["claims"]):
                raise ValueError("Selected claim index is outside the existing claims.")
        elif payload["status"] != "no_matching_claim" or index is not None:
            raise ValueError("No matching claim requires a null index.")
        return {**payload, "claim": doc["claims"][index] if index is not None else None}
    label = payload["label"]
    if label not in SCHEMAS[stage]["label"]["enum"]:
        raise ValueError("Invalid correctness label.")
    no_claim = doc["selection"]["claim"] is None
    if no_claim != (label == "no_claim"):
        raise ValueError("Correctness label does not match claim availability.")
    answer = payload["answer_text"]
    if no_claim and answer is not None:
        raise ValueError("Missing claims cannot have an extracted answer.")
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError("Answer must be null or nonempty text.")
    if label in ("match", "mismatch") and answer is None:
        raise ValueError("Definite correctness requires an answer.")
    return {**payload, "is_correct": True if label == "match" else False if label == "mismatch" else None}


def pending(documents, stage):
    return [
        (doc["qid"], None, None)
        for doc in documents
        if stage not in doc and (stage == "selection" or "selection" in doc)
    ]


def summary(documents):
    labels = Counter(doc["correctness"]["label"] for doc in documents if "correctness" in doc)
    selected = sum(doc.get("selection", {}).get("status") == "selected" for doc in documents)
    complete = sum("correctness" in doc for doc in documents)
    decided = labels["match"] + labels["mismatch"]
    return {
        "updated_at": now(),
        "model": MODEL,
        "documents": len(documents),
        "selection_completed": sum("selection" in doc for doc in documents),
        "selected_claims": selected,
        "no_matching_claim": sum(doc.get("selection", {}).get("status") == "no_matching_claim" for doc in documents),
        "correctness_completed": complete,
        "complete": complete == len(documents),
        "labels": {label: labels[label] for label in SCHEMAS["correctness"]["label"]["enum"]},
        "match_rate_all_questions": labels["match"] / len(documents) if complete == len(documents) else None,
        "match_rate_decided_claims": labels["match"] / decided if decided else None,
    }


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_concurrency < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Concurrency and limit must be positive.")
    rows, gold = read_jsonl(args.input), read_jsonl(args.gold)
    gold_by_id = {row["qid"]: row for row in gold}
    if not rows or len({r["qid"] for r in rows}) != len(rows) or len(gold_by_id) != len(gold):
        raise ValueError("Inputs must contain unique IDs and nonempty claims data.")
    if {r["qid"] for r in rows} != set(gold_by_id):
        raise ValueError("Fact-check and gold question IDs must match exactly.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = APIConfig.from_env("factchecker")
    protocol = {
        "model": MODEL,
        "connection": {"api_type": config.api_type, "endpoint": config.endpoint},
        "input_file": str(args.input.resolve()),
        "input_sha256": digest(args.input),
        "gold_file": str(args.gold.resolve()),
        "gold_sha256": digest(args.gold),
        "documents": len(rows),
        "sampling": "Provider defaults, as in FACTCHECKER.",
        "selection": "One existing zero-based claim index, or no_matching_claim; no gold answers in this request.",
        "correctness": "Separate request compares the selected claim against the dataset answers aliases.",
        "prompt_sha256": {name: digest(path) for name, path in PROMPTS.items()},
        "implementation_sha256": {
            name: digest(ROOT / name)
            for name in (
                "scripts/evaluate_final_answer_claims.py",
                "scripts/factcheck_answers.py",
                "scripts/generate_answers.py",
                "src/clients.py",
                "src/utils.py",
            )
        },
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Input, prompts or protocol changed; choose another output directory.")
    else:
        if (output / "documents").exists():
            raise ValueError("Checkpoint directory exists without a protocol.")
        write_json(protocol_path, protocol)
    (output / "documents").mkdir(exist_ok=True)
    (output / "prompts").mkdir(exist_ok=True)
    prompts = {}
    for name, source in PROMPTS.items():
        path = output / "prompts" / source.name
        if path.exists() and digest(path) != protocol["prompt_sha256"][name]:
            raise ValueError("Saved prompt differs from protocol.")
        if not path.exists():
            path.write_bytes(source.read_bytes())
        prompts[name] = path.read_text()

    documents = []
    for row in rows:
        qid = row["qid"]
        reference = gold_by_id[qid]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", qid):
            raise ValueError("Unsafe question ID.")
        if row["question"] != reference["question"] or not row["decomposition_complete"]:
            raise ValueError(f"Question mismatch or missing decomposition for {qid}.")
        answers = reference["answers"]
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(a, str) or not a.strip() for a in answers)
        ):
            raise ValueError(f"Invalid reference answers for {qid}.")
        expected = {
            "qid": qid,
            "question": row["question"],
            "response": row["response"],
            "claims": [claim["claim"] for claim in row["claims"]],
            "reference_answers": answers,
            "model": MODEL,
        }
        path = output / "documents" / f"{qid}.json"
        doc = json.loads(path.read_text()) if path.exists() else expected.copy()
        if any(doc.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Checkpoint does not match inputs for {qid}.")
        documents.append(doc)
    selected = documents[: args.limit] if args.limit else documents
    by_id = {doc["qid"]: doc for doc in documents}
    client = get_client("factchecker")
    client.timeout = 180
    client.max_retries = 2
    errors = []

    def checkpoint():
        write_json(output / "summary.json", {**summary(documents), "errors_this_run": errors})

    checkpoint()
    for stage in ("selection", "correctness"):

        def invoke(task):
            doc = by_id[task[0]]
            for attempt in range(3):
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompts[stage]},
                        {"role": "user", "content": json.dumps(model_input(doc, stage), ensure_ascii=False)},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"final_answer_{stage}",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": SCHEMAS[stage],
                                "required": list(SCHEMAS[stage]),
                                "additionalProperties": False,
                            },
                        },
                    },
                )
                try:
                    value = validate(parse_json_response(response), doc, stage)
                except ValueError:
                    if attempt == 2:
                        raise
                    sleep(1)
                    continue
                return {
                    **value,
                    "model": MODEL,
                    "response_model": response.model,
                    "response_id": response.id,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "attempts": attempt + 1,
                    "completed_at": now(),
                }

        def accept(task, value):
            doc = by_id[task[0]]
            doc[stage] = value
            doc["status"] = "complete" if "correctness" in doc else "correctness"
            write_json(output / "documents" / f"{doc['qid']}.json", doc)

        stage_errors = run_stage(
            stage, pending(selected, stage), invoke, accept, checkpoint, args.max_concurrency, output / "errors.jsonl"
        )
        errors.extend(stage_errors)
        checkpoint()
        if stage_errors:
            break
    write_atomic(output / "results.jsonl", "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents))
    checkpoint()
    print(json.dumps(summary(documents), ensure_ascii=False), flush=True)
    if errors or any("correctness" not in doc for doc in selected):
        raise SystemExit("Incomplete evaluation. Rerun the same command to retry missing operations.")


if __name__ == "__main__":
    main()
