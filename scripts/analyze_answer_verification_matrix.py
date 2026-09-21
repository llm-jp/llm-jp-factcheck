"""Join final-answer accuracy to the same claim's verdict and reproducibly sample every cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factcheck_answers import now, run_stage
from generate_answers import read_jsonl, write_atomic, write_json

from clients import APIConfig, get_client
from utils import parse_json_response
from verify import VERIFICATION_LABELS

ROWS = ("match", "mismatch", "undetermined", "no_claim")
COLS = (*VERIFICATION_LABELS, "Not checkworthy", "No selected claim")
MECHANISMS = (
    "aligned",
    "question_claim_gap",
    "extra_detail",
    "missing_evidence",
    "inference_boundary",
    "verification_questionable",
    "correctness_questionable",
    "checkworthiness_gate",
    "no_answer_claim",
)
FIELDS = {
    "primary_mechanism": {"type": "string", "enum": list(MECHANISMS)},
    "correctness_review": {"type": "string", "enum": ["agree", "questionable", "not_evaluable", "not_run"]},
    "verification_review": {"type": "string", "enum": ["agree", "questionable", "not_evaluable", "not_run"]},
    "observation": {"type": "string"},
    "evidence_quote": {"type": ["string", "null"]},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_records(answers, factchecks):
    by_id = {row["qid"]: row for row in factchecks}
    if len(by_id) != len(factchecks) or len({r["qid"] for r in answers}) != len(answers):
        raise ValueError("Duplicate question IDs.")
    if {r["qid"] for r in answers} != set(by_id):
        raise ValueError("Inputs must cover exactly the same questions.")
    joined = []
    for row in answers:
        fact = by_id[row["qid"]]
        if any(row[k] != fact[k] for k in ("question", "response")):
            raise ValueError("Question or response mismatch.")
        if row["status"] != "complete" or fact["status"] != "complete":
            raise ValueError("Only complete evaluation inputs are accepted.")
        index = row["selection"]["claim_index"]
        evidence = None
        claim = None
        if index is None:
            if row["selection"]["claim"] is not None or row["correctness"]["label"] != "no_claim":
                raise ValueError("Inconsistent missing claim.")
            verdict = "No selected claim"
        else:
            if type(index) is not int or not 0 <= index < len(fact["claims"]):
                raise ValueError("Invalid claim index.")
            claim = fact["claims"][index]
            if claim["claim"] != row["selection"]["claim"]:
                raise ValueError("Selected claim text does not match source index.")
            if claim["is_checkworthy"] is False:
                if claim["evidences"]:
                    raise ValueError("Unexpected evidence for a skipped claim.")
                verdict = "Not checkworthy"
            else:
                if not claim["retrieval_complete"] or len(claim["evidences"]) != 1:
                    raise ValueError("Expected exactly one retrieved evidence per checked claim.")
                evidence = claim["evidences"][0]
                verdict = evidence["verification"]["label"]
        if verdict not in COLS or row["correctness"]["label"] not in ROWS:
            raise ValueError("Unknown label.")
        joined.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "response": row["response"],
                "reference_answers": row["reference_answers"],
                "claim_index": index,
                "claim": row["selection"]["claim"],
                "selection_rationale": row["selection"]["rationale"],
                "available_claims": [c["claim"] for c in fact["claims"]],
                "correctness": row["correctness"]["label"],
                "answer_text": row["correctness"]["answer_text"],
                "correctness_rationale": row["correctness"]["rationale"],
                "verification": verdict,
                "is_checkworthy": claim["is_checkworthy"] if claim else None,
                "evidence": evidence["passage"] if evidence else None,
                "verification_rationale": evidence["verification"]["rationale"] if evidence else None,
                "evidence_document_id": evidence.get("document_id") if evidence else None,
            }
        )
    return joined


def stratify(joined, seed=20260920, limit=20):
    cells = defaultdict(list)
    for row in joined:
        cells[(row["correctness"], row["verification"])].append(row)
    samples = []
    for actual in ROWS:
        for verdict in COLS:
            key = (actual, verdict)
            population = sorted(cells[key], key=lambda r: r["qid"])
            cell_seed = int.from_bytes(hashlib.sha256(f"{seed}|{actual}|{verdict}".encode()).digest(), "big")
            samples.extend(random.Random(cell_seed).sample(population, min(limit, len(population))))
    return cells, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--factchecks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("Sample size must be positive.")
    joined = join_records(read_jsonl(args.answers), read_jsonl(args.factchecks))
    cells, samples = stratify(joined, args.seed, args.sample_size)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "answers": str(args.answers.resolve()),
        "answers_sha256": digest(args.answers),
        "factchecks": str(args.factchecks.resolve()),
        "factchecks_sha256": digest(args.factchecks),
        "unit": "One question, paired by qid and exact zero-based selected claim index and text.",
        "seed": args.seed,
        "max_sample_size_per_cell": args.sample_size,
        "sampling": "Sorted qids; random.Random seeded by SHA256(seed|correctness|verification), without replacement per cell.",
        "implementation_sha256": digest(Path(__file__)),
    }
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Inputs or sampling changed; use another output directory.")
    write_json(path, protocol)
    matrix = {
        "rows": list(ROWS),
        "columns": list(COLS),
        "counts": [[len(cells[(r, c)]) for c in COLS] for r in ROWS],
        "sample_counts": [[min(args.sample_size, len(cells[(r, c)])) for c in COLS] for r in ROWS],
        "total": len(joined),
        "sample_total": len(samples),
    }
    write_json(out / "matrix.json", matrix)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["correctness", *COLS, "total"])
    for row, counts in zip(ROWS, matrix["counts"]):
        writer.writerow([row, *counts, sum(counts)])
    write_atomic(out / "matrix.csv", stream.getvalue())
    for name, rows in (("joined", joined), ("samples", samples)):
        write_atomic(out / f"{name}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(matrix), flush=True)
    if not args.review:
        return

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    model = "gpt-oss-120b"
    prompt_path = ROOT / "prompts/answer_verification_review.txt"
    prompt = prompt_path.read_text()
    config = APIConfig.from_env("factchecker")
    review_protocol = {
        "model": model,
        "endpoint": config.endpoint,
        "prompt_sha256": digest(prompt_path),
        "samples_sha256": digest(out / "samples.jsonl"),
        "sampling": "Provider defaults.",
        "note": "Model-assisted qualitative review; original labels remain unchanged; no external sources.",
        "quotation_policy": "Only exact substrings are published as quotes; rejected model quotes are retained separately.",
    }
    path = out / "review_protocol.json"
    if path.exists() and json.loads(path.read_text()) != review_protocol:
        raise ValueError("Review protocol changed.")
    write_json(path, review_protocol)
    write_atomic(out / "review_prompt.txt", prompt)
    (out / "reviews").mkdir(exist_ok=True)
    by_id = {r["qid"]: r for r in samples}
    reviews = {}
    for row in samples:
        path = out / "reviews" / f"{row['qid']}.json"
        if path.exists():
            reviews[row["qid"]] = json.loads(path.read_text())
    client = get_client("factchecker")
    client.timeout = 180
    client.max_retries = 2

    def invoke(task):
        row = by_id[task[0]]
        for attempt in range(3):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(row, ensure_ascii=False)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qualitative_review",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": FIELDS,
                            "required": list(FIELDS),
                            "additionalProperties": False,
                        },
                    },
                },
            )
            try:
                value = parse_json_response(response)
                if set(value) != set(FIELDS):
                    raise ValueError("Unexpected fields.")
                for key, field in FIELDS.items():
                    if "enum" in field and value[key] not in field["enum"]:
                        raise ValueError("Unknown category.")
                if not isinstance(value["observation"], str) or not value["observation"].strip():
                    raise ValueError("Missing observation.")
                quote = value["evidence_quote"]
                quote_valid = quote is None or (
                    isinstance(quote, str) and bool(quote) and quote in (row["evidence"] or "")
                )
            except ValueError:
                if attempt == 2:
                    raise
                continue
            if not quote_valid:
                value["rejected_model_quote"] = quote
                value["evidence_quote"] = None
            return {
                "qid": row["qid"],
                **value,
                "model": model,
                "response_model": response.model,
                "response_id": response.id,
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
                "completed_at": now(),
            }

    def accept(task, value):
        reviews[task[0]] = value
        write_json(out / "reviews" / f"{task[0]}.json", value)

    def checkpoint():
        write_json(
            out / "review_progress.json", {"completed": len(reviews), "total": len(samples), "updated_at": now()}
        )

    tasks = [(r["qid"], None, None) for r in samples if r["qid"] not in reviews]
    errors = run_stage("qualitative_review", tasks, invoke, accept, checkpoint, 8, out / "review_errors.jsonl")
    write_atomic(
        out / "reviewed_samples.jsonl",
        "".join(json.dumps({**r, "review": reviews.get(r["qid"])}, ensure_ascii=False) + "\n" for r in samples),
    )
    aggregate = []
    for actual in ROWS:
        for verdict in COLS:
            selected = [r for r in samples if r["correctness"] == actual and r["verification"] == verdict]
            if not selected:
                continue
            aggregate.append(
                {
                    "correctness": actual,
                    "verification": verdict,
                    "population": len(cells[(actual, verdict)]),
                    "sample_size": len(selected),
                    "qids": [r["qid"] for r in selected],
                    **{
                        key: dict(Counter(reviews[r["qid"]][key] for r in selected if r["qid"] in reviews))
                        for key in ("primary_mechanism", "correctness_review", "verification_review")
                    },
                }
            )
    write_json(out / "qualitative_summary.json", aggregate)
    if errors or len(reviews) != len(samples):
        raise SystemExit("Some reviews are missing; rerun the same command to resume.")


if __name__ == "__main__":
    main()
