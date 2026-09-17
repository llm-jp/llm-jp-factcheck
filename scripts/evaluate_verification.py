"""Evaluate verdict prediction with short rationales on AIO over three runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_decomposition import digest, read_git, write_json

from verdict_metrics import evaluate_verdicts, normalize_label
from verify import DEFAULT_PROMPT_PATH, RESPONSE_FORMAT, verify_claim


def load_cohort(repo: Path, source: dict) -> tuple[list[dict], list[int]]:
    rows = json.loads(read_git(repo, source["revision"], source["test_source"]))
    documents = json.loads(read_git(repo, source["revision"], source["question_source"]))
    questions = {row["textId"]: row["prompt"] for row in documents}
    if len({row["ID"] for row in rows}) != len(rows):
        raise ValueError("Duplicate test pair IDs.")
    kept, excluded = [], []
    for row in rows:
        if row["evidence"].strip() == questions[row["textID"]].strip():
            excluded.append(row["ID"])
        else:
            if not row["claim"].strip() or not row["evidence"].strip():
                raise ValueError(f"Blank claim or evidence in pair {row['ID']}.")
            kept.append({**row, "label": normalize_label(row["label"])})
    return kept, excluded


def read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Discard only an incomplete final write after a process interruption.
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        end = raw.rfind(b"\n") + 1
        path.write_bytes(raw[:end])
        raw = raw[:end]
    return [json.loads(line) for line in raw.splitlines()]


def reference_scores(repo: Path, source: dict, gold: list[dict]) -> dict:
    by_id = {row["ID"]: row for row in gold}
    runs = []
    for filename in source["reference_predictions"]:
        predictions = json.loads(read_git(repo, source["revision"], filename))
        changed_inputs = [
            row["ID"]
            for row in predictions
            if any(row[field] != by_id[row["ID"]][field] for field in ("claim", "evidence"))
        ]
        changed_gold = [row["ID"] for row in predictions if normalize_label(row["gold"]) != by_id[row["ID"]]["label"]]
        prediction_ids = {row["ID"] for row in predictions}
        selected = [{"ID": row["ID"], "label": normalize_label(row["gold"])} for row in predictions]
        scores = evaluate_verdicts(
            [{"ID": row["ID"], "label": normalize_label(row["predicted"])} for row in predictions], selected
        )
        runs.append(
            {
                "source": filename,
                "missing_ids": sorted(by_id.keys() - prediction_ids),
                "changed_input_ids": changed_inputs,
                "changed_gold_ids": changed_gold,
                "scores": scores,
            }
        )
    return {"runs": runs}


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "result/verification-rationale")
    parser.add_argument("--model", default=os.getenv("FACTCHECKER_MODEL", "").strip() or "gpt-5.4-2026-03-05")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--limit", type=int, help="Use only the first N pairs for a smoke test.")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--retry-errors", action="store_true", help="Retry failed pairs, preserving successful predictions."
    )
    args = parser.parse_args()
    if min(args.runs, args.max_concurrency, args.request_timeout) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Runs, concurrency, timeout, and limit must be positive.")
    if args.retry_errors and args.score_only:
        parser.error("--retry-errors cannot be combined with --score-only.")
    source = json.loads((ROOT / "evaluations/verification_source.json").read_text())
    full_gold, excluded = load_cohort(args.dataset_repo, source)
    gold = full_gold[: args.limit] if args.limit else full_gold
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    prompt_bytes = args.prompt.read_bytes()
    protocol = {
        "source": source,
        "model": args.model,
        "runs": args.runs,
        "dataset": "AIO test, excluding original-input evidence" + (" (smoke-test subset)" if args.limit else ""),
        "pairs": len(gold),
        "excluded_input_evidence": len(excluded),
        "gold_sha256": digest(json.dumps(gold, ensure_ascii=False, sort_keys=True).encode()),
        "prompt_sha256": digest(prompt_bytes),
        "response_format": RESPONSE_FORMAT,
        "implementation_sha256": {
            name: digest((ROOT / "src" / name).read_bytes()) for name in ("verify.py", "verdict_metrics.py")
        },
        "versions": {"openai": importlib.metadata.version("openai")},
        "sampling": "provider defaults; no temperature, top_p, seed, or reasoning-effort override",
        "output": "Japanese label mapped to English, followed by a short English rationale (one sentence, at most 40 words)",
        "aggregation": "pair accuracy; macro P/R/F1 over the fixed six labels; mean and population SD over runs",
        "errors": "explicit failed predictions count as incorrect; no missing pair is silently excluded",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Evaluation settings changed. Use a new output directory.")
    elif args.score_only:
        raise ValueError("No saved protocol found.")
    else:
        write_json(protocol_path, protocol)
        write_json(output / "gold.json", gold)
        write_json(output / "excluded_input_ids.json", excluded)
        (output / "prompt.yaml").write_bytes(prompt_bytes)
    if not (output / "reference_scores.json").exists():
        write_json(output / "reference_scores.json", reference_scores(args.dataset_repo, source, full_gold))
    if digest((output / "prompt.yaml").read_bytes()) != protocol["prompt_sha256"]:
        raise ValueError("Saved prompt snapshot changed.")
    if json.loads((output / "gold.json").read_text()) != gold:
        raise ValueError("Saved gold snapshot changed.")
    gold_by_id = {row["ID"]: row for row in gold}
    runs = []
    for run in range(1, args.runs + 1):
        path = output / f"predictions_{run:02d}.jsonl"
        saved = read_predictions(path)
        by_id = {row["ID"]: row for row in saved}
        if len(by_id) != len(saved) or not by_id.keys() <= gold_by_id.keys():
            raise ValueError("Saved predictions have duplicate or unexpected IDs.")
        if args.retry_errors:
            by_id = {key: row for key, row in by_id.items() if row.get("label") is not None}
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in by_id.values()))
        pending = [row for row in gold if row["ID"] not in by_id]
        if pending and args.score_only:
            raise ValueError(f"Run {run} is incomplete: {len(pending)} missing pairs.")
        if pending:
            from clients import get_client

            get_client("factchecker").timeout = args.request_timeout
            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                futures = {
                    executor.submit(
                        verify_claim, row["claim"], row["evidence"], args.model, prompt_path=output / "prompt.yaml"
                    ): row["ID"]
                    for row in pending
                }
                try:
                    for future in as_completed(futures):
                        identifier = futures[future]
                        try:
                            result = {"ID": identifier, **future.result()}
                        except Exception as exc:
                            result = {
                                "ID": identifier,
                                "label": None,
                                "error": {
                                    "type": type(exc).__name__,
                                    "status_code": getattr(exc, "status_code", None),
                                    "code": getattr(exc, "code", None),
                                },
                            }
                        by_id[identifier] = result
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        if len(by_id) % 50 == 0 or len(by_id) == len(gold):
                            errors = sum(row.get("label") is None for row in by_id.values())
                            print(f"Run {run}/{args.runs}: {len(by_id)}/{len(gold)} pairs; {errors} errors", flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        scores = evaluate_verdicts(list(by_id.values()), gold)
        write_json(output / f"scores_{run:02d}.json", scores)
        runs.append(scores)
        print(
            f"Run {run}: accuracy={scores['accuracy']:.4f}; macro={scores['macro']}; errors={scores['errors']}",
            flush=True,
        )
    values = [{"accuracy": run["accuracy"], **run["macro"]} for run in runs]
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "runs": runs,
        "mean": {key: mean(row[key] for row in values) for key in values[0]},
        "std": {key: pstdev(row[key] for row in values) for key in values[0]},
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
