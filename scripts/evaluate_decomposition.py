"""Evaluate an editable decomposition prompt on the paper's AIO test split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_metrics import evaluate_documents
from decompose import DEFAULT_PROMPT_PATH, RESPONSE_FORMAT, decompose_document_into_claims


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_git(repo: Path, revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])


def load_test_data(repo: Path, source: dict) -> list[dict]:
    revision = source["revision"]
    test = json.loads(read_git(repo, revision, source["test_source"]))
    gold = json.loads(read_git(repo, revision, source["gold_source"]))
    by_id = {row["textId"]: row for row in gold}
    if len(by_id) != len(gold) or len({row["textId"] for row in test}) != len(test):
        raise ValueError("Duplicate document IDs in the reference dataset.")
    rows = []
    for item in test:
        row = by_id[item["textId"]]
        if row["text"] != item["text"]:
            raise ValueError(f"Test and gold text differ for {item['textId']}.")
        rows.append({"textId": row["textId"], "text": row["text"], "claims": row["claims"]})
    return rows


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "result/decomposition")
    parser.add_argument("--model", default=os.getenv("FACTCHECKER_MODEL", "").strip() or "gpt-5.4-2026-03-05")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "evaluations/decomposition_8shot_source.json")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120, help="API request timeout in seconds.")
    parser.add_argument("--limit", type=int, help="Smoke test only: evaluate the first N documents.")
    parser.add_argument("--score-only", action="store_true", help="Score saved predictions without API requests.")
    parser.add_argument(
        "--refresh-metrics", action="store_true", help="With --score-only, allow a changed claim_metrics.py."
    )
    args = parser.parse_args()
    if args.refresh_metrics and not args.score_only:
        parser.error("--refresh-metrics requires --score-only.")
    if (
        min(args.runs, args.max_concurrency) < 1
        or args.request_timeout <= 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Runs, concurrency, timeout, and limit must be positive.")
    source = json.loads(args.source_manifest.read_text())
    gold = load_test_data(args.dataset_repo, source)
    if args.limit:
        gold = gold[: args.limit]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    prompt_bytes = args.prompt.read_bytes()
    gold_bytes = json.dumps(gold, ensure_ascii=False, sort_keys=True).encode()
    protocol = {
        "source": source,
        "dataset": "AIO test" if args.limit is None else "AIO test subset (smoke test)",
        "documents": len(gold),
        "gold_claims": sum(len(row["claims"]) for row in gold),
        "model": args.model,
        "runs": args.runs,
        "prompt_sha256": digest(prompt_bytes),
        "gold_sha256": digest(gold_bytes),
        "implementation_sha256": {
            name: digest((ROOT / "src" / name).read_bytes()) for name in ("decompose.py", "claim_metrics.py")
        },
        "response_format": RESPONSE_FORMAT,
        "context": "none (generated text only, as in the reference experiment)",
        "sampling": "provider defaults; no temperature, top_p, or seed override",
        "matching": "maximum-weight one-to-one assignment, then similarity >= threshold",
        "fuzzy_threshold": 0.8,
        "tokenizer": "MeCab, UniDic-lite, POS noun/verb/adjective/adverb; feature[10] with surface fallback",
        "aggregation": "macro P/R/F1 across documents, then mean and population SD across independent runs",
        "versions": {
            name: importlib.metadata.version(name) for name in ("openai", "mecab-python3", "unidic-lite", "scipy")
        },
    }
    protocol_path = output / "protocol.json"
    previous_scoring_protocol = None
    if protocol_path.exists():
        saved_protocol = json.loads(protocol_path.read_text())
        if saved_protocol != protocol:
            comparable = dict(protocol)
            comparable["implementation_sha256"] = dict(protocol["implementation_sha256"])
            comparable["implementation_sha256"]["claim_metrics.py"] = saved_protocol["implementation_sha256"][
                "claim_metrics.py"
            ]
            if not args.refresh_metrics or comparable != saved_protocol:
                raise ValueError("Evaluation settings changed. Use a new output directory to avoid mixing runs.")
            previous_scoring_protocol = saved_protocol
    elif args.score_only:
        raise ValueError("No saved evaluation protocol found.")
    else:
        write_json(protocol_path, protocol)
        (output / "prompt.json").write_bytes(prompt_bytes)
        write_json(output / "gold.json", gold)
    if digest((output / "prompt.json").read_bytes()) != protocol["prompt_sha256"]:
        raise ValueError("Saved prompt snapshot differs from the evaluation protocol.")
    if json.loads((output / "gold.json").read_text()) != gold:
        raise ValueError("Saved gold data differs from the evaluation protocol.")

    results = []
    for run in range(1, args.runs + 1):
        path = output / f"predictions_{run:02d}.json"
        saved = json.loads(path.read_text()) if path.exists() else []
        by_id = {row["textId"]: row for row in saved}
        gold_by_id = {row["textId"]: row for row in gold}
        if len(by_id) != len(saved) or not by_id.keys() <= gold_by_id.keys():
            raise ValueError("Saved predictions have duplicate or unexpected IDs.")
        for row in saved:
            if row["text"] != gold_by_id[row["textId"]]["text"]:
                raise ValueError("Saved prediction text differs from the reference.")
        pending = [row for row in gold if row["textId"] not in by_id]
        if pending and args.score_only:
            raise ValueError(f"Run {run} is incomplete: {len(pending)} documents missing.")
        if pending:
            # Validate credentials before queueing requests; never serialize connection secrets.
            from clients import get_client

            get_client("factchecker").timeout = args.request_timeout
            executor = ThreadPoolExecutor(max_workers=args.max_concurrency)
            futures = {
                executor.submit(
                    decompose_document_into_claims,
                    row["text"],
                    args.model,
                    prompt_path=output / "prompt.json",
                ): row
                for row in pending
            }
            try:
                for future in as_completed(futures):
                    row = futures[future]
                    claims = future.result()
                    by_id[row["textId"]] = {
                        "textId": row["textId"],
                        "text": row["text"],
                        "claims": [{"text": text} for text in claims],
                    }
                    write_json(path, [by_id[row["textId"]] for row in gold if row["textId"] in by_id])
                    print(f"Run {run}/{args.runs}: {len(by_id)}/{len(gold)} documents", flush=True)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        scores = evaluate_documents(list(by_id.values()), gold)
        write_json(output / f"scores_{run:02d}.json", scores)
        results.append(scores["macro"])
        print(f"Run {run}: {json.dumps(scores['macro'])}", flush=True)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "runs": results,
        "mean": {
            metric: {key: mean(run[metric][key] for run in results) for key in ("precision", "recall", "f1")}
            for metric in ("exact", "fuzzy_content")
        },
        "std": {
            metric: {key: pstdev(run[metric][key] for run in results) for key in ("precision", "recall", "f1")}
            for metric in ("exact", "fuzzy_content")
        },
    }
    if previous_scoring_protocol is not None:
        write_json(output / "previous_scoring_protocol.json", previous_scoring_protocol)
        write_json(protocol_path, protocol)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
