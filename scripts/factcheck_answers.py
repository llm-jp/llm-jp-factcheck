"""Fact-check generated answer JSONL, checkpointing every completed operation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_answers import read_jsonl, write_atomic, write_json

import checkworthy
import decompose
import verify
from clients import APIConfig, get_client
from pipeline import get_search_client, get_tokenizer
from retrieval import search_documents


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def refresh_status(document: dict) -> None:
    for claim in document["claims"]:
        if claim["is_checkworthy"] is None:
            claim["status"] = "checkworthiness"
        elif claim["is_checkworthy"] is False:
            claim["status"] = "complete"
        elif not claim["retrieval_complete"]:
            claim["status"] = "retrieval"
        elif any("verification" not in evidence for evidence in claim["evidences"]):
            claim["status"] = "verification"
        else:
            claim["status"] = "complete"
    if not document["decomposition_complete"]:
        document["status"] = "decomposition"
    else:
        document["status"] = next(
            (
                stage
                for stage in ("checkworthiness", "retrieval", "verification")
                if any(claim["status"] == stage for claim in document["claims"])
            ),
            "complete",
        )


def apply_result(document: dict, stage: str, claim_index: int | None, evidence_index: int | None, value) -> None:
    if stage == "decomposition":
        document["claims"] = [
            {
                "claim": text,
                "is_checkworthy": None,
                "retrieval_complete": False,
                "no_evidence": False,
                "evidences": [],
            }
            for text in value
        ]
        document["decomposition_complete"] = True
    elif stage == "checkworthiness":
        document["claims"][claim_index]["is_checkworthy"] = value
    elif stage == "retrieval":
        claim = document["claims"][claim_index]
        claim.update(value)
        claim["retrieval_complete"] = True
        claim["no_evidence"] = not claim["evidences"]
    elif stage == "verification":
        document["claims"][claim_index]["evidences"][evidence_index]["verification"] = value
    else:
        raise ValueError(f"Unknown stage: {stage}")
    refresh_status(document)
    document["updated_at"] = now()


def pending_tasks(documents: list[dict], stage: str) -> list[tuple[str, int | None, int | None]]:
    tasks = []
    for doc in documents:
        qid = doc["qid"]
        if stage == "decomposition":
            if not doc["decomposition_complete"]:
                tasks.append((qid, None, None))
            continue
        for index, claim in enumerate(doc["claims"]):
            if stage == "checkworthiness" and claim["is_checkworthy"] is None:
                tasks.append((qid, index, None))
            elif stage == "retrieval" and claim["is_checkworthy"] is True and not claim["retrieval_complete"]:
                tasks.append((qid, index, None))
            elif stage == "verification" and claim["is_checkworthy"] is True and claim["retrieval_complete"]:
                for position, evidence in enumerate(claim["evidences"]):
                    if "verification" not in evidence:
                        tasks.append((qid, index, position))
    return tasks


def summarize(documents: list[dict]) -> dict:
    claims = [claim for doc in documents for claim in doc["claims"]]
    evidences = [e for claim in claims for e in claim["evidences"]]
    verdicts = Counter(e["verification"]["label"] for e in evidences if "verification" in e)
    return {
        "updated_at": now(),
        "documents": len(documents),
        "documents_decomposed": sum(doc["decomposition_complete"] for doc in documents),
        "documents_complete": sum(doc["status"] == "complete" for doc in documents),
        "complete": all(doc["status"] == "complete" for doc in documents),
        "claims": len(claims),
        "checkworthiness_completed": sum(claim["is_checkworthy"] is not None for claim in claims),
        "checkworthy": sum(claim["is_checkworthy"] is True for claim in claims),
        "not_checkworthy": sum(claim["is_checkworthy"] is False for claim in claims),
        "retrieval_completed": sum(claim["retrieval_complete"] for claim in claims),
        "no_evidence": sum(claim["no_evidence"] for claim in claims),
        "evidence_pairs": len(evidences),
        "verification_completed": sum(verdicts.values()),
        "verdicts": {label: verdicts[label] for label in verify.VERIFICATION_LABELS},
    }


def run_stage(stage, tasks, invoke, accept, checkpoint, concurrency, error_path):
    """Bound in-flight work; commit successes even if another request fails."""
    errors = []
    completed = 0
    print(f"{stage}: {len(tasks)} pending operations", flush=True)
    iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        fatal = False

        def fill():
            while not fatal and len(futures) < concurrency:
                task = next(iterator, None)
                if task is None:
                    break
                futures[executor.submit(invoke, task)] = task

        fill()
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in ready:
                task = futures.pop(future)
                try:
                    value = future.result()
                except Exception as exc:
                    error = {
                        "at": now(),
                        "stage": stage,
                        "qid": task[0],
                        "claim_index": task[1],
                        "evidence_index": task[2],
                        "type": type(exc).__name__,
                    }
                    if isinstance(exc, ValueError):
                        error["message"] = str(exc)
                    status = getattr(exc, "status_code", None)
                    if status is not None:
                        error["status_code"] = status
                    errors.append(error)
                    with error_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                    print(f"ERROR {json.dumps(error, ensure_ascii=False)}", flush=True)
                    fatal = fatal or status in (401, 403, 404) or len(errors) >= 8
                else:
                    accept(task, value)
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    checkpoint()
                    print(f"{stage}: {completed}/{len(tasks)} finished; {len(errors)} errors", flush=True)
            fill()
    checkpoint()
    return errors


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("FACTCHECKER_MODEL", "").strip() or "gpt-5.4-2026-03-05")
    parser.add_argument("--tokenizer-name", default=os.getenv("TOKENIZER_NAME", "").strip() or "llm-jp/llm-jp-3-13b")
    parser.add_argument("--es-host", default=os.getenv("ES_HOST", "").strip() or "http://10.2.73.12:9200")
    parser.add_argument("--es-index", default=os.getenv("ES_DUMP_INDEX", "").strip() or "llm-jp-corpus-v3")
    parser.add_argument("--num-evidences", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--limit", type=int, help="Process the first N answers; omit later to resume the rest.")
    args = parser.parse_args()
    if min(args.num_evidences, args.max_concurrency, args.request_timeout) <= 0 or (
        args.limit is not None and args.limit < 1
    ):
        parser.error("Counts and timeout must be positive.")
    rows = read_jsonl(args.input)
    if not rows or len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("Input must contain answers with unique IDs.")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["qid"]):
            raise ValueError("Unsafe qid for checkpoint filename.")
        for key in ("question", "response", "model"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing nonempty {key} in {row['qid']}.")
        if row.get("finish_reason") != "stop":
            raise ValueError(f"Input answer {row['qid']} is incomplete.")

    prompt_sources = {
        "decomposition": decompose.DEFAULT_PROMPT_PATH,
        "checkworthiness": checkworthy.DEFAULT_PROMPT_PATH,
        "verification": verify.DEFAULT_PROMPT_PATH,
    }
    connection = APIConfig.from_env("factchecker")
    protocol = {
        "input_file": str(args.input.resolve()),
        "input_sha256": sha256(args.input.read_bytes()),
        "documents": len(rows),
        "model": args.model,
        "context": "Question as preceding user message, serialized as JSON; no reference answers.",
        "sampling": "Provider defaults, as in the application; no overrides.",
        "connection": {"api_type": connection.api_type, "endpoint": connection.endpoint},
        "tokenizer_name": args.tokenizer_name,
        "es_host": args.es_host,
        "es_index": args.es_index,
        "num_evidences": args.num_evidences,
        "retrieval": "Token IDs, match query; full decoded passages in hit order; skip empty passages.",
        "prompt_sha256": {name: sha256(path.read_bytes()) for name, path in prompt_sources.items()},
        "implementation_sha256": {
            name: sha256((ROOT / "src" / name).read_bytes())
            for name in ("decompose.py", "checkworthy.py", "verify.py", "retrieval.py", "prompts.py", "utils.py")
        },
        "versions": {name: importlib.metadata.version(name) for name in ("openai", "transformers", "elasticsearch")},
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Input or protocol changed; use a different output directory.")
    else:
        if (output / "documents").exists():
            raise ValueError("Checkpoint directory exists without a protocol.")
        write_json(protocol_path, protocol)
    (output / "documents").mkdir(exist_ok=True)
    (output / "prompts").mkdir(exist_ok=True)
    prompts = {}
    for name, source in prompt_sources.items():
        path = output / "prompts" / f"{name}.yaml"
        if path.exists() and sha256(path.read_bytes()) != protocol["prompt_sha256"][name]:
            raise ValueError("Saved prompt differs from protocol.")
        if not path.exists():
            path.write_bytes(source.read_bytes())
        prompts[name] = path

    documents = []
    for row in rows:
        path = output / "documents" / f"{row['qid']}.json"
        expected = {key: row[key] for key in ("qid", "question", "response")}
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            if any(doc.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Checkpoint input mismatch for {row['qid']}.")
        else:
            doc = {
                **expected,
                "chatbot_model": row["model"],
                "factchecker_model": args.model,
                "decomposition_complete": False,
                "claims": [],
            }
        refresh_status(doc)
        documents.append(doc)
    by_id = {doc["qid"]: doc for doc in documents}
    selected = documents[: args.limit] if args.limit else documents
    errors = []
    started = monotonic()

    def checkpoint():
        summary = summarize(documents)
        summary["errors_this_run"] = errors
        summary["elapsed_seconds_this_run"] = round(monotonic() - started, 3)
        write_json(output / "summary.json", summary)

    client = get_client("factchecker")
    client.timeout = args.request_timeout
    client.max_retries = 2
    checkpoint()
    for stage in ("decomposition", "checkworthiness", "retrieval", "verification"):
        tasks = pending_tasks(selected, stage)
        tokenizer = None
        es = None
        encoded = {}
        if stage == "retrieval" and tasks:
            tokenizer = get_tokenizer(args.tokenizer_name)
            es = get_search_client(args.es_host)
            for task in tasks:
                encoded[task] = tokenizer.encode(by_id[task[0]]["claims"][task[1]]["claim"], add_special_tokens=False)

        def invoke(task):
            qid, index, position = task
            doc = by_id[qid]
            if stage == "decomposition":
                context = json.dumps([{"role": "user", "content": doc["question"]}], ensure_ascii=False, indent=2)
                return decompose.decompose_document_into_claims(
                    doc["response"], args.model, context, prompt_path=prompts[stage]
                )
            claim = doc["claims"][index]
            if stage == "checkworthiness":
                return checkworthy.identify_checkworthiness(claim["claim"], args.model, prompt_path=prompts[stage])
            if stage == "retrieval":
                return search_documents(
                    es,
                    args.es_index,
                    body={"query": {"match": {"token_ids": " ".join(map(str, encoded[task]))}}},
                    size=args.num_evidences,
                    max_concurrent_shard_requests=64,
                )
            return verify.verify_claim(
                claim["claim"], claim["evidences"][position]["passage"], args.model, prompt_path=prompts[stage]
            )

        def accept(task, value):
            qid, index, position = task
            if stage == "retrieval":
                evidences = []
                for rank, hit in enumerate(value, 1):
                    source = hit["_source"]
                    passage = tokenizer.decode(list(map(int, source["token_ids"].split()))).strip()
                    if passage:
                        evidences.append(
                            {
                                "passage": passage,
                                "dataset": source.get("dataset_name", ""),
                                "training_step": source.get("iteration"),
                                "index": hit.get("_index"),
                                "document_id": hit.get("_id"),
                                "score": hit.get("_score"),
                                "rank": rank,
                            }
                        )
                value = {"evidences": evidences, "retrieval_query_token_ids": encoded[task], "retrieved_at": now()}
            apply_result(by_id[qid], stage, index, position, value)
            write_json(output / "documents" / f"{qid}.json", by_id[qid])

        stage_errors = run_stage(
            stage, tasks, invoke, accept, checkpoint, args.max_concurrency, output / "errors.jsonl"
        )
        errors.extend(stage_errors)
        checkpoint()
        if stage_errors:
            break

    write_atomic(output / "results.jsonl", "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents))
    checkpoint()
    summary = summarize(documents)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if errors or any(doc["status"] != "complete" for doc in selected):
        raise SystemExit("Incomplete fact-check. Successful operations are saved; rerun to retry missing operations.")


if __name__ == "__main__":
    main()
