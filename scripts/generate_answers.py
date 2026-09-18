"""Generate resumable answers to a question JSONL using the CHATBOT connection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clients import get_client


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for responses and generation metadata.")
    parser.add_argument("--model", default=os.getenv("CHATBOT_MODEL", "").strip())
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--limit", type=int, help="Process only the first N questions; a later run can resume the rest."
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("Set --model or CHATBOT_MODEL.")
    if (
        args.max_tokens < 1
        or args.max_concurrency < 1
        or not math.isfinite(args.request_timeout)
        or args.request_timeout <= 0
        or not math.isfinite(args.temperature)
        or not 0 <= args.temperature <= 2
        or not math.isfinite(args.top_p)
        or not 0 < args.top_p <= 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Counts and timeout must be positive; temperature must be in [0, 2] and top-p in (0, 1].")
    rows = read_jsonl(args.input)
    if not rows or any(
        not isinstance(row.get(key), str) or not row[key].strip() for row in rows for key in ("qid", "question")
    ):
        raise ValueError("Input must contain nonempty qid and question strings.")
    source = {row["qid"]: row for row in rows}
    if len(source) != len(rows):
        raise ValueError("Duplicate qid in input.")

    protocol = {
        "input_file": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "questions": len(rows),
        "model": args.model,
        "messages": [{"role": "user", "content": "{{question}}"}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "other_sampling_parameters": "provider defaults",
        "connection": "CHATBOT_* from environment and project-root .env",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    responses_path = args.output / "responses.jsonl"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Generation settings or input changed. Use a new output directory.")
    elif responses_path.exists():
        raise ValueError("Existing responses have no protocol; use a new output directory.")
    else:
        write_json(protocol_path, protocol)

    saved = read_jsonl(responses_path) if responses_path.exists() else []
    by_id = {row["qid"]: row for row in saved}
    if len(by_id) != len(saved) or not by_id.keys() <= source.keys():
        raise ValueError("Saved responses contain duplicate or unknown IDs.")
    for qid, row in by_id.items():
        if (
            row.get("question") != source[qid]["question"]
            or row.get("model") != args.model
            or row.get("finish_reason") not in ("stop", "length")
            or not isinstance(row.get("response"), str)
            or not row["response"].strip()
        ):
            raise ValueError(f"Invalid saved response for {qid}.")

    selected = rows[: args.limit] if args.limit else rows
    pending = [row for row in selected if row["qid"] not in by_id]
    client = get_client("chatbot").with_options(timeout=args.request_timeout, max_retries=2) if pending else None

    def generate(row: dict) -> dict:
        started = monotonic()
        completion = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": row["question"]}],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        if len(completion.choices) != 1:
            raise ValueError("Unexpected number of completion choices.")
        choice = completion.choices[0]
        content = choice.message.content
        if choice.finish_reason not in ("stop", "length") or not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty or unsuccessful response (finish_reason={choice.finish_reason}).")
        return {
            "qid": row["qid"],
            "question": row["question"],
            "response": content,
            "model": args.model,
            "response_model": completion.model,
            "finish_reason": choice.finish_reason,
            "usage": completion.usage.model_dump() if completion.usage else None,
            "completion_id": completion.id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(monotonic() - started, 3),
        }

    errors = []
    started = monotonic()
    print(f"Saved: {len(by_id)}/{len(rows)}; pending in this run: {len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        queue = iter(pending)
        futures = {}

        def fill_queue() -> None:
            while len(futures) < args.max_concurrency:
                row = next(queue, None)
                if row is None:
                    return
                futures[executor.submit(generate, row)] = row

        fill_queue()
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                row = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    # Do not log provider error bodies, which can include connection details.
                    error = {"qid": row["qid"], "type": type(exc).__name__}
                    if isinstance(exc, ValueError):
                        error["message"] = str(exc)
                    if getattr(exc, "status_code", None) is not None:
                        error["status_code"] = exc.status_code
                    errors.append(error)
                    print(f"Generation failed: {json.dumps(error)}", flush=True)
                    continue
                by_id[row["qid"]] = result
                write_atomic(
                    responses_path,
                    "".join(
                        json.dumps(by_id[item["qid"]], ensure_ascii=False) + "\n"
                        for item in rows
                        if item["qid"] in by_id
                    ),
                )
                print(f"Saved {len(by_id)}/{len(rows)} ({row['qid']}, {result['elapsed_seconds']:.1f}s)", flush=True)
            # After a failure, preserve in-flight successes but submit no additional requests.
            if not errors:
                fill_queue()

    write_json(
        args.output / "summary.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "questions": len(rows),
            "completed": len(by_id),
            "remaining": len(rows) - len(by_id),
            "complete": len(by_id) == len(rows),
            "finish_reasons": {
                reason: sum(row["finish_reason"] == reason for row in by_id.values()) for reason in ("stop", "length")
            },
            "elapsed_seconds_this_run": round(monotonic() - started, 3),
            "errors_this_run": errors,
            "usage": {
                key: sum((row.get("usage") or {}).get(key, 0) or 0 for row in by_id.values())
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        },
    )
    if errors:
        raise SystemExit("Generation stopped after an error; completed responses are saved. Rerun to resume.")
    print(f"Finished: {len(by_id)}/{len(rows)} saved to {responses_path}", flush=True)


if __name__ == "__main__":
    main()
