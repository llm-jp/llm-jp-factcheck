"""Reuse completed claim analysis in a separate fact-check run with more evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from factcheck_answers import refresh_status, sha256, summarize
from generate_answers import read_jsonl, write_atomic, write_json


def reset_evidence(document):
    """Keep claim identities and check-worthiness; discard all old retrieval/verdict state."""
    if document.get("status") != "complete" or not document.get("decomposition_complete"):
        raise ValueError("Source claim analysis must be complete.")
    result = {
        key: copy.deepcopy(document[key])
        for key in (
            "qid",
            "question",
            "response",
            "chatbot_model",
            "factchecker_model",
            "decomposition_complete",
        )
    }
    result["claims"] = []
    for claim in document["claims"]:
        if type(claim.get("is_checkworthy")) is not bool:
            raise ValueError("Source check-worthiness is incomplete.")
        result["claims"].append(
            {
                "claim": claim["claim"],
                "is_checkworthy": claim["is_checkworthy"],
                "retrieval_complete": False,
                "no_evidence": False,
                "evidences": [],
            }
        )
    refresh_status(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-evidences", type=int, default=3)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Output already exists. Resume it with factcheck_answers.py, or choose a new directory.")
    protocol = json.loads((source / "protocol.json").read_text())
    if args.num_evidences <= protocol["num_evidences"]:
        raise ValueError("Expansion must request more evidence than the source run.")
    input_path = Path(protocol["input_file"])
    if sha256(input_path.read_bytes()) != protocol["input_sha256"]:
        raise ValueError("Original chatbot input changed.")
    rows = read_jsonl(source / "results.jsonl")
    inputs = read_jsonl(input_path)
    if len(rows) != len(inputs) or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Input row count or unique IDs do not match.")
    for doc, original in zip(rows, inputs):
        if any(doc[key] != original[key] for key in ("qid", "question", "response")):
            raise ValueError("Source result does not match the chatbot input.")
        if doc != json.loads((source / "documents" / f"{doc['qid']}.json").read_text()):
            raise ValueError("Source consolidated results differ from checkpoints.")
    documents = [reset_evidence(doc) for doc in rows]
    for stage, expected in protocol["prompt_sha256"].items():
        if sha256((source / "prompts" / f"{stage}.yaml").read_bytes()) != expected:
            raise ValueError("Source prompt snapshot changed.")
    protocol["num_evidences"] = args.num_evidences
    (output / "documents").mkdir(parents=True)
    (output / "prompts").mkdir()
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "reuse_provenance.json",
        {
            "source_directory": str(source),
            "source_protocol_sha256": sha256((source / "protocol.json").read_bytes()),
            "source_results_sha256": sha256((source / "results.jsonl").read_bytes()),
            "reused": ["decomposition", "checkworthiness"],
            "rerun": ["retrieval", "verification"],
            "verification_unit": "Each claim-evidence pair separately; no concatenation or aggregation.",
        },
    )
    for stage in protocol["prompt_sha256"]:
        name = f"{stage}.yaml"
        (output / "prompts" / name).write_bytes((source / "prompts" / name).read_bytes())
    for doc in documents:
        write_json(output / "documents" / f"{doc['qid']}.json", doc)
    write_atomic(output / "results.jsonl", "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents))
    write_json(output / "summary.json", summarize(documents))
    print(json.dumps(summarize(documents), ensure_ascii=False))


if __name__ == "__main__":
    main()
