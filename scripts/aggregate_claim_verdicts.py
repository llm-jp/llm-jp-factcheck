"""Aggregate three completed passage verdicts per claim, giving support priority."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SUPPORT = ("Fully supported", "Inferentially supported", "Partially supported")
REFUTE = ("Fully refuted", "Inferentially refuted")
NEI = "Not enough information"
AGGREGATED_LABELS = ("Supported", "Refuted", "Insufficient evidence")


def aggregate_claim(claim):
    checkworthy = claim["is_checkworthy"]
    if type(checkworthy) is not bool or claim["status"] != "complete":
        raise ValueError("Claim analysis must be complete.")
    if not checkworthy:
        if claim["evidences"]:
            raise ValueError("Non-check-worthy claims must not have evidence verdicts.")
        evidences = []
    else:
        evidences = sorted(claim["evidences"], key=lambda evidence: evidence["rank"])
        if not claim["retrieval_complete"] or [e["rank"] for e in evidences] != [1, 2, 3]:
            raise ValueError("Expected exactly three retrieved passages with ranks 1, 2, 3.")
    passages = []
    for evidence in evidences:
        verdict = evidence.get("verification", {})
        if verdict.get("label") not in (*SUPPORT, *REFUTE, NEI):
            raise ValueError("Missing or unknown passage verdict.")
        passages.append(
            {"rank": evidence["rank"], "label": verdict["label"], "document_id": evidence.get("document_id")}
        )
    support_ranks = [p["rank"] for p in passages if p["label"] in SUPPORT]
    refute_ranks = [p["rank"] for p in passages if p["label"] in REFUTE]
    label = None
    if checkworthy:
        label = "Supported" if support_ranks else "Refuted" if refute_ranks else "Insufficient evidence"
    return {
        "claim": claim["claim"],
        "is_checkworthy": checkworthy,
        "status": "aggregated" if checkworthy else "not_checkworthy",
        "label": label,
        "passage_verdicts": passages,
        "support_ranks": support_ranks,
        "refute_ranks": refute_ranks,
        "support_refute_conflict": bool(support_ranks and refute_ranks),
    }


def aggregate_documents(documents):
    if not documents or len({doc["qid"] for doc in documents}) != len(documents):
        raise ValueError("Expected nonempty input with unique question IDs.")
    records = []
    for doc in documents:
        if doc["status"] != "complete" or not doc["decomposition_complete"]:
            raise ValueError("All input answers must have complete fact-check results.")
        for index, claim in enumerate(doc["claims"]):
            records.append({"qid": doc["qid"], "claim_index": index, **aggregate_claim(claim)})
    labels = Counter(row["label"] for row in records if row["is_checkworthy"])
    summary = {
        "documents": len(documents),
        "claims": len(records),
        "checkworthy": sum(labels.values()),
        "not_checkworthy": sum(not row["is_checkworthy"] for row in records),
        "passage_verdicts": sum(len(row["passage_verdicts"]) for row in records),
        "labels": {label: labels[label] for label in AGGREGATED_LABELS},
        "support_refute_conflicts": sum(row["support_refute_conflict"] for row in records),
    }
    assert summary["checkworthy"] + summary["not_checkworthy"] == summary["claims"]
    assert summary["passage_verdicts"] == 3 * summary["checkworthy"]
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve().parent:
        parser.error("Use a separate output directory to preserve the passage-level results.")
    source_bytes = args.input.read_bytes()
    # LF separates JSONL records; Unicode line separators may occur inside passages.
    documents = [json.loads(line) for line in source_bytes.split(b"\n") if line.strip()]
    records, summary = aggregate_documents(documents)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unit": "One existing claim, identified by question ID and zero-based claim index.",
        "passages_per_checkworthy_claim": 3,
        "support_labels": list(SUPPORT),
        "refute_labels": list(REFUTE),
        "insufficient_evidence_label": NEI,
        "rule": "Any support => Supported; else any refutation => Refuted; else Insufficient evidence.",
        "not_checkworthy": "Excluded from label counts; retained with a null aggregate label.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("protocol", protocol), ("summary", summary)):
        (args.output / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "claims.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    total = summary["checkworthy"]
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count", "proportion_of_checkworthy_claims"])
        for label, count in summary["labels"].items():
            writer.writerow([label, count, count / total if total else ""])
    report = [
        "# Claimレベルの集約結果",
        "",
        f"対象：{summary['documents']:,}回答、全{summary['claims']:,} claims。",
        f"集約対象はcheck-worthyな{total:,} claims、元の判定は{summary['passage_verdicts']:,}ペアです。",
        f"check-worthyでない{summary['not_checkworthy']:,} claimsは集約・割合の分母から除外しています。",
        "",
        "完全支持・推論的支持・部分支持をsupport、完全反証・推論的反証をrefutationとして扱います。",
        "3件のうちsupportが1件でもあればSupported、なければrefutationが1件でもあればRefuted、",
        "どちらもなければInsufficient evidenceに集約します。多数決は行いません。",
        "",
        "| 集約ラベル | claim数 | 割合 |",
        "|---|---:|---:|",
    ]
    for label, count in summary["labels"].items():
        percentage = f"{count / total:.2%}" if total else "—"
        report.append(f"| {label} | {count:,} | {percentage} |")
    report.extend(
        [
            f"| **合計** | **{total:,}** | **{'100%' if total else '—'}** |",
            "",
            f"supportとrefutationが混在した{summary['support_refute_conflicts']:,} claimsはSupportedに含まれます。",
            "",
            "[全claimの集約結果](claims.jsonl)／[集計CSV](summary.csv)／[集計条件・入力ハッシュ](protocol.json)",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
