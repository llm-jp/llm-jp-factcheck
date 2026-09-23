"""Draw reproducible random cases for manual assessment of answer correctness."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "result/aio_02_dev_v1.0_final_answer_claims_gpt-oss-120b/results.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluations/aio_02_factual_correctness_human_eval_30"
HUMAN_FIELDS = ["human_answer_text", "human_label", "human_rationale", "selection_issue"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def blockquote(value):
    return "\n".join("> " + line for line in value.split("\n"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    # Refuse to overwrite a worksheet that may already contain human annotations.
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new --output-dir.")
    with args.source.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert len({r["qid"] for r in records}) == len(records)
    assert all(r["status"] == "complete" for r in records)
    population = sorted(
        (r for r in records
         if r["selection"]["claim_index"] is not None
         and r["correctness"]["label"] in {"match", "mismatch", "undetermined"}),
        key=lambda r: r["qid"],
    )
    assert all(r["selection"]["claim"] == r["claims"][r["selection"]["claim_index"]]
               for r in population)
    if not 1 <= args.count <= len(population):
        parser.error("--count must be between 1 and the population size.")
    draw = random.Random(args.seed).sample(population, args.count)
    selected = sorted(draw, key=lambda r: r["qid"])
    cases, judgments = [], []
    for number, r in enumerate(selected, 1):
        cases.append({
            "sample_id": number, "qid": r["qid"], "question": r["question"],
            "selected_claim_index": r["selection"]["claim_index"],
            "selected_claim": r["selection"]["claim"],
            "reference_answers": json.dumps(r["reference_answers"], ensure_ascii=False),
            "response": r["response"], **dict.fromkeys(HUMAN_FIELDS, ""),
        })
        judgments.append({
            "sample_id": number, "qid": r["qid"],
            "auto_label": r["correctness"]["label"],
            "auto_answer_text": r["correctness"]["answer_text"],
            "auto_rationale": r["correctness"]["rationale"],
            "selection_rationale": r["selection"]["rationale"],
        })
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "cases.csv", cases)
    write_csv(args.output_dir / "model_judgments.csv", judgments)
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            record = {**case, "reference_answers": json.loads(case["reference_answers"])}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    lines = [
        f"# Factual correctness の人手評価：無作為{args.count}例", "",
        f"母集団はclaim抽出済みの{len(population):,}件（match / mismatch / undetermined）。"
        f"全{len(records):,}回答のうちclaim未抽出など対象外は{len(records) - len(population)}件。"
        f"QID順の母集団からPythonのrandom.Random({args.seed}).sampleで重複なしに{args.count}件を抽出し、提示順はQID順とした。"
        "正誤ラベルやcheck-worthy、verificationのラベルによる層化は行っていない。", "",
        "この評価では、質問・選択claim・参照正解を用いて、claimが問題の答えとして述べている内容を採点する。"
        "元回答は文脈確認用に掲載しているが、自動判定器が受け取ったのは質問・選択claim・参照正解である。"
        "元回答に正しい答えがあっても、claimにない答えを補って採点しない。", "",
        "## 記入方法", "",
        "- **human_answer_text**：対象claimが答えとして示した語句・値をそのまま抜き出す。特定できなければ空欄。",
        "- **human_label**：`match`（参照正解のいずれかと意味的に一致）、`mismatch`（異なる、または必要な項目が不足）、`undetermined`（答えを特定できない、または同一性を判断する情報が不足）のいずれか。",
        "- **human_rationale**：判定理由を簡潔に記入する。",
        "- **selection_issue**：選択claimが質問への答えを表していないなど、抽出側の問題があれば記入する。なければ空欄。", "",
        "表記揺れ・同義語・略称・翻訳名・同じ値の単位換算は許容する。"
        "必要な複数項目が揃わない場合や、誤った候補が未解決のまま併記される場合は一致としない。"
        "問題への答えと無関係な補足説明の誤り、および検索根拠による支持・反証は採点対象に含めない。", "",
        "この評価票は既存のanswer correctness判定の精度を調べるもの。claimの全命題の事実性や、回答全体の事実性の独立評価とは評価単位が異なる。", "",
        "[記入用CSV](cases.csv)には元回答も全文収録している。自動判定と理由は[照合用CSV](model_judgments.csv)に分離した。"
        "人手判定を記入した後、sample_idまたはqidで照合できる。", "",
    ]
    for case in cases:
        lines.extend([
            f"## {case['sample_id']:02d}. {case['qid']}", "",
            "**質問**：" + case["question"], "", "**対象claim**：", "",
            blockquote(case["selected_claim"]), "",
            "**参照正解**：" + " / ".join(json.loads(case["reference_answers"])), "",
            f"claim index：{case['selected_claim_index']}（0始まり）。", "",
            "**元回答（全文・文脈確認用）**：", "", blockquote(case["response"]), "",
            "- human_answer_text：", "- human_label：", "- human_rationale：", "- selection_issue：", "",
        ])
    (args.output_dir / "cases.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()), "source_sha256": sha256(args.source),
        "script_sha256": sha256(Path(__file__)),
        "seed": args.seed, "sample_size": args.count, "population_size": len(population),
        "input_answers": len(records),
        "population_definition": "Complete records with a selected claim and label match, mismatch, or undetermined.",
        "excluded_by_label": dict(Counter(r["correctness"]["label"] for r in records if r not in population)),
        "sampling": "Simple random sample without replacement; population sorted by qid; random.Random(seed).sample.",
        "stratified": False, "display_order": "qid ascending",
        "random_draw_order": [r["qid"] for r in draw],
        "display_order_qids": [r["qid"] for r in selected],
        "human_fields": HUMAN_FIELDS,
        "automatic_judgments_file": "model_judgments.csv",
        "verification_labels_used_for_sampling": False,
        "output_sha256": {name: sha256(args.output_dir / name)
                          for name in ("cases.csv", "cases.md", "cases.jsonl", "model_judgments.csv")},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Verify the actual worksheet files, including multiline CSV round-tripping.
    with (args.output_dir / "cases.csv").open(encoding="utf-8-sig", newline="") as handle:
        readback = list(csv.DictReader(handle))
    assert len(readback) == len({r["qid"] for r in readback}) == args.count
    for row, source in zip(readback, selected):
        assert row["qid"] == source["qid"]
        assert row["response"] == source["response"]
        assert row["selected_claim"] == source["selection"]["claim"]
        assert json.loads(row["reference_answers"]) == source["reference_answers"]
        assert all(row[field] == "" for field in HUMAN_FIELDS)
        assert not any(field.startswith("auto_") for field in row)
    assert [r["qid"] for r in random.Random(args.seed).sample(population, args.count)] == manifest["random_draw_order"]
    print(f"Population: {len(population)}; sample: {len(selected)}; seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    for r in selected:
        print(r["qid"], r["question"])


if __name__ == "__main__":
    main()
