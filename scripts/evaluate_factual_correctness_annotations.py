"""Compare completed human labels with the frozen automatic correctness labels."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evaluations/aio_02_factual_correctness_human_eval_30"
LABELS = ("match", "mismatch", "undetermined")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes, total):
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    folder = args.input_dir.resolve()
    output = args.output_dir or folder / "human_evaluation"
    manifest = json.loads((folder / "manifest.json").read_text())
    cases = read_csv(folder / "cases.csv")
    keys = read_csv(folder / "model_judgments.csv")
    with (folder / "cases.jsonl").open(encoding="utf-8") as handle:
        original = {r["qid"]: r for r in map(json.loads, handle)}
    assert digest(folder / "cases.jsonl") == manifest["output_sha256"]["cases.jsonl"]
    assert digest(folder / "model_judgments.csv") == manifest["output_sha256"]["model_judgments.csv"]
    assert digest(Path(manifest["source"])) == manifest["source_sha256"]
    assert len(cases) == len(keys) == len(original) == manifest["sample_size"]
    assert len({r["qid"] for r in cases}) == len(cases)
    assert len({r["qid"] for r in keys}) == len(keys)
    assert set(original) == {r["qid"] for r in cases} == {r["qid"] for r in keys}
    auto = {r["qid"]: r for r in keys}
    comparison = []
    normalizations = []
    for case in cases:
        qid = case["qid"]
        source = original[qid]
        assert int(case["sample_id"]) == int(auto[qid]["sample_id"]) == source["sample_id"]
        assert int(case["selected_claim_index"]) == source["selected_claim_index"]
        for field in ("question", "selected_claim", "response"):
            assert case[field] == source[field], (qid, field, "Non-annotation field changed")
        assert json.loads(case["reference_answers"]) == source["reference_answers"]
        raw = case["human_label"]
        normalized = raw.strip().lower()
        normalized = {"unmatch": "mismatch"}.get(normalized, normalized)
        if normalized not in LABELS:
            raise ValueError(f"Missing or unknown human_label for {qid}: {raw!r}")
        assert auto[qid]["auto_label"] in LABELS
        if raw != normalized:
            normalizations.append({"qid": qid, "raw": raw, "normalized": normalized})
        comparison.append({**case, "human_label_normalized": normalized,
                           **{k: v for k, v in auto[qid].items() if k not in ("sample_id", "qid")},
                           "agrees": normalized == auto[qid]["auto_label"]})
    n = len(comparison)
    agrees = sum(r["agrees"] for r in comparison)
    disagreements = [r for r in comparison if not r["agrees"]]
    human_counts = Counter(r["human_label_normalized"] for r in comparison)
    auto_counts = Counter(r["auto_label"] for r in comparison)
    matrix = {h: {a: sum(r["human_label_normalized"] == h and r["auto_label"] == a
                         for r in comparison) for a in LABELS} for h in LABELS}
    assert sum(sum(row.values()) for row in matrix.values()) == n
    assert sum(matrix[label][label] for label in LABELS) == agrees
    observed = agrees / n
    expected = sum(human_counts[label] * auto_counts[label] for label in LABELS) / n**2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    ci = wilson_interval(agrees, n)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_standard": "User's human annotations; no adjudication or relabeling.",
        "n": n, "agreements": agrees, "disagreements": len(disagreements),
        "accuracy_against_human": observed, "wilson_95_ci_binomial_approximation": ci,
        "cohen_kappa": kappa, "human_counts": dict(human_counts), "automatic_counts": dict(auto_counts),
        "matrix_rows_human_columns_automatic": matrix,
        "label_normalization": {"unmatch": "mismatch"}, "normalizations": normalizations,
        "disagreement_qids": [r["qid"] for r in disagreements],
        "sampling_population": manifest["population_size"], "sampling_seed": manifest["seed"],
        "non_annotation_fields_unchanged": True,
        "human_rationales_provided": sum(bool(r["human_rationale"].strip()) for r in comparison),
        "selection_issue_notes_provided": sum(bool(r["selection_issue"].strip()) for r in comparison),
        "source_sha256": {name: digest(folder / name)
                          for name in ("cases.csv", "cases.jsonl", "model_judgments.csv", "manifest.json")},
        "script_sha256": digest(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    fields = list(comparison[0])
    write_csv(output / "comparison.csv", comparison, fields)
    write_csv(output / "disagreements.csv", disagreements, fields)
    matrix_rows = [{"human_label": h, **matrix[h], "total": sum(matrix[h].values())} for h in LABELS]
    write_csv(output / "confusion_matrix.csv", matrix_rows, ["human_label", *LABELS, "total"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Factual correctness 自動判定の人手評価", "",
        f"人手判定を基準とする自動判定の一致率（accuracy）は **{agrees}/{n} = {observed:.2%}**。"
        f"不一致は{len(disagreements)}件。95% Wilson信頼区間は{ci[0]:.2%}–{ci[1]:.2%}（二項近似）。", "",
        f"Cohenのκは **{kappa:.4f}**。" if kappa is not None else "Cohenのκは算出不能。", "",
        "## 混同行列", "",
        "行は人手判定、列は自動判定。match=正解、mismatch=不正解、undetermined=判定不能。", "",
        "| 人手判定＼自動判定 | match | mismatch | undetermined | 合計 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in matrix_rows:
        lines.append(f"| {row['human_label']} | {row['match']} | {row['mismatch']} | {row['undetermined']} | {row['total']} |")
    lines.extend([
        f"| 合計 | {auto_counts['match']} | {auto_counts['mismatch']} | {auto_counts['undetermined']} | {n} |", "",
        "## 不一致事例", "",
    ])
    for r in disagreements:
        lines.extend([
            f"### {r['qid']}（sample_id={r['sample_id']}）", "",
            "質問：" + r["question"], "", "対象claim：" + r["selected_claim"], "",
            "参照正解：" + " / ".join(json.loads(r["reference_answers"])), "",
            f"人手：**{r['human_label']} → {r['human_label_normalized']}**。自動：**{r['auto_label']}**。", "",
            "自動判定の理由：" + r["auto_rationale"], "",
            "人手判定の理由：" + (r["human_rationale"] or "未記入。人手側の判断理由は推定しない。"), "",
        ])
        if r["qid"] == "AIO02-0610":
            assert r["auto_label"] == "match" and r["human_label_normalized"] == "mismatch"
            lines.extend([
                "自動判定はエンドオブライフケアとターミナルケアを同義と扱っている。人手ラベルを基準にすると、誤ってmatchにした1件に当たる。用語の同義性そのものを、この照合で別途再判定したわけではない。", "",
                "元回答には「ターミナルケア」という語も含まれるが、対象claimには「エンドオブライフケア」が選ばれている。今回の精度は選択claimへの判定について集計しており、元回答全体への採点とは区別する。", "",
            ])
    lines.extend([
        "## 集計条件", "",
        f"- 母集団はclaim抽出済み{manifest['population_size']}件。seed={manifest['seed']}で無作為抽出した{n}件すべてに人手ラベルが記入済み。",
        f"- 人手ラベルのunmatchをmismatchに正規化したものは{len(normalizations)}件。入力cases.csvは変更していない。",
        "- 質問・claim・参照正解・元回答・sample_idの一致を抽出時のデータと照合済み。自動判定ファイルと元の評価結果のハッシュも確認済み。",
        "- 一致率は自動採点器の人手判定への一致を表す。回答生成モデル自体の正答率とは異なる。",
        "- 30件の標本での結果であり、母集団全体で同じ精度と確定したものではない。claim抽出そのものの精度、claim未抽出23件、verificationの精度はこの集計の対象外。", "",
        "[全30件の照合CSV](comparison.csv) · [不一致のみ](disagreements.csv) · [集計JSON](summary.json) · [混同行列CSV](confusion_matrix.csv)", "",
    ])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    assert digest(folder / "cases.csv") == summary["source_sha256"]["cases.csv"]
    assert len(read_csv(output / "comparison.csv")) == n
    assert len(read_csv(output / "disagreements.csv")) == len(disagreements)
    print(json.dumps({k: summary[k] for k in ("n", "agreements", "disagreements", "accuracy_against_human", "wilson_95_ci_binomial_approximation", "cohen_kappa", "matrix_rows_human_columns_automatic", "disagreement_qids")}, ensure_ascii=False, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
