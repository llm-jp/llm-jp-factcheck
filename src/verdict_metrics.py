"""Six-class verdict metrics with the paper's fixed-label macro averaging."""

from __future__ import annotations

from statistics import mean

from verify import MODEL_LABELS, VERIFICATION_LABELS

GOLD_LABELS = {**MODEL_LABELS, "完全矛盾": "Fully refuted", "推定矛盾": "Inferentially refuted"}


def normalize_label(label: str) -> str:
    if label in VERIFICATION_LABELS:
        return label
    if label in GOLD_LABELS:
        return GOLD_LABELS[label]
    raise ValueError(f"Unknown verdict label: {label!r}")


def evaluate_verdicts(predictions: list[dict], gold: list[dict]) -> dict:
    """Score every pair, treating explicit failed predictions as incorrect."""
    by_id = {row["ID"]: row for row in predictions}
    gold_ids = {row["ID"] for row in gold}
    if not gold or len(gold_ids) != len(gold):
        raise ValueError("Gold data must have unique IDs and be nonempty.")
    if len(by_id) != len(predictions) or set(by_id) != gold_ids:
        raise ValueError("Predictions must cover every gold ID exactly once.")
    labels = list(VERIFICATION_LABELS)
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    errors = 0
    for row in gold:
        actual = normalize_label(row["label"])
        prediction = by_id[row["ID"]]
        if prediction["label"] is None:
            if not prediction.get("error"):
                raise ValueError("A missing label requires an explicit error record.")
            column = len(labels)
            errors += 1
        else:
            column = labels.index(normalize_label(prediction["label"]))
        matrix[labels.index(actual)][column] += 1
    per_label = {}
    for i, label in enumerate(labels):
        correct = matrix[i][i]
        predicted = sum(row[i] for row in matrix)
        support = sum(matrix[i])
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": support,
        }
    return {
        "count": len(gold),
        "errors": errors,
        "accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(gold),
        "macro": {key: mean(values[key] for values in per_label.values()) for key in ("precision", "recall", "f1")},
        "per_label": per_label,
        "confusion": {"rows": labels, "columns": labels + ["Error"], "counts": matrix},
    }
