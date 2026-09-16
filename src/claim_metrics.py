"""Claim matching from Masano et al. (LREC 2026), without an LLM judge."""

from __future__ import annotations

from functools import lru_cache
from statistics import mean

CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}


class ContentWords:
    """Use the reference evaluator's UniDic field 10 and POS filter."""

    def __init__(self):
        import MeCab
        import unidic_lite

        self.tagger = MeCab.Tagger(f'-r /dev/null -d "{unidic_lite.DICDIR}"')

    @lru_cache(maxsize=32768)
    def __call__(self, text: str) -> frozenset[str]:
        node = self.tagger.parseToNode(text)
        words = set()
        while node:
            features = node.feature.split(",")
            if features[0] in CONTENT_POS:
                words.add(features[10] if len(features) > 10 and features[10] != "*" else node.surface)
            node = node.next
        return frozenset(words)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def reference_assignment(scores_matrix: list[list[float]]) -> tuple[list[int], list[int]]:
    """Preserve the reference Hungarian implementation, including its tie resolution.

    Adapted from the pinned experiment revision in decomposition_source.json.
    Zero padding does not mutate the caller's similarity matrix.
    """
    import numpy as np

    size = max(len(scores_matrix), len(scores_matrix[0]))
    matrix = np.zeros((size, size))
    matrix[: len(scores_matrix), : len(scores_matrix[0])] = scores_matrix
    M = matrix.max()
    matrix = M - matrix
    matrix -= matrix.min(axis=1, keepdims=True)
    matrix -= matrix.min(axis=0, keepdims=True)

    n = matrix.shape[0]
    u = np.zeros(n)
    v = np.zeros(n)
    match_v = -np.ones(n, dtype=int)
    for i in range(n):
        min_slack = np.full(n, np.inf)
        from_row = -np.ones(n, dtype=int)
        prev_col = -np.ones(n, dtype=int)
        visited_row = np.zeros(n, dtype=bool)
        visited_col = np.zeros(n, dtype=bool)
        marked_row = i
        visited_row[marked_row] = True
        marked_col = -1
        while True:
            slack = matrix[marked_row] - u[marked_row] - v
            better = slack < min_slack
            min_slack = np.where(better, slack, min_slack)
            prev_col = np.where(better, marked_col, prev_col)
            from_row = np.where(better, marked_row, from_row)

            mask = ~visited_col
            delta = min(min_slack[mask])
            u[visited_row] += delta
            v[visited_col] -= delta
            min_slack[mask] -= delta

            marked_col = np.where(mask & (min_slack == 0))[0][0]
            visited_col[marked_col] = True

            if match_v[marked_col] == -1:
                while marked_col != -1:
                    row = from_row[marked_col]
                    match_v[marked_col] = row
                    marked_col = prev_col[marked_col]
                break
            else:
                marked_row = match_v[marked_col]
                visited_row[marked_row] = True

    row_index = [int(r) for r in match_v if r != -1]
    col_index = [c for c, r in enumerate(match_v) if r != -1]

    return row_index, col_index


def match_claims(predictions: list[str], gold: list[str], similarity, threshold: float) -> dict:
    """Maximize total similarity first, then threshold the one-to-one assignment."""
    pairs = []
    if predictions and gold:
        scores = [[similarity(prediction, target) for target in gold] for prediction in predictions]
        rows, columns = reference_assignment(scores)
        pairs = [
            {"prediction": int(row), "gold": int(column), "similarity": scores[row][column]}
            for row, column in zip(rows, columns)
            if row < len(predictions) and column < len(gold)
        ]
    correct = sum(pair["similarity"] >= threshold for pair in pairs)
    precision = correct / len(predictions) if predictions else 0.0
    recall = correct / len(gold) if gold else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "correct": correct,
        "predicted": len(predictions),
        "gold": len(gold),
        "pairs": pairs,
    }


def evaluate_documents(predictions: list[dict], gold: list[dict]) -> dict:
    """Macro-average document scores; refuse missing, duplicate, or changed inputs."""
    predicted_by_id = {row["textId"]: row for row in predictions}
    gold_ids = {row["textId"] for row in gold}
    if not gold or len(gold_ids) != len(gold):
        raise ValueError("Gold documents must be nonempty and have unique textId values.")
    if len(predicted_by_id) != len(predictions) or set(predicted_by_id) != gold_ids:
        raise ValueError("Predictions must cover every gold textId exactly once.")
    tokenizer = ContentWords()
    documents = []
    for row in gold:
        prediction = predicted_by_id[row["textId"]]
        if prediction["text"] != row["text"]:
            raise ValueError(f"Input text differs for {row['textId']}.")
        predicted_claims = [claim["text"] for claim in prediction["claims"]]
        gold_claims = [claim["text"] for claim in row["claims"]]
        documents.append(
            {
                "textId": row["textId"],
                "exact": match_claims(predicted_claims, gold_claims, lambda a, b: float(a == b), 1.0),
                "fuzzy_content": match_claims(
                    predicted_claims, gold_claims, lambda a, b: jaccard(tokenizer(a), tokenizer(b)), 0.8
                ),
            }
        )
    return {
        "macro": {
            metric: {score: mean(row[metric][score] for row in documents) for score in ("precision", "recall", "f1")}
            for metric in ("exact", "fuzzy_content")
        },
        "documents": documents,
    }
