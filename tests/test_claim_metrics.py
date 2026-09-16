import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_metrics import ContentWords, evaluate_documents, jaccard, match_claims, reference_assignment

HAS_EVALUATION = all(importlib.util.find_spec(name) for name in ("MeCab", "unidic_lite", "scipy"))


@unittest.skipUnless(HAS_EVALUATION, "Install the evaluation dependency group.")
class ClaimMetricTests(unittest.TestCase):
    def test_duplicate_predictions_cannot_reuse_gold(self):
        result = match_claims(["A", "A"], ["A"], lambda a, b: float(a == b), 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_assignment_maximizes_similarity_before_thresholding(self):
        matrix = [[1.0, 0.8], [0.8, 0.7]]
        result = match_claims([0, 1], [0, 1], lambda a, b: matrix[a][b], 0.8)
        self.assertEqual(result["correct"], 1)
        self.assertEqual([(pair["prediction"], pair["gold"]) for pair in result["pairs"]], [(0, 0), (1, 1)])

    def test_threshold_is_inclusive_as_in_reference_code(self):
        result = match_claims(["A"], ["B"], lambda a, b: 0.8, 0.8)
        self.assertEqual(result["f1"], 1)

    def test_equal_weight_assignments_preserve_reference_tie_resolution(self):
        matrix = [[0, 0.5], [0.5, 1]]
        result = match_claims([0, 1], [0, 1], lambda a, b: matrix[a][b], 0.8)
        self.assertEqual(result["correct"], 1)

    def test_reference_assignment_is_optimal_for_rectangular_matrices(self):
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        rng = np.random.default_rng(0)
        for rows, columns in [(1, 5), (5, 1), (4, 4), (4, 7), (7, 4)]:
            for _ in range(20):
                matrix = rng.integers(0, 10, size=(rows, columns)).tolist()
                original = [row[:] for row in matrix]
                rr, cc = reference_assignment(matrix)
                actual = sum(matrix[r][c] for r, c in zip(rr, cc) if r < rows and c < columns)
                rr, cc = linear_sum_assignment(matrix, maximize=True)
                expected = sum(matrix[r][c] for r, c in zip(rr, cc))
                self.assertEqual(actual, expected)
                self.assertEqual(matrix, original)

    def test_empty_claims_score_zero(self):
        for predictions, gold in [([], []), ([], ["A"]), (["A"], [])]:
            result = match_claims(predictions, gold, lambda a, b: 1, 1)
            self.assertEqual((result["precision"], result["recall"], result["f1"]), (0, 0, 0))

    def test_exact_match_does_not_normalize_gold_whitespace(self):
        result = match_claims(["Claim"], ["Claim "], lambda a, b: float(a == b), 1)
        self.assertEqual(result["correct"], 0)

    def test_content_words_exclude_particles_and_use_reference_forms(self):
        words = ContentWords()
        self.assertEqual(words("猫は走った。"), frozenset({"猫", "走る"}))
        self.assertEqual(jaccard(words("猫は走った。"), words("猫が走る。")), 1)
        self.assertEqual(jaccard(words("。"), words("。")), 0)

    @staticmethod
    def document(identifier, claims):
        return {"textId": identifier, "text": "Original " + identifier, "claims": [{"text": c} for c in claims]}

    def test_macro_average_is_over_documents_including_empty_predictions(self):
        gold = [self.document("1", ["A"]), self.document("2", ["B", "C", "D"])]
        predictions = [self.document("2", []), self.document("1", ["A"])]
        result = evaluate_documents(predictions, gold)
        self.assertEqual(result["macro"]["exact"], {"precision": 0.5, "recall": 0.5, "f1": 0.5})

    def test_dataset_coverage_and_input_text_are_checked(self):
        row = self.document("1", ["A"])
        for predictions in ([], [row, row], [self.document("2", ["A"])], [dict(row, text="Changed")]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_documents(predictions, [row])


if __name__ == "__main__":
    unittest.main()
