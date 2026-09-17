import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verdict_metrics import evaluate_verdicts, normalize_label
from verify import VERIFICATION_LABELS


class VerdictMetricTests(unittest.TestCase):
    def test_refutation_spellings_are_normalized_without_collapsing_inference(self):
        self.assertEqual(normalize_label("完全矛盾"), normalize_label("完全否定"))
        self.assertEqual(normalize_label("推定矛盾"), normalize_label("推定否定"))
        self.assertNotEqual(normalize_label("完全矛盾"), normalize_label("推定矛盾"))
        with self.assertRaises(ValueError):
            normalize_label("Partially refuted")

    def test_fixed_six_class_macro_differs_from_accuracy_and_includes_errors(self):
        gold = [{"ID": i, "label": "完全支持"} for i in range(10)]
        predictions = [{"ID": i, "label": "完全支持"} for i in range(9)]
        predictions.append({"ID": 9, "label": None, "error": {"type": "BadRequestError"}})
        result = evaluate_verdicts(predictions, gold)
        self.assertEqual(result["accuracy"], 0.9)
        self.assertEqual(result["errors"], 1)
        self.assertAlmostEqual(result["macro"]["precision"], 1 / 6)
        self.assertAlmostEqual(result["macro"]["recall"], 0.9 / 6)
        self.assertAlmostEqual(result["macro"]["f1"], (18 / 19) / 6)
        self.assertEqual(result["confusion"]["counts"][0][-1], 1)

    def test_missing_duplicate_unknown_and_unexplained_null_predictions_fail(self):
        gold = [{"ID": 1, "label": "完全支持"}]
        good = {"ID": 1, "label": "完全支持"}
        for predictions in ([], [good, good], [{"ID": 2, "label": "完全支持"}], [{"ID": 1, "label": None}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_verdicts(predictions, gold)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install the evaluation dependency group.")
    def test_matches_reference_sklearn_metrics_with_unbalanced_classes_and_errors(self):
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        rng = np.random.default_rng(13)
        labels = list(VERIFICATION_LABELS)
        actual = rng.choice(labels, size=400, p=[0.3, 0.09, 0.04, 0.01, 0.01, 0.55]).tolist()
        predicted = rng.choice(labels + ["Error"], size=400).tolist()
        gold = [{"ID": i, "label": label} for i, label in enumerate(actual)]
        predictions = [
            {"ID": i, "label": label} if label != "Error" else {"ID": i, "label": None, "error": "API failure"}
            for i, label in enumerate(predicted)
        ]
        result = evaluate_verdicts(predictions, gold)
        expected = precision_recall_fscore_support(actual, predicted, labels=labels, average="macro", zero_division=0)
        self.assertEqual(result["accuracy"], accuracy_score(actual, predicted))
        for index, key in enumerate(("precision", "recall", "f1")):
            self.assertAlmostEqual(result["macro"][key], expected[index])


if __name__ == "__main__":
    unittest.main()
