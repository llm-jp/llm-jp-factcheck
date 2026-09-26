import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("evaluate_verification", ROOT / "scripts/evaluate_verification.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class VerificationEvaluationTests(unittest.TestCase):
    def test_input_question_evidence_is_excluded_but_retrieved_passages_are_retained(self):
        source = {"revision": "pinned", "test_source": "test.json", "question_source": "questions.json"}
        rows = [
            {"ID": 1, "textID": "a", "claim": "Claim", "evidence": "Question ", "label": "不明"},
            {"ID": 2, "textID": "a", "claim": "Claim", "evidence": "Retrieved", "label": "完全矛盾"},
        ]
        with patch.object(
            evaluation, "read_git", side_effect=[json.dumps(rows), json.dumps([{"textId": "a", "prompt": "Question"}])]
        ):
            gold, excluded = evaluation.load_cohort(Path("repo"), source)
        self.assertEqual(excluded, [1])
        self.assertEqual([row["ID"] for row in gold], [2])
        self.assertEqual(gold[0]["label"], "Fully refuted")

    def test_resume_discards_only_an_incomplete_final_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            complete = b'{"ID": 1, "label": "Fully supported"}\n'
            path.write_bytes(complete + b'{"ID": 2, "la')
            self.assertEqual(evaluation.read_predictions(path), [{"ID": 1, "label": "Fully supported"}])
            self.assertEqual(path.read_bytes(), complete)
            path.write_bytes(b"{broken}\n")
            with self.assertRaises(json.JSONDecodeError):
                evaluation.read_predictions(path)


if __name__ == "__main__":
    unittest.main()
