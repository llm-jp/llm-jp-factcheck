import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_final_answer_claims import model_input, pending, summary, validate


class FinalAnswerClaimsTests(unittest.TestCase):
    def document(self):
        return {
            "qid": "q1",
            "question": "何県?",
            "response": "京都府です。",
            "claims": ["京都府です。"],
            "reference_answers": ["大阪府"],
        }

    def test_selection_cannot_see_gold_and_preserves_even_wrong_claim(self):
        doc = self.document()
        data = model_input(doc, "selection")
        self.assertEqual(set(data), {"question", "response", "claims"})
        self.assertNotIn("大阪府", json.dumps(data, ensure_ascii=False))
        doc["selection"] = validate({"status": "selected", "claim_index": 0, "rationale": "答え。"}, doc, "selection")
        self.assertEqual(doc["selection"]["claim"], doc["claims"][0])
        self.assertEqual(
            model_input(doc, "correctness"),
            {
                "question": "何県?",
                "selected_claim": "京都府です。",
                "reference_answers": ["大阪府"],
            },
        )

    def test_rejects_fabricated_or_invalid_claim_index(self):
        doc = self.document()
        for index in (-1, 1, True, "0", None):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate({"status": "selected", "claim_index": index, "rationale": "理由"}, doc, "selection")
        with self.assertRaises(ValueError):
            validate({"status": "no_matching_claim", "claim_index": 0, "rationale": "理由"}, doc, "selection")

    def test_missing_claim_is_not_graded_incorrect_and_resume_skips_saved_stages(self):
        doc = self.document()
        doc["claims"] = []
        doc["selection"] = validate(
            {"status": "no_matching_claim", "claim_index": None, "rationale": "空"}, doc, "selection"
        )
        for label in ("match", "mismatch", "undetermined"):
            with self.assertRaises(ValueError):
                validate({"label": label, "answer_text": None, "rationale": "理由"}, doc, "correctness")
        restored = json.loads(json.dumps(doc))
        self.assertEqual(pending([restored], "selection"), [])
        self.assertEqual(pending([restored], "correctness"), [("q1", None, None)])
        restored["correctness"] = validate(
            {"label": "no_claim", "answer_text": None, "rationale": "空"}, restored, "correctness"
        )
        self.assertIsNone(restored["correctness"]["is_correct"])
        self.assertEqual(pending([restored], "correctness"), [])
        report = summary([restored])
        self.assertTrue(report["complete"])
        self.assertEqual(report["labels"]["mismatch"], 0)
        self.assertEqual(report["labels"]["no_claim"], 1)


if __name__ == "__main__":
    unittest.main()
