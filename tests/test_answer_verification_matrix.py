import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_answer_verification_matrix import join_records, stratify


class AnswerVerificationMatrixTests(unittest.TestCase):
    def inputs(self):
        answer = {
            "qid": "q1",
            "question": "Q",
            "response": "A",
            "status": "complete",
            "reference_answers": ["gold"],
            "selection": {"claim_index": 1, "claim": "chosen", "rationale": "r"},
            "correctness": {"label": "mismatch", "answer_text": "A", "rationale": "r"},
        }

        def claim(text, label):
            return {
                "claim": text,
                "is_checkworthy": True,
                "retrieval_complete": True,
                "evidences": [
                    {
                        "passage": "evidence",
                        "verification": {"label": label, "rationale": "r"},
                    }
                ],
            }

        fact = {
            "qid": "q1",
            "question": "Q",
            "response": "A",
            "status": "complete",
            "claims": [claim("other", "Fully refuted"), claim("chosen", "Fully supported")],
        }
        return answer, fact

    def test_join_uses_exact_selected_claim_not_another_claim_verdict(self):
        answer, fact = self.inputs()
        row = join_records([answer], [fact])[0]
        self.assertEqual(row["verification"], "Fully supported")
        self.assertEqual(row["correctness"], "mismatch")
        invalid = copy.deepcopy(answer)
        invalid["selection"]["claim"] = "fabricated"
        with self.assertRaises(ValueError):
            join_records([invalid], [fact])

    def test_not_checkworthy_and_missing_claim_are_not_nei(self):
        answer, fact = self.inputs()
        fact["claims"][1].update(is_checkworthy=False, evidences=[])
        self.assertEqual(join_records([answer], [fact])[0]["verification"], "Not checkworthy")
        answer["selection"].update(claim_index=None, claim=None)
        answer["correctness"]["label"] = "no_claim"
        self.assertEqual(join_records([answer], [fact])[0]["verification"], "No selected claim")

    def test_random_samples_are_reproducible_order_independent_and_capped(self):
        rows = [{"qid": f"q{i:03}", "correctness": "match", "verification": "Fully supported"} for i in range(35)]
        rows += [{"qid": f"x{i}", "correctness": "mismatch", "verification": "Fully refuted"} for i in range(3)]
        _, samples = stratify(rows)
        self.assertEqual(samples, stratify(list(reversed(rows)))[1])
        self.assertEqual(len(samples), 23)
        self.assertEqual(len({r["qid"] for r in samples}), 23)
        self.assertEqual(sum(r["correctness"] == "mismatch" for r in samples), 3)
        self.assertNotEqual(samples, stratify(rows, seed=42)[1])


if __name__ == "__main__":
    unittest.main()
