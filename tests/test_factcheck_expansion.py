import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from factcheck_answers import apply_result, pending_tasks
from prepare_factcheck_expansion import reset_evidence


class FactcheckExpansionTests(unittest.TestCase):
    def test_reuse_retains_claim_identity_but_discards_old_evidence_and_verdicts(self):
        source = {
            "qid": "q1",
            "question": "Q",
            "response": "A",
            "chatbot_model": "chat",
            "factchecker_model": "gpt-oss-120b",
            "decomposition_complete": True,
            "status": "complete",
            "claims": [
                {
                    "claim": "first",
                    "is_checkworthy": True,
                    "retrieval_complete": True,
                    "retrieval_query_token_ids": [1],
                    "retrieved_at": "old",
                    "evidences": [{"verification": {}}],
                },
                {"claim": "second", "is_checkworthy": False, "evidences": []},
            ],
        }
        before = copy.deepcopy(source)
        result = reset_evidence(source)
        self.assertEqual(source, before)
        self.assertEqual([c["claim"] for c in result["claims"]], ["first", "second"])
        self.assertEqual(pending_tasks([result], "decomposition"), [])
        self.assertEqual(pending_tasks([result], "checkworthiness"), [])
        self.assertEqual(pending_tasks([result], "retrieval"), [("q1", 0, None)])
        self.assertEqual(pending_tasks([result], "verification"), [])
        self.assertNotIn("retrieved_at", result["claims"][0])
        self.assertNotIn("retrieval_query_token_ids", result["claims"][0])
        self.assertFalse(result["claims"][1]["is_checkworthy"])
        evidence = [{"rank": rank, "passage": f"passage {rank}"} for rank in (1, 2, 3)]
        apply_result(result, "retrieval", 0, None, {"evidences": evidence})
        self.assertEqual(pending_tasks([result], "verification"), [("q1", 0, 0), ("q1", 0, 1), ("q1", 0, 2)])
        apply_result(result, "verification", 0, 1, {"label": "Fully refuted", "rationale": "Second passage."})
        self.assertEqual(pending_tasks([result], "verification"), [("q1", 0, 0), ("q1", 0, 2)])
        self.assertNotIn("verification", result["claims"][0]["evidences"][0])
        self.assertEqual(result["claims"][0]["evidences"][1]["verification"]["label"], "Fully refuted")
        self.assertNotIn("verification", result["claims"][0]["evidences"][2])

    def test_incomplete_analysis_cannot_be_reused(self):
        with self.assertRaises(ValueError):
            reset_evidence({"status": "checkworthiness", "decomposition_complete": True})


if __name__ == "__main__":
    unittest.main()
