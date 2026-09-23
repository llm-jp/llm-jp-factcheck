import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_claim_verdicts import aggregate_claim, aggregate_documents


def claim(labels, checkworthy=True):
    return {
        "claim": "Claim",
        "status": "complete",
        "is_checkworthy": checkworthy,
        "retrieval_complete": checkworthy,
        "evidences": [{"rank": rank, "verification": {"label": label}} for rank, label in enumerate(labels, 1)],
    }


class AggregateClaimVerdictsTests(unittest.TestCase):
    def test_one_partial_support_overrides_two_refutations_regardless_of_rank(self):
        labels = ["Partially supported", "Fully refuted", "Inferentially refuted"]
        for order in itertools.permutations(labels):
            with self.subTest(order=order):
                result = aggregate_claim(claim(order))
                self.assertEqual(result["label"], "Supported")
                self.assertTrue(result["support_refute_conflict"])

    def test_inferential_support_and_refutation_are_included(self):
        for label, expected in (("Inferentially supported", "Supported"), ("Inferentially refuted", "Refuted")):
            with self.subTest(label=label):
                self.assertEqual(
                    aggregate_claim(claim(["Not enough information", label, "Not enough information"]))["label"],
                    expected,
                )

    def test_only_three_nei_verdicts_produce_insufficient_evidence(self):
        self.assertEqual(aggregate_claim(claim(["Not enough information"] * 3))["label"], "Insufficient evidence")
        with self.assertRaises(ValueError):
            aggregate_claim(claim(["Not enough information"] * 2))

    def test_skipped_claims_are_separate_from_insufficient_evidence_and_keep_identity(self):
        documents = [
            {
                "qid": "q1",
                "status": "complete",
                "decomposition_complete": True,
                "claims": [claim([], False), claim(["Not enough information"] * 3)],
            }
        ]
        records, summary = aggregate_documents(documents)
        self.assertIsNone(records[0]["label"])
        self.assertEqual([(row["qid"], row["claim_index"]) for row in records], [("q1", 0), ("q1", 1)])
        self.assertEqual(summary["checkworthy"], 1)
        self.assertEqual(summary["not_checkworthy"], 1)
        self.assertEqual(summary["labels"]["Insufficient evidence"], 1)

    def test_duplicate_ranks_and_missing_or_unknown_verdicts_fail(self):
        duplicate = claim(["Fully supported"] * 3)
        duplicate["evidences"][2]["rank"] = 1
        missing = claim(["Fully supported"] * 3)
        missing["evidences"][2].pop("verification")
        unknown = claim(["Fully supported", "Fully supported", "unknown"])
        for invalid in (duplicate, missing, unknown):
            with self.subTest(claim=invalid), self.assertRaises(ValueError):
                aggregate_claim(invalid)


if __name__ == "__main__":
    unittest.main()
