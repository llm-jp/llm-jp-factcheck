"""Offline fixture behavior and isolation from all real backend services."""

import socket
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mock_backend
from pipeline import PipelineConfig

LABELS = ("Supported", "Partially supported", "Partially refuted", "Refuted", "Not enough information")


class MockBackendTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(mock_backend, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)
        self.messages = [{"role": "user", "content": "Show me a sample.", "id": "user-1"}]

    def response(self, messages=None, model="unused-model"):
        return "".join(mock_backend.stream_mock_chat_response(self.messages if messages is None else messages, model))

    def results(self, document, count=1):
        return [
            event.result
            for event in mock_backend.run_mock_factcheck(document, None, PipelineConfig(num_evidences=count))
            if event.result is not None
        ]

    def test_chat_is_deterministic_fictional_and_tracks_user_turns(self):
        original = deepcopy(self.messages)
        first = self.response()
        self.assertEqual(first, self.response(model="another-unused-model"))
        self.assertIn("Fictional sample response (turn 1)", first)
        self.assertIn("invented Northstar Museum", first)
        self.assertTrue(all(first.count(claim) == 1 for claim in mock_backend.MOCK_CLAIMS))
        self.assertEqual(self.messages, original)
        history = self.messages + [
            {"role": "assistant", "content": first, "factcheck": {"mock": True}},
            {"role": "user", "content": "Another question."},
        ]
        snapshot = deepcopy(history)
        second = self.response(history)
        self.assertIn("Fictional sample response (turn 2)", second)
        self.assertTrue(all(claim in second for claim in mock_backend.MOCK_CLAIMS))
        self.assertEqual(history, snapshot)

    def test_stream_fragments_and_simulated_delays_are_visible_and_bounded(self):
        fragments = list(mock_backend.stream_mock_chat_response(self.messages, "unused-model"))
        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(0 < len(fragment) <= 48 for fragment in fragments))
        chat_delay = sum(call.args[0] for call in self.sleep.call_args_list)
        self.assertGreaterEqual(chat_delay, 1)
        self.assertLess(chat_delay, 3)
        self.sleep.reset_mock()
        self.results("\n".join(mock_backend.MOCK_CLAIMS), count=100)
        factcheck_delay = sum(call.args[0] for call in self.sleep.call_args_list)
        self.assertGreaterEqual(factcheck_delay, 8)
        self.assertLess(factcheck_delay, 15)

    def test_each_claim_has_its_own_coherent_five_label_fixture(self):
        results = self.results(self.response())
        self.assertEqual([result["claim"] for result in results], list(mock_backend.MOCK_CLAIMS))
        self.assertFalse(results[-1]["is_checkworthy"])
        self.assertFalse(results[-1]["no_evidence"])
        self.assertEqual(results[-1]["evidences"], [])
        results = [result for result in results if result["is_checkworthy"]]
        self.assertEqual([result["evidences"][0]["verification"]["label"] for result in results], list(LABELS))
        for result in results:
            self.assertTrue(result["mock"])
            self.assertFalse(result["no_evidence"])
            self.assertEqual(len(result["evidences"]), 1)
            evidence = result["evidences"][0]
            self.assertEqual(evidence["dataset"], "Mock evidence")
            self.assertNotIn("meta", evidence)
            self.assertTrue(evidence["passage"].startswith("Fictional"))
            self.assertTrue(evidence["verification"]["rationale"])
        self.assertIn("2012", results[0]["evidences"][0]["passage"])
        self.assertIn("does not list a closing time", results[1]["evidences"][0]["passage"])
        self.assertIn("does not have a gift shop", results[2]["evidences"][0]["passage"])
        self.assertIn("open every Monday", results[3]["evidences"][0]["passage"])
        self.assertNotIn("Morgan Vale", results[4]["evidences"][0]["passage"])

    def test_checks_only_selected_claims_in_source_order(self):
        selected = [mock_backend.MOCK_CLAIMS[3], mock_backend.MOCK_CLAIMS[1]]
        results = self.results("\n".join(selected))
        self.assertEqual([result["claim"] for result in results], selected)
        self.assertEqual(
            [result["evidences"][0]["verification"]["label"] for result in results],
            ["Refuted", "Partially supported"],
        )
        self.assertNotEqual(results[0]["evidences"][0]["passage"], results[1]["evidences"][0]["passage"])

    def test_unknown_text_and_context_do_not_produce_fixture_claims(self):
        for document in ["", "A different museum opened in 2001."]:
            with self.subTest(document=document):
                events = list(mock_backend.run_mock_factcheck(document, self.response(), PipelineConfig()))
                self.assertEqual(events[-1].stage, "complete")
                self.assertEqual(events[-1].claims, [])
                self.assertFalse(any(event.result is not None for event in events))
                self.assertFalse(any(event.stage == "retrieval" for event in events))

    def test_evidence_limit_uses_only_two_finite_variants(self):
        document = "\n".join(mock_backend.MOCK_CLAIMS)
        for requested, expected in [(1, 1), (2, 2), (20, 2)]:
            with self.subTest(requested=requested):
                for result in self.results(document, count=requested):
                    if not result["is_checkworthy"]:
                        self.assertEqual(result["evidences"], [])
                        continue
                    self.assertEqual(len(result["evidences"]), expected)
                    self.assertEqual(len({evidence["passage"] for evidence in result["evidences"]}), expected)
                    self.assertEqual(len({evidence["verification"]["label"] for evidence in result["evidences"]}), 1)

    def test_progress_precedes_delays_and_finishes_after_all_pairs(self):
        events = mock_backend.run_mock_factcheck(
            "\n".join(mock_backend.MOCK_CLAIMS), None, PipelineConfig(num_evidences=2)
        )
        self.assertEqual(next(events).stage, "decomposition")
        self.sleep.assert_not_called()
        remaining = list(events)
        self.assertEqual(remaining[-1].stage, "complete")
        self.assertEqual(remaining[-1].current_claim, 6)
        self.assertEqual(remaining[-1].total_claims, 6)
        self.assertIn("checkworthiness", [event.stage for event in remaining])
        self.assertEqual(sum(event.stage == "verification" for event in remaining), 10)
        for index in range(1, 6):
            stages = [event.stage for event in remaining if event.current_claim == index and event.stage != "complete"]
            self.assertEqual(stages, ["retrieval", "verification", "verification", "claim_complete"])
        self.assertEqual(
            [event.stage for event in remaining if event.current_claim == 6 and event.stage != "complete"],
            ["claim_complete"],
        )

    def test_only_non_checkworthy_sample_skips_retrieval(self):
        events = list(mock_backend.run_mock_factcheck(mock_backend.MOCK_CLAIMS[-1], None, PipelineConfig()))
        self.assertEqual(
            [event.stage for event in events],
            ["decomposition", "decomposition", "checkworthiness", "claim_complete", "complete"],
        )
        result = next(event.result for event in events if event.result is not None)
        self.assertFalse(result["is_checkworthy"])
        self.assertEqual(result["evidences"], [])

    def test_mutating_results_does_not_modify_other_pairs_or_future_runs(self):
        document = "\n".join(mock_backend.MOCK_CLAIMS)
        results = self.results(document, count=2)
        original = deepcopy(results)
        results[0]["claim"] = "Changed"
        results[0]["evidences"][0]["verification"]["label"] = "Changed"
        self.assertEqual(results[0]["evidences"][1], original[0]["evidences"][1])
        self.assertEqual(results[1:], original[1:])
        self.assertEqual(self.results(document, count=2), original)

    def test_mock_operations_never_call_external_backends(self):
        targets = [
            "clients.get_client",
            "pipeline.decompose_document_into_claims",
            "pipeline.identify_checkworthiness",
            "pipeline.verify_claim",
            "pipeline.get_tokenizer",
            "pipeline.get_search_client",
            "pipeline.search_documents",
        ]
        mocks = []
        for target in targets:
            patcher = patch(target, side_effect=AssertionError(f"Unexpected external call: {target}"))
            mocks.append(patcher.start())
            self.addCleanup(patcher.stop)
        with patch.object(socket, "socket", side_effect=AssertionError("Unexpected network access")):
            self.assertEqual(len(self.results(self.response(), count=2)), 6)
        for mocked in mocks:
            mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
