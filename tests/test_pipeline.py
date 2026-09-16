import unittest
from unittest.mock import Mock, call, patch

from pipeline import PipelineConfig, run_factcheck


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig(
            num_evidences=2,
            decomposition_prompt="custom-decomposition.json",
            verification_prompt="custom-verification.json",
        )
        self.decompose = self.patch("decompose_document_into_claims", return_value=["クレーム A", "クレーム B"])
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2]
        self.tokenizer.decode.return_value = "根拠テキスト"
        self.load_tokenizer = self.patch("get_tokenizer", return_value=self.tokenizer)
        self.load_es = self.patch("get_search_client")
        self.load_scorer = self.patch("get_relevance_scorer", return_value=lambda claim, passage: len(passage))
        self.search = self.patch("search_documents")
        self.chunk = self.patch("chunk_document", return_value=["短文", "長い根拠の文章"])
        self.verify = self.patch("verify_claim", side_effect=self.verdict)

    def patch(self, name, **kwargs):
        patcher = patch(f"pipeline.{name}", **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def verdict(claim, evidence, **kwargs):
        return {"label": "Supported" if evidence == "短文" else "Refuted", "rationale": f"{claim} / {evidence}"}

    @staticmethod
    def hit():
        return {"_source": {"token_ids": "1 2", "dataset_name": "test-corpus", "iteration": 10}}

    def test_verifies_every_pair_separately_and_preserves_evidence_association(self):
        self.search.side_effect = [[self.hit()], [self.hit()]]
        events = list(run_factcheck("生成文", "文脈", self.config))
        results = [event.result for event in events if event.stage == "claim_complete"]
        self.assertEqual(len(results), 2)
        self.assertEqual(self.verify.call_count, 4)
        self.verify.assert_has_calls(
            [
                call(claim, evidence, model=self.config.engine, prompt_path=self.config.verification_prompt)
                for claim in ["クレーム A", "クレーム B"]
                for evidence in ["長い根拠の文章", "短文"]
            ]
        )
        self.assertEqual(results[0]["evidences"][0]["verification"]["label"], "Refuted")
        self.assertEqual(results[0]["evidences"][1]["verification"]["label"], "Supported")
        self.assertEqual(self.search.call_count, 2)
        for search_call in self.search.call_args_list:
            self.assertEqual(search_call.args[1], self.config.es_dump_index)
        for result in results:
            for evidence in result["evidences"]:
                self.assertEqual(evidence["dataset"], "test-corpus")
                self.assertEqual(evidence["training_step"], 10)
                self.assertNotIn("meta", evidence)
        self.assertNotIn("metadata", [event.stage for event in events])
        self.decompose.assert_called_once_with(
            "生成文", model=self.config.engine, context="文脈", prompt_path=self.config.decomposition_prompt
        )
        self.assertEqual(events[-1].stage, "complete")
        self.assertEqual(events[-1].current_claim, 2)

    def test_progress_precedes_slow_operations(self):
        self.search.return_value = []
        events = run_factcheck("生成文", None, self.config)
        self.assertEqual(next(events).stage, "decomposition")
        self.decompose.assert_not_called()
        self.assertIsNotNone(next(events).claims)
        self.assertEqual(next(events).stage, "preparation")
        self.load_tokenizer.assert_not_called()
        self.assertEqual(next(events).stage, "preparation")
        self.load_es.assert_not_called()
        self.assertEqual(next(events).stage, "retrieval")
        self.search.assert_not_called()
        list(events)

    def test_no_evidence_is_explicit_and_does_not_invoke_verifier(self):
        self.search.return_value = []
        events = list(run_factcheck("生成文", None, self.config))
        results = [event.result for event in events if event.result is not None]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["no_evidence"] and not result["evidences"] for result in results))
        self.verify.assert_not_called()
        self.load_scorer.assert_not_called()

    def test_empty_document_or_no_claims_skips_retrieval(self):
        events = list(run_factcheck(" \n", None, self.config))
        self.assertEqual(events[-1].claims, [])
        self.decompose.assert_not_called()
        self.decompose.return_value = []
        events = list(run_factcheck("生成文", None, self.config))
        self.assertEqual(events[-1].stage, "complete")
        self.load_tokenizer.assert_not_called()
        self.load_es.assert_not_called()

    def test_verification_failure_is_not_an_insufficient_evidence_verdict(self):
        self.search.return_value = [self.hit()]
        self.verify.side_effect = RuntimeError("API unavailable")
        events = run_factcheck("生成文", None, self.config)
        completed = []
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            for event in events:
                if event.stage == "claim_complete":
                    completed.append(event.result)
        self.assertEqual(completed, [])

    def test_invalid_evidence_count_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(num_evidences=0)


if __name__ == "__main__":
    unittest.main()
