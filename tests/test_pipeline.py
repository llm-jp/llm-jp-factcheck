import unittest
from unittest.mock import Mock, call, patch

from pipeline import PipelineConfig, run_factcheck


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig(
            num_evidences=2,
            decomposition_prompt="custom-decomposition.json",
            checkworthiness_prompt="custom-checkworthiness.json",
            verification_prompt="custom-verification.json",
        )
        self.decompose = self.patch("decompose_document_into_claims", return_value=["クレーム A", "クレーム B"])
        self.checkworthy = self.patch(
            "identify_checkworthiness", side_effect=lambda claims, **kwargs: [True] * len(claims)
        )
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2]
        self.tokenizer.decode.side_effect = lambda ids: "短文" if ids == [1, 2] else "長い根拠の文章"
        self.load_tokenizer = self.patch("get_tokenizer", return_value=self.tokenizer)
        self.load_es = self.patch("get_search_client")
        self.search = self.patch("search_documents")
        self.verify = self.patch("verify_claim", side_effect=self.verdict)

    def patch(self, name, **kwargs):
        patcher = patch(f"pipeline.{name}", **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def verdict(claim, evidence, **kwargs):
        return {"label": "Supported" if evidence == "短文" else "Refuted", "rationale": f"{claim} / {evidence}"}

    @staticmethod
    def hit(token_ids="1 2", dataset="test-corpus", training_step=10):
        return {"_source": {"token_ids": token_ids, "dataset_name": dataset, "iteration": training_step}}

    def test_verifies_every_pair_separately_and_preserves_evidence_association(self):
        self.search.return_value = [self.hit(training_step=0), self.hit("3 4", "second-corpus", 20)]
        events = list(run_factcheck("生成文", "文脈", self.config))
        results = [event.result for event in events if event.stage == "claim_complete"]
        self.assertEqual(len(results), 2)
        self.assertEqual(self.verify.call_count, 4)
        self.verify.assert_has_calls(
            [
                call(claim, evidence, model=self.config.engine, prompt_path=self.config.verification_prompt)
                for claim in ["クレーム A", "クレーム B"]
                for evidence in ["短文", "長い根拠の文章"]
            ]
        )
        self.assertEqual(results[0]["evidences"][0]["verification"]["label"], "Supported")
        self.assertEqual(results[0]["evidences"][1]["verification"]["label"], "Refuted")
        self.assertEqual(self.search.call_count, 2)
        for search_call in self.search.call_args_list:
            self.assertEqual(search_call.args, (self.load_es.return_value, self.config.es_dump_index))
            self.assertEqual(
                search_call.kwargs,
                {
                    "body": {"query": {"match": {"token_ids": "1 2"}}},
                    "size": 2,
                    "max_concurrent_shard_requests": 64,
                },
            )
        for result in results:
            self.assertEqual(
                [
                    (evidence["passage"], evidence["dataset"], evidence["training_step"])
                    for evidence in result["evidences"]
                ],
                [("短文", "test-corpus", 0), ("長い根拠の文章", "second-corpus", 20)],
            )
            for evidence in result["evidences"]:
                self.assertNotIn("meta", evidence)
                self.assertNotIn("score", evidence)
        self.assertNotIn("metadata", [event.stage for event in events])
        self.assertNotIn("ranking", [event.stage for event in events])
        self.assertEqual(
            self.tokenizer.encode.call_args_list,
            [call(claim, add_special_tokens=False) for claim in ["クレーム A", "クレーム B"]],
        )
        self.assertEqual(self.tokenizer.decode.call_args_list, [call([1, 2]), call([3, 4])] * 2)
        self.decompose.assert_called_once_with(
            "生成文", model=self.config.engine, context="文脈", prompt_path=self.config.decomposition_prompt
        )
        self.checkworthy.assert_called_once_with(
            self.decompose.return_value, model=self.config.engine, prompt_path=self.config.checkworthiness_prompt
        )
        self.assertEqual(events[-1].stage, "complete")
        self.assertEqual(events[-1].current_claim, 2)

    def test_progress_precedes_slow_operations(self):
        self.search.return_value = []
        events = run_factcheck("生成文", None, self.config)
        self.assertEqual(next(events).stage, "decomposition")
        self.decompose.assert_not_called()
        self.assertIsNotNone(next(events).claims)
        self.assertEqual(next(events).stage, "checkworthiness")
        self.checkworthy.assert_not_called()
        self.assertEqual(next(events).stage, "checkworthiness")
        self.checkworthy.assert_called_once()
        self.assertEqual(next(events).stage, "preparation")
        self.load_tokenizer.assert_not_called()
        self.assertEqual(next(events).stage, "preparation")
        self.load_es.assert_not_called()
        self.assertEqual(next(events).stage, "retrieval")
        self.search.assert_not_called()
        list(events)

    def test_skips_non_checkworthy_claims_and_preserves_source_order(self):
        self.decompose.return_value = ["Opinion", "Factual claim", "Question"]
        self.checkworthy.side_effect = None
        self.checkworthy.return_value = [False, True, False]
        self.search.return_value = [self.hit()]
        events = list(run_factcheck("Response", None, self.config))
        results = [event.result for event in events if event.result is not None]
        self.assertEqual([result["claim"] for result in results], self.decompose.return_value)
        self.assertEqual([result["is_checkworthy"] for result in results], [False, True, False])
        self.search.assert_called_once()
        self.tokenizer.encode.assert_called_once_with("Factual claim", add_special_tokens=False)
        self.verify.assert_called_once_with(
            "Factual claim", "短文", model=self.config.engine, prompt_path=self.config.verification_prompt
        )
        for result in (results[0], results[2]):
            self.assertEqual(result["evidences"], [])
            self.assertFalse(result["no_evidence"])
        self.assertEqual(events[-1].stage, "complete")
        self.assertEqual(events[-1].current_claim, 3)

    def test_all_non_checkworthy_claims_skip_all_retrieval_setup(self):
        self.checkworthy.side_effect = lambda claims, **kwargs: [False] * len(claims)
        events = list(run_factcheck("Response", None, self.config))
        self.assertEqual(events[-1].stage, "complete")
        results = [event.result for event in events if event.result is not None]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["is_checkworthy"] is False for result in results))
        self.assertNotIn("preparation", [event.stage for event in events])
        self.load_tokenizer.assert_not_called()
        self.load_es.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()

    def test_checkworthiness_failure_stops_before_retrieval(self):
        self.checkworthy.side_effect = ValueError("Invalid check-worthiness labels")
        events = run_factcheck("Response", None, self.config)
        with self.assertRaisesRegex(ValueError, "Invalid check-worthiness labels"):
            list(events)
        self.load_tokenizer.assert_not_called()
        self.load_es.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()

    def test_no_evidence_is_explicit_and_does_not_invoke_verifier(self):
        self.search.return_value = []
        events = list(run_factcheck("生成文", None, self.config))
        results = [event.result for event in events if event.result is not None]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["no_evidence"] and not result["evidences"] for result in results))
        self.verify.assert_not_called()

    def test_requested_evidence_count_controls_search_size(self):
        self.decompose.return_value = ["Claim"]
        for count in [1, 5]:
            with self.subTest(count=count):
                self.search.return_value = [self.hit(str(index)) for index in range(count)]
                self.verify.reset_mock()
                events = list(run_factcheck("Response", None, PipelineConfig(num_evidences=count)))
                results = [event.result for event in events if event.result is not None]
                self.assertEqual(self.search.call_args.kwargs["size"], count)
                self.assertEqual(len(results[0]["evidences"]), count)
                self.assertEqual(self.verify.call_count, count)

    def test_long_hit_is_verified_as_one_complete_passage(self):
        self.decompose.return_value = ["Claim"]
        passage = "Complete evidence passage. " * 100
        self.tokenizer.decode.side_effect = None
        self.tokenizer.decode.return_value = f"  {passage}\n"
        self.search.return_value = [self.hit()]
        events = list(run_factcheck("Response", None, self.config))
        result = next(event.result for event in events if event.result is not None)
        self.assertEqual(len(result["evidences"]), 1)
        self.assertEqual(result["evidences"][0]["passage"], passage.strip())
        self.verify.assert_called_once_with(
            "Claim", passage.strip(), model=self.config.engine, prompt_path=self.config.verification_prompt
        )

    def test_empty_decoded_hits_skip_verification(self):
        self.search.return_value = [self.hit(token_ids="")]
        self.tokenizer.decode.side_effect = None
        self.tokenizer.decode.return_value = " \n "
        events = list(run_factcheck("Response", None, self.config))
        results = [event.result for event in events if event.result is not None]
        self.assertTrue(all(result["no_evidence"] and not result["evidences"] for result in results))
        self.verify.assert_not_called()

    def test_empty_document_or_no_claims_skips_retrieval(self):
        events = list(run_factcheck(" \n", None, self.config))
        self.assertEqual(events[-1].claims, [])
        self.decompose.assert_not_called()
        self.decompose.return_value = []
        events = list(run_factcheck("生成文", None, self.config))
        self.assertEqual(events[-1].stage, "complete")
        self.checkworthy.assert_not_called()
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
