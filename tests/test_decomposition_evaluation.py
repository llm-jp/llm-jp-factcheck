import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_decomposition", ROOT / "scripts/evaluate_decomposition.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationDataTests(unittest.TestCase):
    def test_loads_only_test_split_and_excludes_question_metadata(self):
        source = {"revision": "pinned", "test_source": "test.json", "gold_source": "gold.json"}
        test = [{"textId": "test", "text": "Answer"}]
        gold = [
            {"textId": "dev", "text": "Development", "claims": []},
            {"textId": "test", "text": "Answer", "prompt": "Question", "claims": [{"text": "Gold"}]},
        ]
        with patch.object(evaluation, "read_git", side_effect=[json.dumps(test), json.dumps(gold)]):
            data = evaluation.load_test_data(Path("repo"), source)
        self.assertEqual(data, [{"textId": "test", "text": "Answer", "claims": [{"text": "Gold"}]}])

    def test_rejects_inconsistent_test_and_gold_text(self):
        source = {"revision": "pinned", "test_source": "test.json", "gold_source": "gold.json"}
        test = [{"textId": "1", "text": "Original"}]
        gold = [{"textId": "1", "text": "Changed", "claims": []}]
        with patch.object(evaluation, "read_git", side_effect=[json.dumps(test), json.dumps(gold)]):
            with self.assertRaisesRegex(ValueError, "text differ"):
                evaluation.load_test_data(Path("repo"), source)

    def test_score_only_rejects_missing_predictions_without_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = {"revision": "pinned", "additional_shots": 8}
            manifest = output / "source.json"
            manifest.write_text(json.dumps(source))
            gold = [{"textId": "1", "text": "Text", "claims": [{"text": "Gold"}]}]
            argv = [
                "evaluate",
                "--dataset-repo",
                "repo",
                "--output",
                str(output),
                "--runs",
                "1",
                "--source-manifest",
                str(manifest),
            ]
            # Save a valid protocol without generating a prediction.
            with (
                patch.object(evaluation, "load_test_data", return_value=gold),
                patch.object(evaluation.importlib.metadata, "version", return_value="test"),
                patch("clients.get_client", side_effect=RuntimeError("No API")) as client,
                patch.object(sys, "argv", argv),
            ):
                with self.assertRaisesRegex(RuntimeError, "No API"):
                    evaluation.main()
                self.assertEqual(json.loads((output / "protocol.json").read_text())["source"], source)
                client.reset_mock()
                with patch.object(sys, "argv", argv + ["--score-only"]):
                    with self.assertRaisesRegex(ValueError, "incomplete"):
                        evaluation.main()
                client.assert_not_called()

                prompt = output / "prompt.json"
                prompt.write_text("{}")
                with self.assertRaisesRegex(ValueError, "snapshot differs"):
                    evaluation.main()
                client.assert_not_called()

    def test_refresh_metrics_allows_only_scorer_changes_without_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            gold = [{"textId": "1", "text": "Text", "claims": [{"text": "Gold"}]}]
            argv = ["evaluate", "--dataset-repo", "repo", "--output", str(output), "--runs", "1"]
            macro = {metric: dict(precision=1, recall=1, f1=1) for metric in ("exact", "fuzzy_content")}
            with (
                patch.object(evaluation, "load_test_data", return_value=gold),
                patch.object(evaluation.importlib.metadata, "version", return_value="test"),
                patch.object(evaluation, "evaluate_documents", return_value={"macro": macro}),
                patch("clients.get_client", side_effect=RuntimeError("No API")) as client,
                patch.object(sys, "argv", argv),
            ):
                with self.assertRaisesRegex(RuntimeError, "No API"):
                    evaluation.main()
                path = output / "protocol.json"
                original = json.loads(path.read_text())
                old = json.loads(path.read_text())
                old["implementation_sha256"]["claim_metrics.py"] = "old-scorer"
                path.write_text(json.dumps(old))
                (output / "predictions_01.json").write_text(json.dumps(gold))
                client.reset_mock()
                with patch.object(sys, "argv", argv + ["--score-only", "--refresh-metrics"]):
                    evaluation.main()
                    self.assertEqual(json.loads(path.read_text()), original)
                    self.assertEqual(json.loads((output / "previous_scoring_protocol.json").read_text()), old)
                    changed = dict(original, model="different-model")
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "settings changed"):
                        evaluation.main()
                client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
