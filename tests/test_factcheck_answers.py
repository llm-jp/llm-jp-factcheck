import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from factcheck_answers import apply_result, pending_tasks, refresh_status, run_stage, summarize


class FactcheckAnswersTests(unittest.TestCase):
    @staticmethod
    def document(qid="AIO02-0001"):
        doc = {"qid": qid, "decomposition_complete": False, "claims": []}
        refresh_status(doc)
        return doc

    def test_resume_skips_completed_operations_and_keeps_pair_identity(self):
        doc = self.document()
        apply_result(doc, "decomposition", None, None, ["Opinion", "Fact"])
        apply_result(doc, "checkworthiness", 0, None, False)
        apply_result(doc, "checkworthiness", 1, None, True)
        apply_result(doc, "retrieval", 1, None, {"evidences": [{"passage": "First"}, {"passage": "Second"}]})
        verdict = {"label": "Fully supported", "rationale": "Direct evidence."}
        apply_result(doc, "verification", 1, 0, verdict)
        # Serialize/reload exactly as checkpoint resumption does.
        restored = json.loads(json.dumps(doc))
        refresh_status(restored)
        for stage in ("decomposition", "checkworthiness", "retrieval"):
            self.assertEqual(pending_tasks([restored], stage), [])
        self.assertEqual(pending_tasks([restored], "verification"), [(doc["qid"], 1, 1)])
        self.assertEqual(restored["claims"][1]["evidences"][0]["verification"], verdict)
        apply_result(restored, "verification", 1, 1, verdict)
        summary = summarize([restored])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["not_checkworthy"], 1)
        self.assertEqual(summary["verification_completed"], 2)

    def test_no_evidence_and_no_claims_finish_without_fabricated_verdicts(self):
        doc = self.document()
        apply_result(doc, "decomposition", None, None, ["Fact"])
        apply_result(doc, "checkworthiness", 0, None, True)
        apply_result(doc, "retrieval", 0, None, {"evidences": []})
        empty = self.document("AIO02-0002")
        apply_result(empty, "decomposition", None, None, [])
        self.assertEqual(pending_tasks([doc, empty], "verification"), [])
        summary = summarize([doc, empty])
        self.assertEqual(summary["documents_complete"], 2)
        self.assertEqual(summary["no_evidence"], 1)
        self.assertEqual(summary["verification_completed"], 0)
        self.assertTrue(all(count == 0 for count in summary["verdicts"].values()))

    def test_error_retains_other_successes_and_only_failed_task_needs_retry(self):
        documents = [self.document(f"AIO02-{i:04d}") for i in range(1, 4)]
        by_id = {doc["qid"]: doc for doc in documents}
        tasks = pending_tasks(documents, "decomposition")

        def invoke(task):
            if task[0] == "AIO02-0002":
                raise ValueError("Invalid structured output.")
            return [task[0]]

        def accept(task, value):
            apply_result(by_id[task[0]], "decomposition", None, None, value)

        checkpoint = Mock()
        with tempfile.TemporaryDirectory() as directory:
            error_path = Path(directory) / "errors.jsonl"
            errors = run_stage("decomposition", tasks, invoke, accept, checkpoint, 2, error_path)
            saved_errors = [json.loads(line) for line in error_path.read_text().splitlines()]
        self.assertEqual(saved_errors, errors)
        self.assertEqual(len(errors), 1)
        self.assertEqual(pending_tasks(documents, "decomposition"), [("AIO02-0002", None, None)])
        self.assertEqual(by_id["AIO02-0003"]["claims"][0]["claim"], "AIO02-0003")
        checkpoint.assert_called()


if __name__ == "__main__":
    unittest.main()
