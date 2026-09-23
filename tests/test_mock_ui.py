"""Exercise the real offline mock backend through the Streamlit interface."""

import importlib
import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "serve.py"
sys.path.insert(0, str(ROOT / "src"))

LABELS = {
    "Fully supported",
    "Inferentially supported",
    "Partially supported",
    "Inferentially refuted",
    "Fully refuted",
    "Not enough information",
}
MOCK_NOTICE = "Mock mode: chat responses and fact-check results use synthetic data. No external services are called."


class MockModeUITest(unittest.TestCase):
    def setUp(self):
        backend = importlib.import_module("mock_backend")
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("CHATBOT_", "FACTCHECKER_", "AZURE_OPENAI_", "OPENAI_"))
        }
        self.start_patch(patch.dict(os.environ, environment, clear=True))
        self.start_patch(patch.object(sys, "argv", [str(APP), "--mock"]))
        self.start_patch(patch.object(backend, "sleep", return_value=None))
        self.mock_chat = self.start_patch(
            patch.object(backend, "stream_mock_chat_response", wraps=backend.stream_mock_chat_response)
        )
        self.mock_factcheck = self.start_patch(
            patch.object(backend, "run_mock_factcheck", wraps=backend.run_mock_factcheck)
        )
        self.external_calls = {}
        for target in [
            "chat.stream_chat_response",
            "pipeline.run_factcheck",
            "clients.get_client",
            "chat.get_client",
            "decompose.get_client",
            "checkworthy.get_client",
            "pipeline.identify_checkworthiness",
            "verify.get_client",
            "pipeline.get_tokenizer",
            "pipeline.get_search_client",
            "retrieval.create_elasticsearch_client",
            "retrieval.search_documents",
            "dotenv.load_dotenv",
            "socket.create_connection",
            "socket.socket.connect",
        ]:
            self.external_calls[target] = self.start_patch(
                patch(target, side_effect=AssertionError(f"Mock mode must not call {target}"))
            )
        self.app = AppTest.from_file(str(APP), default_timeout=10).run()
        self.assert_app_ok()

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def tearDown(self):
        for target, blocked in self.external_calls.items():
            with self.subTest(external_call=target):
                blocked.assert_not_called()

    def assert_app_ok(self):
        for _ in range(100):
            jobs = self.app.session_state.filtered_state.get("factcheck_jobs", {})
            active = [
                job
                for response_id, job in jobs.items()
                if self.app.session_state["factchecks"][response_id]["state"] == "running"
            ]
            if not active:
                break
            for job in active:
                job.advance().result(timeout=3)
            self.app.run()
        self.assertEqual(len(self.app.exception), 0, [exception.value for exception in self.app.exception])
        self.assertEqual(len(self.app.error), 0, [error.value for error in self.app.error])

    def send(self, message):
        self.app.chat_input(key="chat_input").set_value(message).run()
        self.assert_app_ok()
        return self.assistants()[-1]

    def assistants(self):
        return [message for message in self.app.session_state["messages"] if message["role"] == "assistant"]

    def check(self, response_id):
        self.app.button(key=f"factcheck_{response_id}").click().run()
        self.assert_app_ok()
        return self.app.session_state["factchecks"][response_id]

    def assert_synthetic_results(self, run):
        self.assertIs(run["mock"], True)
        self.assertEqual(run["state"], "complete")
        self.assertEqual(len(run["claims"]), 7)
        self.assertEqual(len(run["results"]), 7)
        self.assertEqual(sum(result["is_checkworthy"] for result in run["results"]), 6)
        labels = set()
        for result in run["results"]:
            self.assertIs(result["mock"], True)
            self.assertFalse(result["no_evidence"])
            if not result["is_checkworthy"]:
                self.assertEqual(result["evidences"], [])
                continue
            self.assertEqual(len(result["evidences"]), 3)
            for evidence in result["evidences"]:
                self.assertEqual(evidence["dataset"], "Mock evidence")
                self.assertNotIn("meta", evidence)
                labels.add(evidence["verification"]["label"])
        self.assertEqual(labels, LABELS)
        summaries = [block.value for block in self.app.markdown if 'aria-label="Verdict summary"' in block.value]
        self.assertTrue(summaries)
        for summary in summaries:
            self.assertIn("Claims <strong>7</strong>", summary)
            self.assertIn("Check-worthy claims <strong>6</strong>", summary)
            self.assertIn("Verdicts <strong>18</strong>", summary)
            for label in LABELS:
                first, last = label.rsplit(" ", 1)
                self.assertIn(f"*{first}  \n{last}* **3**", [button.label for button in self.app.button])
        self.assertTrue(any("Not check-worthy" in caption.value for caption in self.app.caption))

    def test_initial_mock_page_explains_synthetic_mode_without_credentials(self):
        self.assertTrue(any(info.value == MOCK_NOTICE for info in self.app.info))
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.app.session_state["factchecks"], {})
        self.mock_chat.assert_not_called()
        self.mock_factcheck.assert_not_called()

    def test_multi_turn_checks_use_selected_response_context_and_remain_independent(self):
        first_question = "Tell me about the demonstration."
        second_question = "What else can I ask about?"
        first = deepcopy(self.send(first_question))
        second = deepcopy(self.send(second_question))
        self.assertEqual(len(self.app.session_state["messages"]), 4)
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(self.mock_chat.call_count, 2)
        self.assertEqual(
            self.mock_chat.call_args.args[0],
            [
                {"role": "user", "content": first_question},
                {"role": "assistant", "content": first["content"]},
                {"role": "user", "content": second_question},
            ],
        )
        self.mock_factcheck.assert_not_called()

        first_run = deepcopy(self.check(first["id"]))
        self.assert_synthetic_results(first_run)
        self.assertEqual(first_run["document"], first["content"])
        self.assertEqual(json.loads(first_run["context"]), [{"role": "user", "content": first_question}])
        self.assertEqual(set(self.app.session_state["factchecks"]), {first["id"]})

        second_run = deepcopy(self.check(second["id"]))
        self.assert_synthetic_results(second_run)
        self.assertEqual(second_run["document"], second["content"])
        self.assertEqual(json.loads(second_run["context"]), self.mock_chat.call_args.args[0])
        saved_checks = {first["id"]: first_run, second["id"]: second_run}
        self.assertEqual(self.app.session_state["factchecks"], saved_checks)
        self.assertEqual(self.mock_chat.call_count, 2)
        self.assertEqual(self.mock_factcheck.call_count, 2)
        self.assertEqual(len(self.app.get("download_button")), 0)

        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["factchecks"], saved_checks)
        self.assertEqual(self.mock_chat.call_count, 2)
        self.assertEqual(self.mock_factcheck.call_count, 2)
        self.assertEqual(len(self.app.get("download_button")), 0)

    def test_conversation_can_continue_after_checking_without_rechecking(self):
        response = self.send("Show me a mock response.")
        saved_run = deepcopy(self.check(response["id"]))
        self.send("Continue the conversation.")
        self.assertEqual(len(self.app.session_state["messages"]), 4)
        self.assertEqual(self.app.session_state["factchecks"], {response["id"]: saved_run})
        self.assertEqual(self.mock_chat.call_count, 2)
        self.mock_factcheck.assert_called_once()
        self.assertTrue(any(info.value == MOCK_NOTICE for info in self.app.info))

    def test_new_chat_clears_results_and_starts_fresh_in_mock_mode(self):
        response = self.send("First conversation.")
        self.check(response["id"])
        self.app.button(key="new_chat").click().run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.app.session_state["factchecks"], {})
        self.assertTrue(any(info.value == MOCK_NOTICE for info in self.app.info))
        self.mock_chat.assert_called_once()
        self.mock_factcheck.assert_called_once()
        self.send("A fresh conversation.")
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(self.mock_chat.call_args.args[0], [{"role": "user", "content": "A fresh conversation."}])
        self.mock_factcheck.assert_called_once()


if __name__ == "__main__":
    unittest.main()
