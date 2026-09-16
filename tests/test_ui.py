"""Chat UI regressions without credentials, Elasticsearch, or model downloads."""

import argparse
import importlib.util
import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.runtime.scriptrunner import StopException
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "serve.py"
LABELS = ["Supported", "Partially supported", "Partially refuted", "Refuted", "Not enough information"]


def event(stage, **kwargs):
    fields = {
        "stage": stage,
        "message": f"Working: {stage}",
        "current_claim": 0,
        "total_claims": 0,
        "result": None,
        "claims": None,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def claim_result(claim="Tokyo is Japan's capital.", labels=None):
    return {
        "claim": claim,
        "no_evidence": False,
        "evidences": [
            {
                "passage": f"Evidence {i}: <script>unsafe</script>",
                "dataset": "test dataset",
                "training_step": 123,
                "score": 0.85,
                "meta": {"url": "https://example.com/evidence"},
                "verification": {"label": label, "rationale": f"Reason {i}"},
            }
            for i, label in enumerate(labels or ["Supported"], 1)
        ],
    }


def completed_events(result):
    return iter(
        [
            event("decomposition", claims=[result["claim"]], total_claims=1),
            event("preparation", total_claims=1),
            event("retrieval", current_claim=1, total_claims=1),
            event("ranking", current_claim=1, total_claims=1),
            event("metadata", current_claim=1, total_claims=1),
            event("verification", current_claim=1, total_claims=1),
            event("claim_complete", current_claim=1, total_claims=1, result=result),
            event("complete", current_claim=1, total_claims=1),
        ]
    )


class ChatUITest(unittest.TestCase):
    def setUp(self):
        self.pipeline = ModuleType("pipeline")
        self.pipeline.PipelineConfig = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        self.pipeline.run_factcheck = Mock()
        self.chat = ModuleType("chat")
        self.chat.stream_chat_response = Mock()
        self.modules_patch = patch.dict(sys.modules, {"pipeline": self.pipeline, "chat": self.chat})
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.argv_patch = patch.object(sys, "argv", [str(APP)])
        self.argv_patch.start()
        self.addCleanup(self.argv_patch.stop)
        self.app = AppTest.from_file(str(APP), default_timeout=10).run()
        self.assert_app_ok()

    def assert_app_ok(self):
        self.assertEqual(len(self.app.exception), 0, [exc.value for exc in self.app.exception])

    def send(self, text, response="Tokyo is Japan's capital."):
        self.chat.stream_chat_response.return_value = iter([response[:8], response[8:]])
        self.app.chat_input(key="chat_input").set_value(text).run()
        self.assert_app_ok()

    def assistants(self):
        return [message for message in self.app.session_state["messages"] if message["role"] == "assistant"]

    def check(self, message_id, result=None):
        if result is not None:
            self.pipeline.run_factcheck.return_value = completed_events(result)
        self.app.button(key=f"factcheck_{message_id}").click().run()
        self.assert_app_ok()
        return self.app.session_state["factchecks"][message_id]

    def rendered_text(self):
        return "\n".join(block.value for block in self.app.markdown)

    def assert_no_supplementary_sections(self):
        for expander in self.app.get("expander"):
            self.assertNotIn(expander.label, {"About verification labels", "Verification input"})
            self.assertFalse(expander.label.startswith("Extracted claims"))

    def assert_processing_finished(self):
        self.assertEqual(len(self.app.get("progress")), 0)
        self.assertEqual(len(self.app.get("status")), 0)

    def test_initial_page_is_english_and_loads_no_models(self):
        self.pipeline.PipelineConfig.assert_not_called()
        self.pipeline.run_factcheck.assert_not_called()
        self.chat.stream_chat_response.assert_not_called()
        self.assertEqual(self.app.chat_input[0].placeholder, "Message the model")
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.app.session_state["factchecks"], {})
        labels = [button.label for button in self.app.button]
        visible = "\n".join(labels + [block.value for block in self.app.title])
        visible += "\n" + "\n".join(block.value for block in self.app.caption)
        visible += "\n" + "\n".join(expander.label for expander in self.app.get("expander"))
        visible += "\n" + re.sub(r"<style>.*?</style>", "", self.rendered_text(), flags=re.DOTALL)
        self.assertNotRegex(visible, r"[ぁ-んァ-ン一-龯]")
        self.assertIn("New chat", labels)
        self.assertIn("Start a conversation", visible)
        self.assertNotIn("research demo", visible.lower())
        self.assertNotIn("verification labels", visible.lower())
        for label in LABELS:
            self.assertNotIn(label, visible)

    def test_results_show_collapsed_evidence_without_supplementary_sections(self):
        self.send("Tell me about Japan.")
        first = self.assistants()[0]
        self.assert_no_supplementary_sections()
        unchecked_text = re.sub(r"<style>.*?</style>", "", self.rendered_text(), flags=re.DOTALL)
        unchecked_text += "\n" + "\n".join(block.value for block in self.app.caption)
        self.assertNotIn("research demo", unchecked_text.lower())
        for label in LABELS:
            self.assertNotIn(label, unchecked_text)

        self.check(first["id"], claim_result())
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.assertTrue(any("Fact-check complete" in caption.value for caption in self.app.caption))
        checked_message = [message for message in self.app.chat_message if message.name == "assistant"][0]
        passages = [expander for expander in checked_message.expander if expander.label == "Evidence passage"]
        self.assertEqual(len(passages), 1)
        self.assertFalse(passages[0].proto.expanded)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", "\n".join(block.value for block in passages[0].markdown))
        self.assertTrue(any(expander.label == "Source details" for expander in checked_message.expander))
        self.assertTrue(any("Fact-check results" in block.value for block in checked_message.markdown))

        self.send("Tell me more.", "Tokyo was formerly called Edo.")
        checked_message, unchecked_message = [
            message for message in self.app.chat_message if message.name == "assistant"
        ]
        self.assertTrue(any(expander.label == "Evidence passage" for expander in checked_message.expander))
        self.assertEqual(len(unchecked_message.expander), 0)
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.pipeline.run_factcheck.assert_called_once()

    def test_two_chat_turns_send_conversation_and_join_stream_chunks(self):
        first_question = "What is the capital of Japan?"
        first_answer = "Tokyo is Japan's capital."
        second_question = "What was its former name?"
        second_answer = "Tokyo was formerly called Edo."
        self.send(first_question, first_answer)
        self.send(second_question, second_answer)
        calls = self.chat.stream_chat_response.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0], [{"role": "user", "content": first_question}])
        self.assertEqual(
            calls[1].args[0],
            [
                {"role": "user", "content": first_question},
                {"role": "assistant", "content": first_answer},
                {"role": "user", "content": second_question},
            ],
        )
        messages = self.app.session_state["messages"]
        self.assertEqual(
            [message["content"] for message in messages], [first_question, first_answer, second_question, second_answer]
        )
        self.assertEqual(len({message["id"] for message in messages}), 4)
        self.assertEqual(len(self.app.chat_message), 4)
        for message in self.assistants():
            self.assertEqual(self.app.button(key=f"factcheck_{message['id']}").label, "Fact-check response")
        self.pipeline.run_factcheck.assert_not_called()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)

    def test_check_uses_selected_response_and_preceding_context_only(self):
        self.send("Tell me about Japan.", "Tokyo is Japan's capital.")
        first = deepcopy(self.assistants()[0])
        self.send("Now tell me about France.", "Paris is France's capital.")
        second = deepcopy(self.assistants()[1])
        run = self.check(first["id"], claim_result(first["content"]))
        args = self.pipeline.run_factcheck.call_args.args
        self.assertEqual(args[0], first["content"])
        self.assertIn("Tell me about Japan.", args[1])
        self.assertNotIn(first["content"], args[1])
        self.assertNotIn("Now tell me about France.", args[1])
        self.assertNotIn(second["content"], args[1])
        self.assertEqual(run["document"], first["content"])
        self.assertEqual(run["context"], args[1])
        self.assertEqual(run["state"], "complete")
        self.assertEqual(set(self.app.session_state["factchecks"]), {first["id"]})
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)

    def test_chat_continues_after_checking_without_losing_or_repeating_checks(self):
        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        saved_run = deepcopy(self.check(message["id"], claim_result()))
        self.send("Tell me more.", "Tokyo was formerly called Edo.")
        self.assertEqual(len(self.app.session_state["messages"]), 4)
        self.assertEqual(self.app.session_state["factchecks"], {message["id"]: saved_run})
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)
        self.pipeline.run_factcheck.assert_called_once()

    def test_whitespace_message_does_not_invoke_models(self):
        self.app.chat_input[0].set_value("  \n  ").run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["messages"], [])
        self.chat.stream_chat_response.assert_not_called()
        self.pipeline.run_factcheck.assert_not_called()

    def test_identical_answers_have_isolated_checks(self):
        answer = "Tokyo is Japan's capital."
        self.send("First question", answer)
        self.send("Second question", answer)
        first, second = self.assistants()
        self.assertNotEqual(first["id"], second["id"])
        first_run = deepcopy(self.check(first["id"], claim_result(labels=["Supported"])))
        second_run = deepcopy(self.check(second["id"], claim_result(labels=["Partially supported"])))
        checks = self.app.session_state["factchecks"]
        self.assertEqual(set(checks), {first["id"], second["id"]})
        self.assertEqual(checks[first["id"]], first_run)
        self.assertEqual(checks[second["id"]], second_run)
        self.assertIn("First question", second_run["context"])
        self.assertIn("Second question", second_run["context"])
        self.assertIn(answer, second_run["context"])
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)
        self.assertEqual(len(self.app.get("download_button")), 0)
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["factchecks"][first["id"]], first_run)
        self.assertEqual(self.app.session_state["factchecks"][second["id"]], second_run)
        self.assertEqual(self.pipeline.run_factcheck.call_count, 2)
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)
        self.assertEqual(len(self.app.get("download_button")), 0)

    def test_rechecking_replaces_only_the_selected_response_results(self):
        self.send("Tell me about Japan.")
        self.send("Tell me about France.", "Paris is France's capital.")
        first, second = self.assistants()
        first_run = deepcopy(self.check(first["id"], claim_result(first["content"], labels=["Supported"])))
        second_run = deepcopy(self.check(second["id"], claim_result(second["content"])))
        previous = self.check(first["id"], claim_result(first["content"], labels=["Refuted"]))
        self.assertEqual(previous, first_run)
        self.assertEqual(self.pipeline.run_factcheck.call_count, 2)
        self.assertIn("Run fact-check again?", self.rendered_text())
        self.app.button(key=f"confirm_factcheck_{first['id']}").click().run()
        self.assert_app_ok()
        rerun = self.app.session_state["factchecks"][first["id"]]
        self.assertEqual(len(rerun["results"]), 1)
        self.assertEqual(rerun["results"][0]["evidences"][0]["verification"]["label"], "Refuted")
        self.assertEqual(self.app.session_state["factchecks"][second["id"]], second_run)
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)
        self.assertEqual(self.pipeline.run_factcheck.call_count, 3)
        self.assertFalse(self.app.session_state.filtered_state.get("recheck_confirmation"))
        self.assertFalse(self.app.session_state.filtered_state.get("pending_factcheck"))
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.pipeline.run_factcheck.call_count, 3)
        self.assertEqual(self.app.session_state["factchecks"][first["id"]], rerun)

    def test_recheck_confirmation_persists_across_chat_and_reruns_until_cancel(self):
        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        saved_run = deepcopy(self.check(message["id"], claim_result()))
        self.assertFalse(self.app.session_state.filtered_state.get("recheck_confirmation"))
        self.check(message["id"])
        self.assertEqual(self.app.session_state["recheck_confirmation"], message["id"])
        self.assertIn("Run fact-check again?", self.rendered_text())
        self.assertEqual(self.app.button(key=f"confirm_factcheck_{message['id']}").label, "Run again")
        self.assertEqual(self.app.button(key=f"cancel_factcheck_{message['id']}").label, "Cancel")
        self.assertEqual(self.app.session_state["factchecks"][message["id"]], saved_run)
        self.pipeline.run_factcheck.assert_called_once()

        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["recheck_confirmation"], message["id"])
        self.assertIn("Run fact-check again?", self.rendered_text())
        self.assertEqual(self.app.session_state["factchecks"][message["id"]], saved_run)
        self.pipeline.run_factcheck.assert_called_once()

        self.send("Tell me more.", "Tokyo was formerly called Edo.")
        self.assertEqual(len(self.app.session_state["messages"]), 4)
        self.assertEqual(self.app.session_state["recheck_confirmation"], message["id"])
        self.assertFalse(self.app.session_state.filtered_state.get("pending_factcheck"))
        self.assertIn("Run fact-check again?", self.rendered_text())
        self.assertEqual(self.app.session_state["factchecks"][message["id"]], saved_run)
        self.pipeline.run_factcheck.assert_called_once()

        self.app.button(key=f"cancel_factcheck_{message['id']}").click().run()
        self.assert_app_ok()
        self.assertFalse(self.app.session_state.filtered_state.get("recheck_confirmation"))
        self.assertFalse(self.app.session_state.filtered_state.get("pending_factcheck"))
        self.assertNotIn("Run fact-check again?", self.rendered_text())
        self.assertEqual(self.app.session_state["factchecks"][message["id"]], saved_run)
        self.pipeline.run_factcheck.assert_called_once()
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["factchecks"][message["id"]], saved_run)
        self.pipeline.run_factcheck.assert_called_once()
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)

    def test_switching_target_or_first_check_dismisses_previous_confirmation(self):
        self.send("Tell me about Japan.")
        self.send("Tell me about France.", "Paris is France's capital.")
        self.send("Tell me about Italy.", "Rome is Italy's capital.")
        first, second, third = self.assistants()
        first_run = deepcopy(self.check(first["id"], claim_result(first["content"])))
        second_run = deepcopy(self.check(second["id"], claim_result(second["content"])))
        self.check(first["id"])
        self.check(second["id"])
        self.assertEqual(self.app.session_state["recheck_confirmation"], second["id"])
        confirmations = [button.key for button in self.app.button if button.label == "Run again"]
        self.assertEqual(confirmations, [f"confirm_factcheck_{second['id']}"])
        self.assertEqual(self.pipeline.run_factcheck.call_count, 2)

        self.check(third["id"], claim_result(third["content"]))
        self.assertFalse(self.app.session_state.filtered_state.get("recheck_confirmation"))
        self.assertNotIn("Run fact-check again?", self.rendered_text())
        self.assertEqual(self.pipeline.run_factcheck.call_count, 3)
        self.assertEqual(self.app.session_state["factchecks"][first["id"]], first_run)
        self.assertEqual(self.app.session_state["factchecks"][second["id"]], second_run)
        self.assertFalse(self.app.session_state.filtered_state.get("pending_factcheck"))
        self.assertEqual(self.chat.stream_chat_response.call_count, 3)

    def test_pair_labels_and_escaped_evidence_are_shown(self):
        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        run = self.check(message["id"], claim_result(labels=LABELS))
        self.assertEqual(len(run["results"][0]["evidences"]), 5)
        content = self.rendered_text()
        for label in LABELS:
            self.assertIn(f">{label}</span>", content)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", content)
        self.assertNotIn("<script>unsafe</script>", content)
        self.assertEqual(len(self.app.json), 0)
        sources = [expander for expander in self.app.expander if expander.label == "Source details"]
        self.assertEqual(len(sources), 5)
        for source in sources:
            self.assertEqual([text.value for text in source.text], ["Source: test dataset", "Training step: 123"])
        visible = content + "\n" + "\n".join(text.value for text in self.app.text)
        visible += "\n" + "\n".join(caption.value for caption in self.app.caption)
        for hidden_detail in ["Relevance score", "https://example.com/evidence"]:
            self.assertNotIn(hidden_detail, visible)
        evidence = run["results"][0]["evidences"][0]
        self.assertEqual(evidence["training_step"], 123)
        self.assertEqual(evidence["score"], 0.85)
        self.assertEqual(evidence["meta"], {"url": "https://example.com/evidence"})
        passages = [expander for expander in self.app.expander if expander.label == "Evidence passage"]
        self.assertEqual(len(passages), 5)
        self.assertTrue(all(not passage.proto.expanded for passage in passages))
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()

    def test_generation_error_keeps_user_and_retry_does_not_duplicate_it(self):
        def broken_stream(*_args, **_kwargs):
            yield "Incomplete response"
            raise RuntimeError("Chat connection failed")

        self.chat.stream_chat_response.side_effect = broken_stream
        self.app.chat_input[0].set_value("Tell me about Japan.").run()
        self.assert_app_ok()
        messages = deepcopy(self.app.session_state["messages"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(self.assistants(), [])
        self.assertIn("Chat connection failed", self.app.session_state["chat_error"])
        self.assertTrue(any("Chat connection failed" in error.value for error in self.app.error))
        self.pipeline.run_factcheck.assert_not_called()
        self.chat.stream_chat_response.side_effect = None
        self.chat.stream_chat_response.return_value = iter(["Tokyo is Japan's capital."])
        self.app.button(key="retry_response").click().run()
        self.assert_app_ok()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(self.app.session_state["messages"][0], messages[0])
        self.assertEqual(len(self.assistants()), 1)
        self.assertFalse(self.app.session_state.filtered_state.get("chat_error"))
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)
        self.assertEqual(
            self.chat.stream_chat_response.call_args.args[0], [{"role": "user", "content": "Tell me about Japan."}]
        )

    def test_factcheck_failure_preserves_completed_claims_and_conversation(self):
        def failing_events(*_args, **_kwargs):
            yield event("decomposition", claims=["Completed claim", "Failed claim"], total_claims=2)
            yield event("claim_complete", current_claim=1, total_claims=2, result=claim_result("Completed claim"))
            raise RuntimeError("Verification connection failed")

        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        saved_messages = deepcopy(self.app.session_state["messages"])
        self.pipeline.run_factcheck.side_effect = failing_events
        run = self.check(message["id"])
        self.assertEqual(run["state"], "error")
        self.assertEqual(len(run["results"]), 1)
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.assertEqual(run["results"][0]["evidences"][0]["verification"]["label"], "Supported")
        self.assertIn("Verification connection failed", run["error"])
        self.assertTrue(any("Verification connection failed" in error.value for error in self.app.error))
        self.assertEqual(self.app.session_state["messages"], saved_messages)
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(len(self.app.session_state["factchecks"][message["id"]]["results"]), 1)
        self.chat.stream_chat_response.assert_called_once()
        self.pipeline.run_factcheck.assert_called_once()
        saved_run = deepcopy(run)
        self.check(message["id"])
        self.assertIn("Run fact-check again?", self.rendered_text())
        self.assertEqual(self.app.session_state["factchecks"][message["id"]], saved_run)
        self.pipeline.run_factcheck.assert_called_once()

        self.pipeline.run_factcheck.side_effect = None
        self.pipeline.run_factcheck.return_value = completed_events(claim_result("Recovered claim"))
        self.app.button(key=f"confirm_factcheck_{message['id']}").click().run()
        self.assert_app_ok()
        recovered = self.app.session_state["factchecks"][message["id"]]
        self.assertEqual(recovered["state"], "complete")
        self.assertIsNone(recovered["error"])
        self.assertEqual([result["claim"] for result in recovered["results"]], ["Recovered claim"])
        self.assertEqual(self.app.session_state["messages"], saved_messages)
        self.assertEqual(len(self.app.error), 0)
        self.assertFalse(self.app.session_state.filtered_state.get("recheck_confirmation"))
        self.assertFalse(self.app.session_state.filtered_state.get("pending_factcheck"))
        self.assert_processing_finished()
        self.assertEqual(self.pipeline.run_factcheck.call_count, 2)
        self.chat.stream_chat_response.assert_called_once()

    def test_factcheck_failure_before_results_removes_progress(self):
        def failing_events(*_args, **_kwargs):
            yield event("decomposition", claims=["A claim"], total_claims=1)
            raise RuntimeError("Evidence retrieval failed")

        self.send("Tell me about Japan.")
        self.pipeline.run_factcheck.side_effect = failing_events
        run = self.check(self.assistants()[0]["id"])
        self.assertEqual(run["state"], "error")
        self.assertEqual(run["results"], [])
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.app.run()
        self.assert_app_ok()
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.pipeline.run_factcheck.assert_called_once()

    def test_interrupted_factcheck_requires_confirmation_before_retry(self):
        def interrupted_events(*_args, **_kwargs):
            yield event("decomposition", claims=["A claim"], total_claims=1)
            raise StopException()

        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        self.pipeline.run_factcheck.side_effect = interrupted_events
        interrupted = deepcopy(self.check(message["id"]))
        self.assertEqual(interrupted["state"], "running")
        self.app.run()
        self.assert_app_ok()
        self.assert_processing_finished()
        self.check(message["id"])
        self.assertEqual(self.app.session_state["factchecks"][message["id"]], interrupted)
        self.assertIn("Run fact-check again?", self.rendered_text())
        self.pipeline.run_factcheck.assert_called_once()

        self.pipeline.run_factcheck.side_effect = None
        self.pipeline.run_factcheck.return_value = completed_events(claim_result())
        self.app.button(key=f"confirm_factcheck_{message['id']}").click().run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["factchecks"][message["id"]]["state"], "complete")
        self.assertEqual(self.pipeline.run_factcheck.call_count, 2)
        self.assert_processing_finished()

    def test_interrupted_stream_recovers_without_replaying_or_saving_partial_output(self):
        def interrupted_stream(*_args, **_kwargs):
            yield "Unfinished assistant text"
            raise StopException()

        self.chat.stream_chat_response.side_effect = interrupted_stream
        self.app.chat_input[0].set_value("Tell me about Japan.").run()
        self.assert_app_ok()
        messages = deepcopy(self.app.session_state["messages"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(self.assistants(), [])
        self.chat.stream_chat_response.assert_called_once()

        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["messages"], messages)
        self.assertIn("interrupted", self.app.session_state["chat_error"].lower())
        self.assertTrue(any("interrupted" in error.value.lower() for error in self.app.error))
        self.assertNotIn("Unfinished assistant text", self.rendered_text())
        self.chat.stream_chat_response.assert_called_once()
        self.pipeline.run_factcheck.assert_not_called()

        self.chat.stream_chat_response.side_effect = None
        self.chat.stream_chat_response.return_value = iter(["Tokyo is Japan's capital."])
        self.app.button(key="retry_response").click().run()
        self.assert_app_ok()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(self.app.session_state["messages"][0], messages[0])
        self.assertEqual(self.assistants()[0]["content"], "Tokyo is Japan's capital.")
        self.assertFalse(self.app.session_state.filtered_state.get("chat_error"))
        calls = self.chat.stream_chat_response.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0], calls[1].args[0])
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.chat.stream_chat_response.call_count, 2)
        self.pipeline.run_factcheck.assert_not_called()

    def test_no_evidence_shows_nei_without_a_fabricated_pair(self):
        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        result = {"claim": "A claim without retrieved evidence", "no_evidence": True, "evidences": []}
        run = self.check(message["id"], result)
        self.assertIn('class="verdict nei">Not enough information', self.rendered_text())
        self.assertEqual(run["results"][0]["evidences"], [])
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.assertFalse(any(expander.label == "Source details" for expander in self.app.get("expander")))

    def test_empty_decomposition_is_explained(self):
        self.send("Hello", "Hello there!")
        message = self.assistants()[0]
        self.pipeline.run_factcheck.return_value = iter([event("decomposition", claims=[]), event("complete")])
        run = self.check(message["id"])
        self.assertEqual(run["state"], "complete")
        self.assertEqual(run["results"], [])
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.assertTrue(any("claim" in info.value.lower() for info in self.app.info))
        self.chat.stream_chat_response.assert_called_once()

    def test_new_chat_clears_conversation_checks_and_error_without_api_calls(self):
        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        self.check(message["id"], claim_result())
        self.check(message["id"])
        self.assertEqual(self.app.session_state["recheck_confirmation"], message["id"])
        self.app.session_state["pending_factcheck"] = message["id"]
        self.app.session_state["chat_error"] = "Previous error"
        self.app.button(key="new_chat").click().run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.app.session_state["factchecks"], {})
        self.assertFalse(self.app.session_state.filtered_state.get("chat_error"))
        self.assertFalse(self.app.session_state.filtered_state.get("recheck_confirmation"))
        self.assertFalse(self.app.session_state.filtered_state.get("pending_factcheck"))
        self.assertEqual(len(self.app.chat_message), 0)
        self.chat.stream_chat_response.assert_called_once()
        self.pipeline.run_factcheck.assert_called_once()

    def test_cli_prompt_paths_chat_engine_and_evidence_validation(self):
        spec = importlib.util.spec_from_file_location("factcheck_ui_for_test", APP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parse_args(
            [
                "--decomposition-prompt",
                "decompose.json",
                "--verification-prompt",
                "verify.json",
                "--chat-engine",
                "chat-model",
            ]
        )
        self.assertEqual(args.decomposition_prompt, "decompose.json")
        self.assertEqual(args.verification_prompt, "verify.json")
        self.assertEqual(args.chat_engine, "chat-model")
        with self.assertRaises(argparse.ArgumentTypeError):
            module.positive_integer("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            module.positive_integer("-2")
        self.assertEqual(module.positive_integer("3"), 3)

    def test_chat_model_selection_does_not_change_factchecker_model(self):
        for chat_engine in [None, "chat-model"]:
            with self.subTest(chat_engine=chat_engine):
                argv = [str(APP), "--engine", "verification-model"]
                if chat_engine:
                    argv.extend(["--chat-engine", chat_engine])
                with patch.object(sys, "argv", argv):
                    self.app = AppTest.from_file(str(APP), default_timeout=10).run()
                    self.assert_app_ok()
                    self.send("Tell me about Japan.")
                    chat_call = self.chat.stream_chat_response.call_args
                    model = chat_call.kwargs.get("model") or chat_call.args[1]
                    self.assertEqual(model, chat_engine or "verification-model")
                    self.check(self.assistants()[0]["id"], claim_result())
                    config = self.pipeline.run_factcheck.call_args.args[2]
                    self.assertEqual(config.engine, "verification-model")


if __name__ == "__main__":
    unittest.main()
