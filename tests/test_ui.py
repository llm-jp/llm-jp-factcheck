"""Chat UI regressions without credentials, Elasticsearch, or model downloads."""

import argparse
import importlib.util
import os
import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import StopException
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "serve.py"
LABELS = [
    "Fully supported",
    "Inferentially supported",
    "Partially supported",
    "Inferentially refuted",
    "Fully refuted",
    "Not enough information",
]


def load_ui_module():
    spec = importlib.util.spec_from_file_location("factcheck_ui_for_test", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(stage, **kwargs):
    fields = {
        "stage": stage,
        "message": f"Working: {stage}",
        "current_claim": 0,
        "total_claims": 0,
        "result": None,
        "claims": None,
        "claim_update": None,
        "progress": None,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def claim_result(claim="Tokyo is Japan's capital.", labels=None):
    return {
        "claim": claim,
        "is_checkworthy": True,
        "no_evidence": False,
        "evidences": [
            {
                "passage": f"Evidence {i}: <script>unsafe</script>",
                "dataset": "test dataset",
                "training_step": 123,
                "verification": {"label": label, "rationale": f"Reason {i}"},
            }
            for i, label in enumerate(labels or ["Fully supported"], 1)
        ],
    }


def completed_events(result):
    return iter(
        [
            event("decomposition", claims=[result["claim"]], total_claims=1),
            event("checkworthiness", total_claims=1),
            event("preparation", total_claims=1),
            event("retrieval", current_claim=1, total_claims=1),
            event("verification", current_claim=1, total_claims=1),
            event("claim_complete", current_claim=1, total_claims=1, result=result),
            event("complete", current_claim=1, total_claims=1),
        ]
    )


class ChatUITest(unittest.TestCase):
    def setUp(self):
        configuration_environment = {"CHATBOT_MODEL", "FACTCHECKER_MODEL", "TOKENIZER_NAME", "ES_HOST", "ES_DUMP_INDEX"}
        self.env_patch = patch.dict(
            os.environ,
            {key: value for key, value in os.environ.items() if key not in configuration_environment},
            clear=True,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.dotenv_patch = patch("dotenv.load_dotenv", return_value=False)
        self.dotenv_patch.start()
        self.addCleanup(self.dotenv_patch.stop)
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
        # AppTest does not schedule timed fragment reruns; advance ready jobs explicitly.
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
                try:
                    job.advance().result(timeout=3)
                except (Exception, StopException):
                    pass
            self.app.run()
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
        return "\n".join(
            [block.value for block in self.app.markdown]
            + [block.proto.dialog.title for block in self.app.get("dialog") if block.proto.dialog.is_open]
        )

    def assert_no_supplementary_sections(self):
        for expander in self.app.get("expander"):
            self.assertNotIn(expander.label, {"About verification labels", "Verification input"})
            self.assertFalse(expander.label.startswith("Extracted claims"))

    def assert_processing_finished(self):
        self.assertEqual(len(self.app.get("progress")), 0)
        self.assertEqual(len(self.app.get("status")), 0)
        self.assertNotIn('class="claim-spinner"', self.rendered_text())
        self.assertFalse(any("Fact-check complete" in caption.value for caption in self.app.caption))

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
        self.assertFalse(any("Fact-check complete" in caption.value for caption in self.app.caption))
        checked_message = [message for message in self.app.chat_message if message.name == "assistant"][0]
        passages = [expander for expander in checked_message.expander if expander.label == "Evidence passage"]
        self.assertEqual(len(passages), 1)
        self.assertFalse(passages[0].proto.expanded)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", "\n".join(block.value for block in passages[0].markdown))
        self.assertFalse(any(expander.label == "Source details" for expander in checked_message.expander))
        self.assertEqual(len(passages[0].text), 0)
        self.assertIn('class="evidence-source">test dataset</p>', self.rendered_text())
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
        first_run = deepcopy(self.check(first["id"], claim_result(labels=["Fully supported"])))
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
        first_run = deepcopy(self.check(first["id"], claim_result(first["content"], labels=["Fully supported"])))
        second_run = deepcopy(self.check(second["id"], claim_result(second["content"])))
        previous = self.check(first["id"], claim_result(first["content"], labels=["Fully refuted"]))
        self.assertEqual(previous, first_run)
        self.assertTrue(self.app.get("dialog")[0].proto.dialog.is_open)
        self.assertEqual(self.pipeline.run_factcheck.call_count, 2)
        self.assertIn("Run fact-check again?", self.rendered_text())
        self.app.button(key=f"confirm_factcheck_{first['id']}").click().run()
        self.assert_app_ok()
        rerun = self.app.session_state["factchecks"][first["id"]]
        self.assertEqual(len(rerun["results"]), 1)
        self.assertEqual(rerun["results"][0]["evidences"][0]["verification"]["label"], "Fully refuted")
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
        # Refresh AppTest's event-block tree after the dialog fragment closes.
        self.app.run()
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
        self.assertEqual(len(run["results"][0]["evidences"]), len(LABELS))
        content = self.rendered_text()
        summary = next(block.value for block in self.app.markdown if 'aria-label="Verdict summary"' in block.value)
        self.assertIn("Claims <strong>1</strong>", summary)
        self.assertIn("Check-worthy claims <strong>1</strong>", summary)
        self.assertIn("Verdicts <strong>6</strong>", summary)
        for label in LABELS:
            first, last = label.rsplit(" ", 1)
            self.assertIn(f"*{first}  \n{last}* **1**", [button.label for button in self.app.button])
        blocks = [block.value for block in self.app.markdown]
        self.assertLess(
            blocks.index(summary), next(i for i, block in enumerate(blocks) if 'class="claim-heading"' in block)
        )
        for label in LABELS:
            self.assertIn(f">{label}</span>", content)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", content)
        self.assertNotIn("<script>unsafe</script>", content)
        self.assertEqual(len(self.app.json), 0)
        self.assertFalse(any(expander.label == "Source details" for expander in self.app.expander))
        visible = content + "\n" + "\n".join(text.value for text in self.app.text)
        visible += "\n" + "\n".join(caption.value for caption in self.app.caption)
        self.assertNotIn("Relevance score", visible)
        self.assertNotIn("Training step", visible)
        self.assertEqual(content.count('class="evidence-source">test dataset</p>'), len(LABELS))
        self.assertLess(
            content.index('<div class="section-label">Rationale</div>'),
            content.index('<div class="section-label">Source</div>'),
        )
        evidence = run["results"][0]["evidences"][0]
        self.assertEqual(evidence["training_step"], 123)
        self.assertNotIn("meta", evidence)
        passages = [expander for expander in self.app.expander if expander.label == "Evidence passage"]
        self.assertEqual(len(passages), len(LABELS))
        for passage in passages:
            self.assertEqual(len(passage.text), 0)
            self.assertNotIn('class="evidence-source"', "\n".join(block.value for block in passage.markdown))
        self.assertTrue(all(not passage.proto.expanded for passage in passages))
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()

    def test_verdict_cards_filter_pairs_toggle_and_clear_without_rerunning_checks(self):
        self.send("Tell me about Japan.")
        response_id = self.assistants()[0]["id"]
        run = self.check(response_id, claim_result(labels=["Fully supported", "Fully refuted", "Fully supported"]))
        saved_results = deepcopy(run["results"])
        key = f"verdict_filter_refuted_{response_id}"
        self.app.button(key=key).click().run()
        self.assert_app_ok()
        self.assertEqual(run["verdict_filter"], "Fully refuted")
        self.assertEqual(len(self.app.expander), 1)
        self.assertIn('evidence-number">Evidence 2', self.rendered_text())
        self.assertNotIn('evidence-number">Evidence 1', self.rendered_text())
        self.assertIn("Verdicts <strong>3</strong>", self.rendered_text())
        self.assertEqual(self.app.button(key=key).proto.type, "primary")
        self.assertEqual(run["results"], saved_results)

        self.app.button(key=key).click().run()
        self.assert_app_ok()
        self.assertIsNone(run["verdict_filter"])
        self.assertEqual(len(self.app.expander), 3)

        self.app.button(key=f"verdict_filter_nei_{response_id}").click().run()
        self.assert_app_ok()
        self.assertEqual(len(self.app.expander), 0)
        self.assertTrue(any("No completed verdicts match" in info.value for info in self.app.info))
        self.app.button(key=f"clear_verdict_filter_{response_id}").click().run()
        self.assert_app_ok()
        self.assertEqual(len(self.app.expander), 3)
        self.pipeline.run_factcheck.assert_called_once()
        self.chat.stream_chat_response.assert_called_once()

    def test_filters_are_independent_for_each_response_and_reset_on_recheck(self):
        self.send("First question", "First response")
        first = self.assistants()[0]["id"]
        self.check(first, claim_result("First claim", ["Fully supported", "Fully refuted"]))
        self.app.button(key=f"verdict_filter_refuted_{first}").click().run()
        self.send("Second question", "Second response")
        second = self.assistants()[1]["id"]
        self.check(second, claim_result("Second claim", ["Fully supported", "Fully refuted"]))
        self.app.button(key=f"verdict_filter_supported_{second}").click().run()
        self.assert_app_ok()
        checks = self.app.session_state["factchecks"]
        self.assertEqual(checks[first]["verdict_filter"], "Fully refuted")
        self.assertEqual(checks[second]["verdict_filter"], "Fully supported")
        self.assertEqual(len(self.app.expander), 2)
        self.assertEqual(self.pipeline.run_factcheck.call_count, 2)
        self.check(first)
        self.pipeline.run_factcheck.return_value = completed_events(claim_result("Rechecked first claim"))
        self.app.button(key=f"confirm_factcheck_{first}").click().run()
        self.assert_app_ok()
        self.assertIsNone(checks[first].get("verdict_filter"))
        self.assertEqual(checks[second]["verdict_filter"], "Fully supported")

    def test_filter_excludes_unverified_claims_and_keeps_original_claim_numbers(self):
        self.send("Tell me about Japan.")
        response_id = self.assistants()[0]["id"]
        run = self.check(response_id, claim_result(labels=["Fully supported", "Not enough information"]))
        run["claim_states"] = [
            {"claim": "Skipped", "is_checkworthy": False, "evidences": []},
            {"claim": "Missing", "is_checkworthy": True, "no_evidence": True, "evidences": []},
            run["results"][0],
            {"claim": "Pending", "is_checkworthy": True, "evidences": [{"passage": "Pending"}]},
        ]
        run["claims"] = [result["claim"] for result in run["claim_states"]]
        self.app.button(key=f"verdict_filter_nei_{response_id}").click().run()
        self.assert_app_ok()
        text = self.rendered_text()
        self.assertIn('class="eyebrow">Claim 3', text)
        self.assertNotIn('class="eyebrow">Claim 1', text)
        self.assertNotIn('class="eyebrow">Claim 2', text)
        self.assertNotIn('class="eyebrow">Claim 4', text)
        self.assertIn('evidence-number">Evidence 2', text)
        self.assertEqual(len(self.app.expander), 1)
        self.assertIn("Claims <strong>4</strong>", text)
        self.assertIn("Check-worthy claims <strong>3</strong>", text)
        self.assertIn("Verdicts <strong>2</strong>", text)
        self.pipeline.run_factcheck.assert_called_once()

    def test_base_verdict_without_rationale_renders_without_an_empty_section(self):
        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        result = claim_result(labels=["Inferentially supported"])
        result["evidences"][0]["verification"].pop("rationale")
        self.check(message["id"], result)
        self.assertIn("Inferentially supported", self.rendered_text())
        self.assertNotIn("Rationale", self.rendered_text())

    def test_generation_error_keeps_user_and_retry_does_not_duplicate_it(self):
        def broken_stream(*_args, **_kwargs):
            yield "Incomplete response"
            raise RuntimeError("Chat connection failed")

        self.chat.stream_chat_response.side_effect = broken_stream
        with self.assertLogs(level="ERROR") as logs:
            self.app.chat_input[0].set_value("Tell me about Japan.").run()
            self.assert_app_ok()
        self.assertIn("Chat connection failed", "\n".join(logs.output))
        messages = deepcopy(self.app.session_state["messages"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(self.assistants(), [])
        self.assertNotIn("Chat connection failed", self.app.session_state["chat_error"])
        self.assertEqual(
            [error.value for error in self.app.error], ["Unable to generate a response. Please select Retry response."]
        )
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
        with self.assertLogs(level="ERROR") as logs:
            run = self.check(message["id"])
        self.assertIn("Verification connection failed", "\n".join(logs.output))
        self.assertEqual(run["state"], "error")
        self.assertEqual(len(run["results"]), 1)
        self.assert_no_supplementary_sections()
        self.assert_processing_finished()
        self.assertEqual(run["results"][0]["evidences"][0]["verification"]["label"], "Fully supported")
        self.assertNotIn("Verification connection failed", run["error"])
        self.assertEqual(
            [error.value for error in self.app.error],
            ["Unable to complete the fact-check. Please try again. Available results are shown below."],
        )
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

    def test_start_failure_is_logged_and_legacy_errors_are_not_displayed(self):
        detail = "ProtocolError('Connection aborted', ConnectionResetError(54, 'Connection reset by peer'))"
        self.send("Tell me about Japan.")
        response_id = self.assistants()[0]["id"]
        self.pipeline.PipelineConfig.side_effect = ConnectionError(detail)
        with self.assertLogs(level="ERROR") as logs:
            run = self.check(response_id)
        self.assertIn(detail, "\n".join(logs.output))
        self.assertEqual(run["state"], "error")
        self.assertEqual(
            [error.value for error in self.app.error],
            ["Unable to complete the fact-check. Please try again."],
        )
        self.assertNotIn(detail, run["error"])
        self.assert_processing_finished()

        # Sessions created before this change can still hold raw exception text.
        run["error"] = detail
        self.app.session_state["chat_error"] = detail
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(len(self.app.error), 2)
        for error in self.app.error:
            self.assertNotIn("ProtocolError", error.value)
            self.assertNotIn("ConnectionResetError", error.value)
            self.assertIn("Unable to", error.value)

    def test_claim_list_is_visible_immediately_after_decomposition(self):
        release = Event()

        def paused_events(*args):
            yield event("decomposition", claims=["First extracted claim", "Second extracted claim"], total_claims=2)
            release.wait(5)

        self.send("Tell me two facts.", "Response to check")
        self.pipeline.run_factcheck.side_effect = paused_events
        response_id = self.assistants()[0]["id"]
        try:
            self.app.button(key=f"factcheck_{response_id}").click().run()
            run = self.app.session_state["factchecks"][response_id]
            self.assertEqual(run["results"], [])
            self.assertEqual(run["state"], "running")
            text = self.rendered_text()
            self.assertLess(text.index("First extracted claim"), text.index("Second extracted claim"))
            self.assertIn("0 / 2 claims processed", text)
            self.assertEqual(text.count('class="claim-spinner"'), 2)
            self.assertEqual(sum(c.value == "Assessing check-worthiness…" for c in self.app.caption), 2)
            self.assertNotIn('class="verdict ', text)
        finally:
            release.set()
            for job in self.app.session_state["factcheck_jobs"].values():
                job.close()

    def test_pause_keeps_results_and_resume_continues_the_same_pipeline(self):
        started = Event()
        release = Event()
        result = claim_result("Claim retained while paused")

        def delayed_events(*args):
            yield event("decomposition", claims=[result["claim"]], total_claims=1)
            started.set()
            if not release.wait(5):
                raise TimeoutError("The in-flight operation was not released")
            yield event("claim_complete", current_claim=1, total_claims=1, result=result)
            yield event("complete", current_claim=1, total_claims=1)

        self.send("Tell me a fact.")
        response_id = self.assistants()[0]["id"]
        self.pipeline.run_factcheck.side_effect = delayed_events
        try:
            self.app.button(key=f"factcheck_{response_id}").click().run()
            for _ in range(10):
                if started.wait(0.02):
                    break
                self.app.run()
            self.assertTrue(started.is_set())
            self.app.button(key=f"pause_factcheck_{response_id}").click().run()
            self.assert_app_ok()
            run = self.app.session_state["factchecks"][response_id]
            self.assertEqual(run["state"], "paused")
            self.assertEqual(run["claims"], [result["claim"]])
            self.assertEqual(run["results"], [])
            self.assertEqual(len(self.app.get("progress")), 1)
            self.assertNotIn('class="claim-spinner"', self.rendered_text())
            self.assertNotIn('class="progress-spinner"', self.rendered_text())
            snapshot = deepcopy(run)
            release.set()
            job = self.app.session_state["factcheck_jobs"][response_id]
            job.advance().result(timeout=2)
            self.app.run()
            self.assertEqual(self.app.session_state["factchecks"][response_id], snapshot)
            self.app.button(key=f"resume_factcheck_{response_id}").click().run()
            self.assert_app_ok()
            run = self.app.session_state["factchecks"][response_id]
            self.assertEqual(run["state"], "complete")
            self.assertEqual(run["results"], [result])
            self.assertFalse(self.app.session_state["factcheck_jobs"])
            self.assertFalse(self.app.session_state.filtered_state.get("recheck_confirmation"))
            self.pipeline.run_factcheck.assert_called_once()
            self.assert_processing_finished()
        finally:
            release.set()
            for job in self.app.session_state.filtered_state.get("factcheck_jobs", {}).values():
                job.close()

    def test_retrieved_evidence_is_visible_before_any_verification_finishes(self):
        release = Event()
        result = claim_result("Retrieved claim")
        result["status"] = "verification"
        del result["evidences"][0]["verification"]

        def paused_events(*args):
            yield event("decomposition", claims=[result["claim"]], total_claims=1)
            yield event("retrieval", current_claim=1, total_claims=1, claim_update=result, progress=0.5)
            release.wait(5)

        self.send("Tell me a fact.", "Response to check")
        self.pipeline.run_factcheck.side_effect = paused_events
        response_id = self.assistants()[0]["id"]
        try:
            self.app.button(key=f"factcheck_{response_id}").click().run()
            run = self.app.session_state["factchecks"][response_id]
            self.assertEqual(run["results"], [])
            self.assertEqual([e.label for e in self.app.expander], ["Evidence passage"])
            self.assertTrue(any(c.value == "Waiting for verification…" for c in self.app.caption))
            self.assertIn('class="evidence-source">test dataset</p>', self.rendered_text())
            self.assertNotIn("Training step", self.rendered_text() + "\n".join(t.value for t in self.app.text))
            self.assertNotIn('class="verdict ', self.rendered_text())
            self.assertEqual(self.rendered_text().count('class="claim-spinner"'), 1)
            # Polling without new events must preserve the visible evidence.
            self.app.run()
            self.assertEqual([e.label for e in self.app.expander], ["Evidence passage"])
            self.assertIn('class="evidence-source">test dataset</p>', self.rendered_text())
        finally:
            release.set()
            for job in self.app.session_state["factcheck_jobs"].values():
                job.close()

    def test_partial_pair_verdict_and_later_claim_survive_failure_in_original_positions(self):
        partial = claim_result("First claim", ["Fully supported", "Fully refuted"])
        partial["status"] = "verification"
        del partial["evidences"][0]["verification"]
        finished = claim_result("Second claim")

        def failing_events(*args):
            yield event("decomposition", claims=["First claim", "Second claim"], total_claims=2)
            yield event("claim_complete", current_claim=2, total_claims=2, result=finished)
            yield event("verification", current_claim=1, total_claims=2, claim_update=partial, progress=0.8)
            raise RuntimeError("Remaining verification failed")

        self.send("Tell me two facts.", "Response to check")
        self.pipeline.run_factcheck.side_effect = failing_events
        run = self.check(self.assistants()[0]["id"])
        self.assertEqual(run["results"], [finished])
        self.assertEqual(run["claim_states"], [partial, finished])
        text = self.rendered_text()
        self.assertLess(text.index("First claim"), text.index("Second claim"))
        self.assertIn("1 / 2 claims processed", text)
        self.assertIn("Fully refuted", text)
        self.assertIn("Fully supported", text)
        self.assertTrue(any(c.value == "Verification did not complete." for c in self.app.caption))
        self.assertEqual(sum(e.label == "Evidence passage" for e in self.app.expander), 3)
        self.assert_processing_finished()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(
            self.app.session_state["factchecks"][self.assistants()[0]["id"]]["claim_states"], [partial, finished]
        )

    def test_interrupted_factcheck_requires_confirmation_before_retry(self):
        def interrupted_events(*_args, **_kwargs):
            yield event("decomposition", claims=["A claim"], total_claims=1)
            raise StopException()

        self.send("Tell me about Japan.")
        message = self.assistants()[0]
        self.pipeline.run_factcheck.side_effect = interrupted_events
        interrupted = deepcopy(self.check(message["id"]))
        self.assertEqual(interrupted["state"], "running")
        interrupted["state"] = "interrupted"
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

    def test_non_checkworthy_claim_is_shown_as_skipped_without_verdict(self):
        with patch.object(sys, "argv", [str(APP), "--checkworthiness-prompt", "custom-checkworthy.json"]):
            self.app = AppTest.from_file(str(APP), default_timeout=10).run()
            self.send("Your opinion?", "The museum is wonderful.")
            message = self.assistants()[0]
            skipped = {"claim": message["content"], "is_checkworthy": False, "no_evidence": False, "evidences": []}
            run = self.check(message["id"], skipped)
            self.assertEqual(run["state"], "complete")
            self.assertEqual(run["results"], [skipped])
            self.assertTrue(any("Not check-worthy" in caption.value for caption in self.app.caption))
            self.assertNotIn('class="verdict ', self.rendered_text())
            self.assertIn("1 / 1 claims processed", self.rendered_text())
            self.assertEqual(len(self.app.expander), 0)
            self.assert_processing_finished()
            config = self.pipeline.run_factcheck.call_args.args[2]
            self.assertEqual(config.checkworthiness_prompt, "custom-checkworthy.json")

    def test_checkworthiness_failure_is_reported_and_removes_progress(self):
        def failing_events(*args):
            yield event("decomposition", claims=["A claim"], total_claims=1)
            yield event("checkworthiness", total_claims=1)
            raise ValueError("Invalid check-worthiness labels")

        self.pipeline.run_factcheck.side_effect = failing_events
        self.send("Tell me a fact.")
        run = self.check(self.assistants()[0]["id"])
        self.assertEqual(run["state"], "error")
        self.assertEqual(run["results"], [])
        self.assertTrue(any("Unable to complete the fact-check" in error.value for error in self.app.error))
        self.assertFalse(any("Invalid check-worthiness labels" in error.value for error in self.app.error))
        self.assert_processing_finished()

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
        module = load_ui_module()
        args = module.parse_args(
            [
                "--decomposition-prompt",
                "decompose.json",
                "--checkworthiness-prompt",
                "checkworthy.json",
                "--verification-prompt",
                "verify.json",
                "--chat-engine",
                "chat-model",
                "--max-concurrency",
                "3",
            ]
        )
        self.assertEqual(args.decomposition_prompt, "decompose.json")
        self.assertEqual(args.checkworthiness_prompt, "checkworthy.json")
        self.assertEqual(args.verification_prompt, "verify.json")
        self.assertEqual(args.chat_engine, "chat-model")
        self.assertEqual(args.max_concurrency, 3)
        self.assertEqual(args.num_evidences, 3)
        with self.assertRaises(argparse.ArgumentTypeError):
            module.positive_integer("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            module.positive_integer("-2")
        self.assertEqual(module.positive_integer("3"), 3)

    def test_model_defaults_from_dotenv_respect_process_environment(self):
        module = load_ui_module()
        self.assertEqual(module.DOTENV_PATH, ROOT / ".env")
        cases = [
            ({}, "file-factchecker", "file-chatbot"),
            ({"FACTCHECKER_MODEL": " process-factchecker "}, "process-factchecker", "file-chatbot"),
            (
                {"FACTCHECKER_MODEL": "process-factchecker", "CHATBOT_MODEL": " process-chatbot "},
                "process-factchecker",
                "process-chatbot",
            ),
            ({"FACTCHECKER_MODEL": " \t ", "CHATBOT_MODEL": ""}, "gpt-5.4-2026-03-05", None),
        ]
        with TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                'FACTCHECKER_MODEL=" file-factchecker "\nCHATBOT_MODEL=" file-chatbot "\n', encoding="utf-8"
            )
            for environment, expected_engine, expected_chat_engine in cases:
                with (
                    self.subTest(environment=environment),
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(module, "DOTENV_PATH", dotenv_path),
                    patch("dotenv.load_dotenv", wraps=load_dotenv) as loader,
                ):
                    args = module.parse_args([])
                    loader.assert_called_once_with(dotenv_path, override=False)
                    self.assertEqual(args.engine, expected_engine)
                    self.assertEqual(args.chat_engine, expected_chat_engine)

    def test_mock_mode_ignores_environment_and_skips_dotenv(self):
        module = load_ui_module()
        environment = {
            "FACTCHECKER_MODEL": "environment-factchecker",
            "CHATBOT_MODEL": "environment-chatbot",
            "TOKENIZER_NAME": "unused-tokenizer",
            "ES_HOST": "http://unused:9200",
            "ES_DUMP_INDEX": "unused-index",
        }
        cases = [
            (["--mock"], "gpt-5.4-2026-03-05", None),
            (["--mock", "--engine", "cli-factchecker"], "cli-factchecker", None),
            (
                ["--mock", "--engine", "cli-factchecker", "--chat-engine", "cli-chatbot"],
                "cli-factchecker",
                "cli-chatbot",
            ),
        ]
        for argv, expected_engine, expected_chat_engine in cases:
            with (
                self.subTest(argv=argv),
                patch.dict(os.environ, environment),
                patch("dotenv.load_dotenv", side_effect=AssertionError("Mock mode must not read .env")) as loader,
            ):
                args = module.parse_args(argv)
                loader.assert_not_called()
                self.assertEqual(args.engine, expected_engine)
                self.assertEqual(args.chat_engine, expected_chat_engine)
                self.assertEqual(args.tokenizer_name, "llm-jp/llm-jp-3-13b")
                self.assertEqual(args.es_host, "http://10.2.73.12:9200")
                self.assertEqual(args.es_dump_index, "llm-jp-corpus-v3")

    def test_retrieval_configuration_precedence(self):
        module = load_ui_module()
        names = ("tokenizer_name", "es_host", "es_dump_index")
        defaults = ("llm-jp/llm-jp-3-13b", "http://10.2.73.12:9200", "llm-jp-corpus-v3")
        file_values = ("file-tokenizer", "http://file:9200", "file-index")
        process_values = ("process-tokenizer", "http://process:9200", "process-index")
        cli_values = ("cli-tokenizer", "http://cli:9200", "cli-index")
        environment = {name.upper(): f" {value} " for name, value in zip(names, process_values)}
        cases = [
            ([], {}, "", defaults),
            ([], {}, "dotenv", file_values),
            ([], environment, "dotenv", process_values),
            ([], {name.upper(): " \t " for name in names}, "dotenv", defaults),
            (
                [item for name, value in zip(names, cli_values) for item in (f"--{name}", value)],
                environment,
                "dotenv",
                cli_values,
            ),
            (
                ["--es-host", "http://override:9200"],
                environment,
                "dotenv",
                (process_values[0], "http://override:9200", process_values[2]),
            ),
        ]
        with TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            for argv, values, file_content, expected in cases:
                with (
                    self.subTest(argv=argv, environment=values, file_content=file_content),
                    patch.dict(os.environ, values, clear=True),
                    patch.object(module, "DOTENV_PATH", dotenv_path),
                    patch("dotenv.load_dotenv", wraps=load_dotenv),
                ):
                    dotenv_path.write_text(
                        "\n".join(f'{name.upper()}=" {value} "' for name, value in zip(names, file_values))
                        if file_content
                        else "",
                        encoding="utf-8",
                    )
                    args = module.parse_args(argv)
                    self.assertEqual(tuple(getattr(args, name) for name in names), expected)
                    self.assertFalse(hasattr(args, "es_meta_index"))

    def test_retrieval_environment_is_passed_to_factchecker(self):
        environment = {
            "TOKENIZER_NAME": "custom-tokenizer",
            "ES_HOST": "http://custom:9200",
            "ES_DUMP_INDEX": "custom-index",
        }
        with patch.dict(os.environ, environment):
            self.app = AppTest.from_file(str(APP), default_timeout=10).run()
            self.assert_app_ok()
            self.send("Tell me about Japan.")
            self.check(self.assistants()[0]["id"], claim_result())
            config = self.pipeline.run_factcheck.call_args.args[2]
            self.assertEqual(config.tokenizer_name, environment["TOKENIZER_NAME"])
            self.assertEqual(config.es_host, environment["ES_HOST"])
            self.assertEqual(config.es_dump_index, environment["ES_DUMP_INDEX"])
            self.assertFalse(hasattr(config, "es_meta_index"))

    def test_chat_model_selection_does_not_change_factchecker_model(self):
        environment = {"FACTCHECKER_MODEL": " env-factchecker ", "CHATBOT_MODEL": " env-chatbot "}
        cases = [
            ([], {}, "gpt-5.4-2026-03-05", "gpt-5.4-2026-03-05"),
            (["--engine", "cli-factchecker"], {}, "cli-factchecker", "cli-factchecker"),
            ([], environment, "env-factchecker", "env-chatbot"),
            ([], {"FACTCHECKER_MODEL": "env-factchecker"}, "env-factchecker", "env-factchecker"),
            ([], {"CHATBOT_MODEL": "env-chatbot"}, "gpt-5.4-2026-03-05", "env-chatbot"),
            (["--engine", "cli-factchecker"], environment, "cli-factchecker", "env-chatbot"),
            (["--chat-engine", "cli-chatbot"], environment, "env-factchecker", "cli-chatbot"),
            (
                ["--engine", "cli-factchecker", "--chat-engine", "cli-chatbot"],
                environment,
                "cli-factchecker",
                "cli-chatbot",
            ),
        ]
        for argv, model_environment, expected_engine, expected_chat_engine in cases:
            with (
                self.subTest(argv=argv, environment=model_environment),
                patch.dict(os.environ, model_environment),
                patch.object(sys, "argv", [str(APP), *argv]),
            ):
                self.app = AppTest.from_file(str(APP), default_timeout=10).run()
                self.assert_app_ok()
                self.send("Tell me about Japan.")
                chat_call = self.chat.stream_chat_response.call_args
                model = chat_call.kwargs.get("model") or chat_call.args[1]
                self.assertEqual(model, expected_chat_engine)
                self.assertEqual(self.assistants()[0]["model"], expected_chat_engine)
                self.assertIn(f'<div class="message-author">{expected_chat_engine}</div>', self.rendered_text())
                self.check(self.assistants()[0]["id"], claim_result())
                self.assertIn(f'<div class="message-author">{expected_chat_engine}</div>', self.rendered_text())
                config = self.pipeline.run_factcheck.call_args.args[2]
                self.assertEqual(config.engine, expected_engine)


class ResultSummaryTests(unittest.TestCase):
    def test_partial_pair_counts_exclude_skipped_missing_evidence_and_pending_outputs(self):
        module = load_ui_module()
        partial = claim_result(labels=["Fully supported", "Fully supported", "Not enough information"])
        partial["status"] = "verification"
        partial["evidences"].append({"passage": "Pending evidence"})
        skipped = {"claim": "Opinion", "is_checkworthy": False, "evidences": []}
        no_evidence = {"claim": "No source", "is_checkworthy": True, "no_evidence": True, "evidences": []}
        run = {"claim_states": [partial, skipped, no_evidence], "results": [skipped, no_evidence]}
        for state in ("running", "paused", "error", "interrupted"):
            with self.subTest(state=state):
                summary = module._summarize_results(dict(run, state=state))
                self.assertEqual(summary["verdicts"], 3)
                self.assertEqual(summary["labels"]["Fully supported"], 2)
                self.assertEqual(summary["labels"]["Not enough information"], 1)
                self.assertEqual(summary["labels"]["Fully refuted"], 0)
                self.assertEqual(summary["claims"], 3)
                self.assertEqual(summary["checkworthy_claims"], 2)

    def test_conflicting_verdicts_are_counted_independently_without_majority_vote(self):
        module = load_ui_module()
        result = claim_result(labels=["Fully supported", "Fully refuted", "Fully refuted"])
        summary = module._summarize_results({"claims": [result["claim"]], "claim_states": [result]})
        self.assertEqual(summary["claims"], 1)
        self.assertEqual(summary["checkworthy_claims"], 1)
        self.assertEqual(summary["verdicts"], 3)
        self.assertEqual(summary["labels"]["Fully supported"], 1)
        self.assertEqual(summary["labels"]["Fully refuted"], 2)
        self.assertEqual(sum(summary["labels"].values()), summary["verdicts"])

    def test_summary_handles_initial_and_historical_results(self):
        module = load_ui_module()
        initial = module._summarize_results({"claim_states": [{"claim": "Pending", "evidences": []}]})
        self.assertEqual(initial["verdicts"], 0)
        self.assertEqual(initial["claims"], 1)
        self.assertEqual(initial["checkworthy_claims"], 0)
        self.assertEqual(initial["labels"], dict.fromkeys(LABELS, 0))
        historical = module._summarize_results({"results": [claim_result(labels=LABELS)]})
        self.assertEqual(historical["verdicts"], 6)
        self.assertEqual(historical["claims"], 1)
        self.assertEqual(historical["checkworthy_claims"], 1)
        self.assertEqual(historical["labels"], dict.fromkeys(LABELS, 1))


if __name__ == "__main__":
    unittest.main()
