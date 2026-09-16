import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import checkworthy
import decompose
import verify


def response_with_content(content, finish_reason="stop", refusal=None, tool_calls=None):
    message = SimpleNamespace(content=content, refusal=refusal, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def response(payload):
    return response_with_content(json.dumps(payload, ensure_ascii=False))


class LLMTests(unittest.TestCase):
    def mock_client(self, module, response_value):
        mocked = Mock()
        mocked.chat.completions.create.return_value = response_value
        patcher = patch.object(module, "get_client", return_value=mocked)
        getter = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked, getter

    def test_decomposes_with_context_and_json_schema(self):
        mocked, getter = self.mock_client(decompose, response({"claims": ["  美咲は大阪に住んでいる。  "]}))
        self.assertEqual(
            decompose.decompose_document_into_claims("彼女は大阪に住んでいる。", "deployment", "美咲について"),
            ["美咲は大阪に住んでいる。"],
        )
        getter.assert_called_once_with("factchecker")
        kwargs = mocked.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deployment")
        self.assertEqual(kwargs["response_format"]["json_schema"]["name"], "decomposition")
        self.assertIn("美咲について", kwargs["messages"][1]["content"])

    def test_decomposition_without_context_does_not_insert_none(self):
        mocked, _ = self.mock_client(decompose, response({"claims": []}))
        self.assertEqual(decompose.decompose_document_into_claims("No assertions", "deployment"), [])
        self.assertNotIn("None", mocked.chat.completions.create.call_args.kwargs["messages"][1]["content"])

    def test_empty_document_needs_no_client(self):
        _, getter = self.mock_client(decompose, None)
        self.assertEqual(decompose.decompose_document_into_claims(" \n", "deployment"), [])
        getter.assert_not_called()

    def test_invalid_decomposition_payloads(self):
        mocked, _ = self.mock_client(decompose, None)
        for payload in [{}, [], {"claims": "one"}, {"claims": [None]}, {"claims": [" "]}, {"claims": [], "extra": 1}]:
            with self.subTest(payload=payload):
                mocked.chat.completions.create.return_value = response(payload)
                with self.assertRaises(ValueError):
                    decompose.decompose_document_into_claims("Text", "deployment")

    def test_accepts_each_of_the_five_labels_for_one_evidence(self):
        labels = ("Supported", "Partially supported", "Partially refuted", "Refuted", "Not enough information")
        self.assertEqual(verify.VERIFICATION_LABELS, labels)
        mocked, getter = self.mock_client(verify, None)
        for label in labels:
            with self.subTest(label=label):
                getter.reset_mock()
                mocked.chat.completions.create.return_value = response(
                    {"label": label, "rationale": " Evidence-based explanation. "}
                )
                self.assertEqual(
                    verify.verify_claim("Claim A", "Evidence B", "deployment"),
                    {"label": label, "rationale": "Evidence-based explanation."},
                )
                getter.assert_called_once_with("factchecker")
                kwargs = mocked.chat.completions.create.call_args.kwargs
                self.assertEqual(kwargs["model"], "deployment")
                self.assertIn("Claim A", kwargs["messages"][1]["content"])
                self.assertIn("Evidence B", kwargs["messages"][1]["content"])
                self.assertEqual(kwargs["response_format"]["json_schema"]["name"], "verification")
                self.assertEqual(
                    kwargs["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"], list(labels)
                )

    def test_empty_evidence_is_nei_without_api_call(self):
        _, getter = self.mock_client(verify, None)
        result = verify.verify_claim("Claim", " \n", "deployment")
        self.assertEqual(result["label"], "Not enough information")
        self.assertTrue(result["rationale"])
        getter.assert_not_called()

    def test_verification_rejects_evidence_lists_and_blank_claims(self):
        _, getter = self.mock_client(verify, None)
        with self.assertRaises(TypeError):
            verify.verify_claim("Claim", ["one", "two"], "deployment")
        with self.assertRaises(ValueError):
            verify.verify_claim(" ", "Evidence", "deployment")
        getter.assert_not_called()

    def test_invalid_verification_payloads(self):
        mocked, _ = self.mock_client(verify, None)
        payloads = [
            {},
            [],
            {"label": True, "rationale": "Old boolean"},
            {"label": None, "rationale": "Old null"},
            {"label": "supported", "rationale": "Wrong case"},
            {"label": "Supported", "rationale": None},
            {"label": "Supported", "rationale": " "},
            {"label": "Supported", "rationale": "Explanation", "extra": True},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                mocked.chat.completions.create.return_value = response(payload)
                with self.assertRaises(ValueError):
                    verify.verify_claim("Claim", "Evidence", "deployment")

    def test_checkworthiness_judges_one_claim_using_factchecker(self):
        mocked, getter = self.mock_client(checkworthy, response({"label": False}))
        claim = "This film is wonderful."
        self.assertIs(checkworthy.identify_checkworthiness(claim, "deployment"), False)
        getter.assert_called_once_with("factchecker")
        kwargs = mocked.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deployment")
        self.assertIn(claim, kwargs["messages"][1]["content"])
        self.assertNotIn("The museum opened in 2012.", kwargs["messages"][1]["content"])
        self.assertEqual(kwargs["response_format"]["json_schema"]["name"], "checkworthiness")
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"]["properties"]["label"]["type"], "boolean")

    def test_empty_or_invalid_checkworthiness_inputs_make_no_api_call(self):
        _, getter = self.mock_client(checkworthy, None)
        for claim in [None, [], ["Claim"], ("Claim",), 1]:
            with self.subTest(claim=claim), self.assertRaises(TypeError):
                checkworthy.identify_checkworthiness(claim, "deployment")
        for claim in ["", " \n "]:
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                checkworthy.identify_checkworthiness(claim, "deployment")
        getter.assert_not_called()

    def test_invalid_checkworthiness_labels_are_rejected(self):
        mocked, _ = self.mock_client(checkworthy, None)
        for payload in [
            {},
            {"labels": [True]},
            {"label": "true"},
            {"label": 1},
            {"label": None},
            {"label": []},
            {"label": True, "extra": 1},
        ]:
            with self.subTest(payload=payload):
                mocked.chat.completions.create.return_value = response(payload)
                with self.assertRaises(ValueError):
                    checkworthy.identify_checkworthiness("Claim A", "deployment")

    def test_checkworthiness_prompt_can_be_replaced_and_reloads(self):
        mocked, _ = self.mock_client(checkworthy, response({"label": True}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            for instruction in ["First instructions", "Updated instructions"]:
                path.write_text(json.dumps({"system": instruction, "user": "Claim: {{claim}}"}), encoding="utf-8")
                checkworthy.identify_checkworthiness("Literal {{claim}} text", "deployment", prompt_path=path)
                messages = mocked.chat.completions.create.call_args.kwargs["messages"]
                self.assertEqual(messages[0]["content"], instruction)
                self.assertEqual(messages[1]["content"], "Claim: Literal {{claim}} text")

    def test_invalid_checkworthiness_prompt_prevents_client_creation(self):
        _, getter = self.mock_client(checkworthy, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"system": "Instructions", "user": "{{document}}"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown placeholders"):
                checkworthy.identify_checkworthiness("Claim", "deployment", prompt_path=path)
        getter.assert_not_called()

    def test_malformed_json_responses_fail_clearly_for_all_operations(self):
        cases = [
            (decompose, lambda: decompose.decompose_document_into_claims("Text", "deployment")),
            (checkworthy, lambda: checkworthy.identify_checkworthiness("Claim", "deployment")),
            (verify, lambda: verify.verify_claim("Claim", "Evidence", "deployment")),
        ]
        for module, invoke in cases:
            mocked, _ = self.mock_client(module, None)
            malformed = [
                SimpleNamespace(choices=[]),
                SimpleNamespace(choices=[SimpleNamespace(), SimpleNamespace()]),
                SimpleNamespace(choices=[SimpleNamespace(message=None)]),
                response_with_content(None),
                response_with_content(""),
                response_with_content(" \n "),
                response_with_content("{"),
                response_with_content("[]"),
                response_with_content("null"),
                response_with_content("```json\n{}\n```"),
            ]
            for ret in malformed:
                with self.subTest(module=module.__name__, response=ret):
                    mocked.chat.completions.create.return_value = ret
                    with self.assertRaises(ValueError):
                        invoke()

    def test_refusals_partial_outputs_and_tool_calls_are_rejected(self):
        cases = [
            (decompose, {"claims": ["Claim"]}, lambda: decompose.decompose_document_into_claims("Text", "deployment")),
            (checkworthy, {"label": True}, lambda: checkworthy.identify_checkworthiness("Claim", "deployment")),
            (
                verify,
                {"label": "Supported", "rationale": "Reason"},
                lambda: verify.verify_claim("Claim", "Evidence", "deployment"),
            ),
        ]
        for module, payload, invoke in cases:
            mocked, _ = self.mock_client(module, None)
            for options, expected_error in [
                ({"refusal": "Cannot comply"}, "refused"),
                ({"finish_reason": "length"}, "length"),
                ({"finish_reason": "content_filter"}, "content_filter"),
                ({"finish_reason": None}, "finish_reason"),
                ({"finish_reason": "tool_calls", "tool_calls": [SimpleNamespace()]}, "tool_calls"),
                ({"tool_calls": [SimpleNamespace()]}, "tool calls"),
            ]:
                with self.subTest(module=module.__name__, options=options):
                    mocked.chat.completions.create.return_value = response_with_content(json.dumps(payload), **options)
                    with self.assertRaisesRegex(ValueError, expected_error):
                        invoke()

    def test_each_task_requests_strict_json_schema_without_tools(self):
        cases = [
            (
                decompose,
                "decomposition",
                {"claims": []},
                lambda: decompose.decompose_document_into_claims("Text", "deployment"),
            ),
            (
                checkworthy,
                "checkworthiness",
                {"label": True},
                lambda: checkworthy.identify_checkworthiness("Claim", "deployment"),
            ),
            (
                verify,
                "verification",
                {"label": "Supported", "rationale": "Reason"},
                lambda: verify.verify_claim("Claim", "Evidence", "deployment"),
            ),
        ]
        for module, name, payload, invoke in cases:
            with self.subTest(task=name):
                mocked, _ = self.mock_client(module, response(payload))
                invoke()
                kwargs = mocked.chat.completions.create.call_args.kwargs
                self.assertNotIn("tools", kwargs)
                self.assertNotIn("tool_choice", kwargs)
                self.assertEqual(kwargs["response_format"]["type"], "json_schema")
                definition = kwargs["response_format"]["json_schema"]
                self.assertEqual(definition["name"], name)
                self.assertIs(definition["strict"], True)
                self.assertEqual(definition["schema"]["type"], "object")
                self.assertIs(definition["schema"]["additionalProperties"], False)
                self.assertEqual(set(definition["schema"]["required"]), set(payload))
                self.assertEqual(set(definition["schema"]["properties"]), set(payload))

    def test_custom_prompts_reload_between_invocations(self):
        mocked, _ = self.mock_client(verify, response({"label": "Supported", "rationale": "Reason"}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            for instruction in ["First instructions", "Replaced instructions"]:
                path.write_text(
                    json.dumps({"system": instruction, "user": "{{claim}} / {{evidence}}"}), encoding="utf-8"
                )
                verify.verify_claim("C", "E", "deployment", prompt_path=path)
                self.assertEqual(mocked.chat.completions.create.call_args.kwargs["messages"][0]["content"], instruction)

    def test_invalid_prompt_prevents_client_creation(self):
        _, getter = self.mock_client(decompose, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"system": "Instructions", "user": "{{unknown}}"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown placeholders"):
                decompose.decompose_document_into_claims("Text", "deployment", prompt_path=path)
        getter.assert_not_called()

    def test_importing_backend_does_not_load_azure_or_dotenv(self):
        code = (
            f"import sys; sys.path.insert(0, {str(SRC)!r}); "
            "import chat, checkworthy, decompose, verify; "
            "assert 'openai' not in sys.modules; assert 'dotenv' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
