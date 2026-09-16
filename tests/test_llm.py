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

import decompose
import verify


def tool_call(name, arguments):
    return SimpleNamespace(type="function", function=SimpleNamespace(name=name, arguments=arguments))


def response_with_calls(calls):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))])


def response(name, payload):
    return response_with_calls([tool_call(name, json.dumps(payload, ensure_ascii=False))])


class LLMTests(unittest.TestCase):
    def mock_client(self, module, response_value):
        mocked = Mock()
        mocked.chat.completions.create.return_value = response_value
        patcher = patch.object(module, "get_client", return_value=mocked)
        getter = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked, getter

    def test_decomposes_with_context_and_forced_tool(self):
        mocked, getter = self.mock_client(
            decompose, response("createClaimList", {"claims": ["  美咲は大阪に住んでいる。  "]})
        )
        self.assertEqual(
            decompose.decompose_document_into_claims("彼女は大阪に住んでいる。", "deployment", "美咲について"),
            ["美咲は大阪に住んでいる。"],
        )
        getter.assert_called_once_with("factchecker")
        kwargs = mocked.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deployment")
        self.assertEqual(kwargs["tool_choice"]["function"]["name"], "createClaimList")
        self.assertIn("美咲について", kwargs["messages"][1]["content"])

    def test_decomposition_without_context_does_not_insert_none(self):
        mocked, _ = self.mock_client(decompose, response("createClaimList", {"claims": []}))
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
                mocked.chat.completions.create.return_value = response("createClaimList", payload)
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
                    "setVerificationResult", {"label": label, "rationale": " Evidence-based explanation. "}
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
                self.assertEqual(kwargs["tool_choice"]["function"]["name"], "setVerificationResult")
                self.assertEqual(
                    kwargs["tools"][0]["function"]["parameters"]["properties"]["label"]["enum"], list(labels)
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
                mocked.chat.completions.create.return_value = response("setVerificationResult", payload)
                with self.assertRaises(ValueError):
                    verify.verify_claim("Claim", "Evidence", "deployment")

    def test_malformed_tool_responses_fail_clearly_for_both_operations(self):
        cases = [
            (decompose, "createClaimList", lambda: decompose.decompose_document_into_claims("Text", "deployment")),
            (verify, "setVerificationResult", lambda: verify.verify_claim("Claim", "Evidence", "deployment")),
        ]
        for module, name, invoke in cases:
            mocked, _ = self.mock_client(module, None)
            malformed = [
                SimpleNamespace(choices=[]),
                SimpleNamespace(choices=[SimpleNamespace(message=None)]),
                response_with_calls(None),
                response_with_calls([]),
                response_with_calls([tool_call("wrongTool", "{}")]),
                response_with_calls([tool_call(name, "{}"), tool_call(name, "{}")]),
                response_with_calls([tool_call(name, "{")]),
                response_with_calls([tool_call(name, None)]),
                response_with_calls([tool_call(name, "[]")]),
            ]
            for ret in malformed:
                with self.subTest(module=module.__name__, response=ret):
                    mocked.chat.completions.create.return_value = ret
                    with self.assertRaises(ValueError):
                        invoke()

    def test_custom_prompts_reload_between_invocations(self):
        mocked, _ = self.mock_client(
            verify, response("setVerificationResult", {"label": "Supported", "rationale": "Reason"})
        )
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
            "import chat, decompose, verify; "
            "assert 'openai' not in sys.modules; assert 'dotenv' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
