import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompts import PROMPT_DIR, render_prompt


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "prompt.json"

    def write(self, template):
        self.path.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8")

    def test_default_templates_have_required_inputs(self):
        cases = [
            ("decomposition.json", {"document": "本文", "context": "背景"}, {"document"}),
            ("checkworthiness.json", {"claims": "- Claim A\n- Claim B"}, {"claims"}),
            ("verification.json", {"claim": "主張", "evidence": "証拠"}, {"claim", "evidence"}),
        ]
        for name, values, required in cases:
            with self.subTest(name=name):
                messages = render_prompt(PROMPT_DIR / name, values, required)
                self.assertEqual([message["role"] for message in messages], ["system", "user"])

    def test_json_examples_and_source_placeholders_remain_literal(self):
        self.write({"system": 'Example: {"claims": ["日本語"]}', "user": "{{document}} / {{context}}"})
        messages = render_prompt(self.path, {"document": "本文 {{context}} {literal}", "context": "背景"}, {"document"})
        self.assertEqual(messages[0]["content"], 'Example: {"claims": ["日本語"]}')
        self.assertEqual(messages[1]["content"], "本文 {{context}} {literal} / 背景")

    def test_reloads_file_on_every_render(self):
        self.write({"system": "First", "user": "{{document}}"})
        self.assertEqual(render_prompt(self.path, {"document": "text"}, {"document"})[0]["content"], "First")
        self.write({"system": "Changed", "user": "{{document}}"})
        self.assertEqual(render_prompt(self.path, {"document": "text"}, {"document"})[0]["content"], "Changed")

    def test_placeholders_may_appear_in_system(self):
        self.write({"system": "Claim: {{claim}}", "user": "Evidence: {{evidence}}"})
        messages = render_prompt(self.path, {"claim": "A", "evidence": "B"}, {"claim", "evidence"})
        self.assertEqual(messages[0]["content"], "Claim: A")

    def test_missing_and_unknown_placeholders_are_rejected(self):
        for user, expected in [("No input", "missing placeholders"), ("{{document}} {{unknown}}", "unknown")]:
            with self.subTest(user=user):
                self.write({"system": "Instructions", "user": user})
                with self.assertRaisesRegex(ValueError, expected):
                    render_prompt(self.path, {"document": "text"}, {"document"})

    def test_invalid_prompt_files_are_rejected(self):
        templates = [
            [],
            {"system": "Instructions"},
            {"system": "Instructions", "user": "{{document}}", "other": "metadata"},
            {"system": 1, "user": "{{document}}"},
            {"system": " ", "user": "{{document}}"},
        ]
        for template in templates:
            with self.subTest(template=template):
                self.write(template)
                with self.assertRaises(ValueError):
                    render_prompt(self.path, {"document": "text"}, {"document"})
        self.path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot load prompt"):
            render_prompt(self.path, {}, set())
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "Cannot load prompt"):
            render_prompt(self.path, {}, set())


if __name__ == "__main__":
    unittest.main()
