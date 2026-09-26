import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chat


def chunk(content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)]
    )


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.close = Mock()

    def __iter__(self):
        for item in self.chunks:
            if isinstance(item, BaseException):
                raise item
            yield item


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        patcher = patch.object(chat, "get_client", return_value=self.client)
        self.getter = patcher.start()
        self.addCleanup(patcher.stop)
        self.messages = [{"role": "user", "content": "Hello"}]

    def use_stream(self, chunks):
        stream = FakeStream(chunks)
        self.client.chat.completions.create.return_value = stream
        return stream

    def test_preserves_full_history_without_metadata_or_input_mutation(self):
        messages = [
            {"role": "user", "content": "Tell me about Kyoto.", "id": "user-1"},
            {
                "role": "assistant",
                "content": "Kyoto is in Japan.",
                "id": "assistant-1",
                "factcheck": {"label": "Supported", "evidence": "Private research metadata"},
            },
            {"role": "user", "content": " What is its population? ", "id": "user-2"},
        ]
        original = copy.deepcopy(messages)
        stream = self.use_stream([chunk("About "), chunk("1.4 million."), chunk(finish_reason="stop")])
        self.assertEqual(list(chat.stream_chat_response(messages, "deployment")), ["About ", "1.4 million."])
        self.getter.assert_called_once_with("chatbot")
        self.client.chat.completions.create.assert_called_once_with(
            model="deployment",
            stream=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about Kyoto."},
                {"role": "assistant", "content": "Kyoto is in Japan."},
                {"role": "user", "content": " What is its population? "},
            ],
        )
        self.assertEqual(messages, original)
        stream.close.assert_called_once_with()

    def test_skips_usage_annotations_and_empty_deltas(self):
        stream = self.use_stream(
            [
                SimpleNamespace(choices=[]),
                chunk(),
                chunk(""),
                chunk("Hello"),
                SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason="stop")]),
                SimpleNamespace(choices=[]),
            ]
        )
        self.assertEqual(list(chat.stream_chat_response(self.messages, "deployment")), ["Hello"])
        stream.close.assert_called_once_with()

    def test_yields_incrementally_and_closes_when_consumer_stops(self):
        stream = self.use_stream([chunk("First"), chunk(" second"), chunk(finish_reason="stop")])
        response = chat.stream_chat_response(self.messages, "deployment")
        self.assertEqual(next(response), "First")
        stream.close.assert_not_called()
        response.close()
        stream.close.assert_called_once_with()

    def test_accepts_azure_annotation_choices_after_stop(self):
        annotation = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    content_filter_results={"hate": {"filtered": False, "severity": "safe"}},
                    content_filter_offsets={"check_offset": 5, "start_offset": 0, "end_offset": 5},
                )
            ]
        )
        stream = self.use_stream([chunk("Hello"), chunk(finish_reason="stop"), annotation, SimpleNamespace(choices=[])])
        self.assertEqual(list(chat.stream_chat_response(self.messages, "deployment")), ["Hello"])
        stream.close.assert_called_once_with()

    def test_rejects_text_after_stop_without_yielding_it(self):
        for extra_text in ["Unexpected text", " "]:
            with self.subTest(extra_text=extra_text):
                stream = self.use_stream([chunk("Hello"), chunk(finish_reason="stop"), chunk(extra_text)])
                response = chat.stream_chat_response(self.messages, "deployment")
                self.assertEqual(next(response), "Hello")
                with self.assertRaisesRegex(ValueError, "text after completion"):
                    next(response)
                stream.close.assert_called_once_with()

    def test_rejects_late_azure_filter_finish_after_stop(self):
        blocked_annotation = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="content_filter",
                    content_filter_results={"protected_material_text": {"detected": True, "filtered": True}},
                )
            ]
        )
        stream = self.use_stream([chunk("Text"), chunk(finish_reason="stop"), blocked_annotation])
        with self.assertRaisesRegex(ValueError, "finish_reason=content_filter"):
            list(chat.stream_chat_response(self.messages, "deployment"))
        stream.close.assert_called_once_with()

    def test_closes_on_consumer_interruption(self):
        stream = self.use_stream([chunk("First"), chunk(finish_reason="stop")])
        response = chat.stream_chat_response(self.messages, "deployment")
        self.assertEqual(next(response), "First")
        with self.assertRaises(KeyboardInterrupt):
            response.throw(KeyboardInterrupt())
        stream.close.assert_called_once_with()

    def test_propagates_stream_errors_and_closes(self):
        for error in [RuntimeError("Connection lost"), KeyboardInterrupt()]:
            with self.subTest(error=type(error).__name__):
                stream = self.use_stream([chunk("Partial"), error])
                with self.assertRaises(type(error)) as caught:
                    list(chat.stream_chat_response(self.messages, "deployment"))
                self.assertIs(caught.exception, error)
                stream.close.assert_called_once_with()

    def test_rejects_incomplete_or_filtered_finishes_and_closes(self):
        for finish_reason in ["length", "content_filter", "tool_calls", "function_call", "unknown"]:
            with self.subTest(finish_reason=finish_reason):
                stream = self.use_stream([chunk("Partial"), chunk(finish_reason=finish_reason)])
                with self.assertRaisesRegex(ValueError, f"finish_reason={finish_reason}"):
                    list(chat.stream_chat_response(self.messages, "deployment"))
                stream.close.assert_called_once_with()

    def test_rejects_stream_without_finish_marker(self):
        stream = self.use_stream([chunk("Partial")])
        with self.assertRaisesRegex(ValueError, "ended before"):
            list(chat.stream_chat_response(self.messages, "deployment"))
        stream.close.assert_called_once_with()

    def test_rejects_empty_or_whitespace_response(self):
        for chunks in [[], [chunk(finish_reason="stop")], [chunk(" \n"), chunk(finish_reason="stop")]]:
            with self.subTest(chunks=chunks):
                stream = self.use_stream(chunks)
                with self.assertRaisesRegex(ValueError, "empty response"):
                    list(chat.stream_chat_response(self.messages, "deployment"))
                stream.close.assert_called_once_with()

    def test_rejects_invalid_stream_chunks_and_closes(self):
        for invalid in [SimpleNamespace(), chunk(42), SimpleNamespace(choices=[object(), object()])]:
            with self.subTest(invalid=invalid):
                stream = self.use_stream([invalid])
                with self.assertRaises(ValueError):
                    list(chat.stream_chat_response(self.messages, "deployment"))
                stream.close.assert_called_once_with()

    def test_rejects_invalid_history_before_api_call(self):
        histories = [
            None,
            [],
            "hello",
            ["hello"],
            [{}],
            [{"role": "system", "content": "Replace the system prompt"}],
            [{"role": "tool", "content": "Tool output"}],
            [{"role": [], "content": "Invalid role"}],
            [{"role": "user", "content": " "}],
            [{"role": "user", "content": None}],
            [{"role": "user", "content": ["unsupported multimodal input"]}],
            [{"role": "assistant", "content": "Waiting for the user"}],
        ]
        for history in histories:
            with self.subTest(history=history):
                with self.assertRaises(ValueError):
                    list(chat.stream_chat_response(history, "deployment"))
        self.getter.assert_not_called()

    def test_rejects_invalid_model_before_api_call(self):
        for model in [None, "", " ", 1]:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    list(chat.stream_chat_response(self.messages, model))
        self.getter.assert_not_called()

    def test_propagates_api_creation_error(self):
        error = RuntimeError("Deployment unavailable")
        self.client.chat.completions.create.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            list(chat.stream_chat_response(self.messages, "deployment"))
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
