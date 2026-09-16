import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import AzureOpenAI, BadRequestError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chat
import checkworthy
import clients
import decompose
import verify


def role_env(role, api_type="openai"):
    prefix = role.upper()
    return {
        f"{prefix}_API_TYPE": api_type,
        f"{prefix}_ENDPOINT": f"https://{role}.example.test" + ("/v1" if api_type == "openai" else "/"),
        f"{prefix}_API_KEY": f"{role}-key",
        f"{prefix}_API_VERSION": f"{role}-version",
    }


LEGACY_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://legacy.example.test",
    "AZURE_OPENAI_API_KEY": "legacy-key",
    "AZURE_OPENAI_API_VERSION": "legacy-version",
}


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.created_clients = []
        self.addCleanup(self.close_clients)
        for patcher in (
            patch.dict(os.environ, {}, clear=True),
            patch("dotenv.load_dotenv"),
            patch("openai.OpenAI", side_effect=lambda **kwargs: self.create_client(OpenAI, **kwargs)),
            patch("openai.AzureOpenAI", side_effect=lambda **kwargs: self.create_client(AzureOpenAI, **kwargs)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        clients.get_client.cache_clear()

    def close_clients(self):
        for client in self.created_clients:
            client.close()
        self.created_clients.clear()
        clients.get_client.cache_clear()

    def create_client(self, client_class, **kwargs):
        client = client_class(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(self.respond)), max_retries=0
        )
        self.created_clients.append(client)
        return client

    def respond(self, request):
        self.requests.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "Kyoto is in Japan."}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
        schema_name = body.get("response_format", {}).get("json_schema", {}).get("name")
        message = {"role": "assistant", "content": "OK"}
        if schema_name:
            payloads = {
                "decomposition": {"claims": ["Kyoto is in Japan."]},
                "checkworthiness": {"labels": [True]},
                "verification": {"label": "Supported", "rationale": "The evidence states the claim."},
            }
            message["content"] = json.dumps(payloads[schema_name])
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    def assert_route(self, request, role, api_type, model):
        self.assertEqual(request.url.host, f"{role}.example.test")
        self.assertEqual(json.loads(request.content)["model"], model)
        self.assertNotIn("OpenAI-Organization", request.headers)
        self.assertNotIn("OpenAI-Project", request.headers)
        if api_type == "azure":
            self.assertEqual(request.url.path, f"/openai/deployments/{model}/chat/completions")
            self.assertEqual(request.url.params["api-version"], f"{role}-version")
            self.assertEqual(request.headers["api-key"], f"{role}-key")
            self.assertNotIn("Authorization", request.headers)
        else:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertFalse(request.url.params)
            self.assertNotIn("api-key", request.headers)
            self.assertEqual(request.headers["Authorization"], f"Bearer {role}-key")

    def test_both_roles_support_each_provider_with_isolated_urls_and_auth(self):
        ambient = {
            **LEGACY_ENV,
            "AZURE_OPENAI_AD_TOKEN": "unrelated-ad-token",
            "OPENAI_API_KEY": "unrelated-openai-key",
            "OPENAI_BASE_URL": "https://unrelated.example.test/v1",
            "OPENAI_ORG_ID": "unrelated-org",
            "OPENAI_PROJECT_ID": "unrelated-project",
            "OPENAI_API_VERSION": "unrelated-version",
        }
        for chatbot_type in ("openai", "azure"):
            for factchecker_type in ("openai", "azure"):
                with self.subTest(chatbot=chatbot_type, factchecker=factchecker_type):
                    self.close_clients()
                    self.requests.clear()
                    with patch.dict(
                        os.environ,
                        {**ambient, **role_env("chatbot", chatbot_type), **role_env("factchecker", factchecker_type)},
                    ):
                        for role, api_type in (("chatbot", chatbot_type), ("factchecker", factchecker_type)):
                            client = clients.get_client(role)
                            self.assertIs(client, clients.get_client(role))
                            client.chat.completions.create(model=f"{role}-model", messages=[])
                            self.assert_route(self.requests[-1], role, api_type, f"{role}-model")
                        self.assertIsNot(clients.get_client("chatbot"), clients.get_client("factchecker"))
                        self.assertEqual(len(self.created_clients), 2)

    def test_real_callers_route_structured_outputs_to_each_factchecker_provider(self):
        for api_type in ("openai", "azure"):
            with (
                self.subTest(api_type=api_type),
                patch.dict(os.environ, {**role_env("chatbot"), **role_env("factchecker", api_type)}),
            ):
                self.close_clients()
                self.requests.clear()
                answer = "".join(
                    chat.stream_chat_response([{"role": "user", "content": "Where is Kyoto?"}], "chat-model")
                )
                claims = decompose.decompose_document_into_claims(answer, "check-model")
                labels = checkworthy.identify_checkworthiness(claims, "check-model")
                verdict = verify.verify_claim(claims[0], "Kyoto is in Japan.", "check-model")
                self.assertEqual(answer, "Kyoto is in Japan.")
                self.assertEqual(claims, [answer])
                self.assertEqual(labels, [True])
                self.assertEqual(verdict["label"], "Supported")
                self.assertEqual(len(self.requests), 4)
                self.assert_route(self.requests[0], "chatbot", "openai", "chat-model")
                self.assertNotIn("response_format", json.loads(self.requests[0].content))
                for request, name in zip(
                    self.requests[1:], ["decomposition", "checkworthiness", "verification"], strict=True
                ):
                    self.assert_route(request, "factchecker", api_type, "check-model")
                    body = json.loads(request.content)
                    self.assertNotIn("tools", body)
                    self.assertNotIn("tool_choice", body)
                    self.assertEqual(body["response_format"]["type"], "json_schema")
                    self.assertEqual(body["response_format"]["json_schema"]["name"], name)
                    self.assertIs(body["response_format"]["json_schema"]["strict"], True)
                self.assertEqual(len(self.created_clients), 2)

    def test_unsupported_structured_outputs_reports_error_without_fallback(self):
        with (
            patch.dict(os.environ, role_env("factchecker")),
            patch.object(
                self,
                "respond",
                return_value=httpx.Response(
                    400, json={"error": {"message": "json_schema is not supported", "type": "invalid_request_error"}}
                ),
            ) as respond,
        ):
            with self.assertRaisesRegex(BadRequestError, "json_schema is not supported"):
                decompose.decompose_document_into_claims("Text", "unsupported-model")
            respond.assert_called_once()

    def test_unconfigured_role_uses_legacy_azure_settings(self):
        with patch.dict(os.environ, LEGACY_ENV):
            for role in ("chatbot", "factchecker"):
                config = clients.APIConfig.from_env(role)
                self.assertEqual(config.api_type, "azure")
                self.assertEqual(config.endpoint, LEGACY_ENV["AZURE_OPENAI_ENDPOINT"])
                self.assertEqual(config.api_key, LEGACY_ENV["AZURE_OPENAI_API_KEY"])
                self.assertEqual(config.api_version, LEGACY_ENV["AZURE_OPENAI_API_VERSION"])

    def test_explicit_chatbot_can_coexist_with_legacy_factchecker(self):
        with patch.dict(os.environ, {**LEGACY_ENV, **role_env("chatbot")}):
            self.assertEqual(clients.APIConfig.from_env("chatbot").api_type, "openai")
            self.assertEqual(clients.APIConfig.from_env("factchecker").api_key, "legacy-key")

    def test_chat_does_not_require_factchecker_configuration(self):
        with patch.dict(os.environ, role_env("chatbot")):
            self.assertIsInstance(clients.get_client("chatbot"), OpenAI)
            self.assertEqual(len(self.created_clients), 1)

    def test_partial_role_configuration_never_inherits_legacy_credentials(self):
        for field in clients.CONNECTION_FIELDS:
            with self.subTest(field=field), patch.dict(os.environ, {**LEGACY_ENV, f"CHATBOT_{field}": ""}):
                with self.assertRaisesRegex(ValueError, "CHATBOT_API_TYPE"):
                    clients.get_client("chatbot")
        self.assertFalse(self.created_clients)

    def test_missing_role_values_report_variable_without_values(self):
        for role in ("chatbot", "factchecker"):
            for field in clients.CONNECTION_FIELDS:
                env = role_env(role, "azure")
                del env[f"{role.upper()}_{field}"]
                with self.subTest(role=role, field=field), patch.dict(os.environ, {**LEGACY_ENV, **env}):
                    with self.assertRaisesRegex(ValueError, f"{role.upper()}_{field}") as raised:
                        clients.get_client(role)
                    self.assertNotIn("legacy-key", str(raised.exception))
                    self.assertNotIn(f"{role}-key", str(raised.exception))
        self.assertFalse(self.created_clients)

    def test_openai_ignores_azure_version_and_accepts_custom_base_paths(self):
        for endpoint in ("http://localhost:8000/v1/", "https://resource.example.test/openai/v1/"):
            with (
                self.subTest(endpoint=endpoint),
                patch.dict(os.environ, {**role_env("chatbot"), "CHATBOT_ENDPOINT": endpoint}),
            ):
                config = clients.APIConfig.from_env("chatbot")
                self.assertEqual(config.endpoint, endpoint.rstrip("/"))
                self.assertIsNone(config.api_version)
                self.assertNotIn("chatbot-key", repr(config))

    def test_invalid_endpoint_and_api_type_fail_before_client_creation(self):
        invalid = (
            "resource.example.test",
            "ftp://resource.example.test",
            "https://user:secret@resource.example.test",
            "https://resource.example.test?key=secret",
            "https://resource.example.test/#secret",
            "https://resource.example.test:invalid",
            "https://[invalid",
        )
        for endpoint in invalid:
            with (
                self.subTest(endpoint=endpoint),
                patch.dict(os.environ, {**role_env("chatbot"), "CHATBOT_ENDPOINT": endpoint}),
            ):
                with self.assertRaisesRegex(ValueError, "CHATBOT_ENDPOINT") as raised:
                    clients.get_client("chatbot")
                self.assertNotIn("secret", str(raised.exception))
        with patch.dict(os.environ, {**role_env("chatbot"), "CHATBOT_API_TYPE": "unsupported"}):
            with self.assertRaisesRegex(ValueError, "CHATBOT_API_TYPE"):
                clients.get_client("chatbot")
        self.assertFalse(self.created_clients)

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Client role"):
            clients.get_client("other")
        self.assertFalse(self.created_clients)


if __name__ == "__main__":
    unittest.main()
