import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI
from openrouter import OpenRouter

from handbook_graph import NodeModels, ScopeDecision, ask, build_graph
from node_models import OllamaStructuredModel, create_models, create_node_model
from test_graph import HIT, ScriptedModel


class NodeModelTests(unittest.TestCase):
    def test_default_roles_and_token_limits(self):
        with patch.dict(os.environ, {"ROUTERAI_API_KEY": "test-routerai"}, clear=True):
            models = create_models()
        for model in (models.supervisor, models.research, models.writer, models.guardrails):
            self.assertIsInstance(model, ChatOpenAI)
            self.assertEqual(model.openai_api_key.get_secret_value(), "test-routerai")
            self.assertEqual(str(model.root_client.base_url), "https://routerai.ru/api/v1/")
            self.assertFalse(model.use_responses_api)
        self.assertEqual(models.supervisor.model_name, "google/gemma-3-4b-it")
        self.assertEqual(models.research.model_name, "qwen/qwen-2.5-7b-instruct")
        self.assertEqual(models.writer.model_name, "openai/gpt-4o-mini")
        self.assertEqual(models.guardrails.model_name, "openai/gpt-4o-mini")
        self.assertEqual(models.supervisor.max_tokens, 300)
        self.assertEqual(models.research.max_tokens, 600)
        self.assertEqual(models.guardrails.max_tokens, 1000)
        self.assertEqual(models.research.request_timeout, 60)

    def test_missing_router_key_does_not_fall_back_to_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai", "OPENROUTER_API_KEY": "test-router"}, clear=True):
            with self.assertRaisesRegex(ValueError, "ROUTERAI_API_KEY"):
                create_node_model("supervisor")

    def test_routerai_endpoint_and_auth_with_strict_schema(self):
        captured = []

        def handle(request):
            captured.append(request)
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": "google/gemma-3-4b-it", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps({"route": "research", "message": ""})}}],
            })

        with httpx.Client(transport=httpx.MockTransport(handle)) as http:
            def construct(**kwargs):
                return ChatOpenAI(**kwargs, http_client=http)

            env = {"ROUTERAI_API_KEY": "test-routerai", "OPENAI_API_KEY": "wrong-openai",
                   "OPENAI_BASE_URL": "https://wrong.example/v1", "OPENROUTER_API_KEY": "wrong-router"}
            with patch.dict(os.environ, env, clear=True), patch("langchain_openai.ChatOpenAI", side_effect=construct):
                model = create_node_model("supervisor")
                result = model.with_structured_output(ScopeDecision, method="json_schema", strict=True).invoke("Workplace question")
        self.assertEqual(result.route, "research")
        self.assertEqual(str(captured[0].url), "https://routerai.ru/api/v1/chat/completions")
        self.assertEqual(captured[0].headers["authorization"], "Bearer test-routerai")
        payload = json.loads(captured[0].content)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertNotIn("provider", payload)

    def test_explicit_per_role_configuration(self):
        env = {"OPENROUTER_API_KEY": "test-router", "WRITER_PROVIDER": "openrouter",
               "WRITER_MODEL": "openai/gpt-4o-mini", "WRITER_MAX_TOKENS": "1900"}
        with patch.dict(os.environ, env, clear=True):
            model = create_node_model("writer")
        self.assertEqual(model.model_name, "openai/gpt-4o-mini")
        self.assertEqual(model.max_tokens, 1900)

    def test_invalid_configuration(self):
        for env in ({"SUPERVISOR_PROVIDER": "unknown"}, {"SUPERVISOR_MODEL": ""},
                    {"SUPERVISOR_MAX_TOKENS": "0"}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                create_node_model("supervisor")

    def test_distinct_models_are_called_only_for_their_roles(self):
        supervisor, research, writer, guardrails = [ScriptedModel() for _ in range(4)]
        graph = build_graph(NodeModels(supervisor, research, writer, guardrails), lambda _: [HIT])
        self.assertEqual(ask(graph, "Как оформить отпуск?").status, "ok")
        for model, schema in ((supervisor, "ScopeDecision"), (research, "SearchPlan"),
                              (writer, "AnswerDraft"), (guardrails, "GroundingReview")):
            self.assertEqual([name for name, _ in model.calls], [schema])

    def test_openrouter_sdk_sends_strict_schema_and_provider_constraint(self):
        captured = []

        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "gen-test", "object": "chat.completion", "created": 1,
                "model": "google/gemma-3-4b-it", "provider": "test", "system_fingerprint": None,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content":
                                          json.dumps({"route": "research", "message": ""})}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        with httpx.Client(transport=httpx.MockTransport(handle)) as http:
            sdk = OpenRouter(api_key="test-router", client=http)
            model = ChatOpenRouter(model="google/gemma-3-4b-it", api_key="test-router", client=sdk,
                                   openrouter_provider={"require_parameters": True}, max_tokens=300)
            response = model.with_structured_output(ScopeDecision, method="json_schema", strict=True).invoke("A workplace question")
        self.assertEqual(response.route, "research")
        self.assertTrue(captured[0]["response_format"]["json_schema"]["strict"])
        self.assertTrue(captured[0]["provider"]["require_parameters"])
        self.assertEqual(captured[0]["max_tokens"], 300)

    def test_ollama_wrapper_still_removes_only_strict(self):
        calls = []

        class LocalModel:
            def with_structured_output(self, schema, **kwargs):
                calls.append(kwargs)
                return "bound"

        self.assertEqual(OllamaStructuredModel(LocalModel()).with_structured_output(
            ScopeDecision, method="json_schema", strict=True), "bound")
        self.assertEqual(calls, [{"method": "json_schema"}])


if __name__ == "__main__":
    unittest.main()
