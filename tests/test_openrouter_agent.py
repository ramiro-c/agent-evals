"""OpenRouter agent: tool loop and transcript shape, with the HTTP layer faked."""

import importlib.util
import io
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from agent_evals.case import Case

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "examples" / "openrouter_agent" / "agent.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("openrouter_agent", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_call_reply(name, arguments):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}
                    ],
                }
            }
        ]
    }


def _text_reply(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class FakePost:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, payload, api_key):
        self.payloads.append(payload)
        return self.replies.pop(0)


class OpenRouterAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_records_tool_calls_with_kb_results_and_final_answer(self):
        post = FakePost(
            [
                _tool_call_reply("kb_search", '{"query": "estado reembolso"}'),
                _text_reply("Tu reembolso 1234 fue aprobado el 3 de marzo."),
            ]
        )
        agent = self.module.build_agent(post=post, api_key="test-key")

        turns = agent(Case(id="c1", prompt="¿Cuál es el estado de mi reembolso?"))

        self.assertEqual(len(turns), 1)
        turn = turns[0]
        self.assertEqual(turn.prompt, "¿Cuál es el estado de mi reembolso?")
        self.assertEqual(turn.response, "Tu reembolso 1234 fue aprobado el 3 de marzo.")
        self.assertEqual(len(turn.tool_calls), 1)
        call = turn.tool_calls[0]
        self.assertEqual(call.name, "kb_search")
        self.assertEqual(call.args, {"query": "estado reembolso"})
        self.assertEqual(call.result, "El reembolso 1234 fue aprobado el 3 de marzo.")
        # The tool result is fed back to the model on the second request.
        tool_messages = [m for m in post.payloads[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(tool_messages[0]["content"], call.result)

    def test_malformed_tool_arguments_are_recorded_not_raised(self):
        post = FakePost([_tool_call_reply("kb_search", "{not json"), _text_reply("No encontre nada.")])
        agent = self.module.build_agent(post=post, api_key="test-key")

        turn = agent(Case(id="c2", prompt="hola"))[0]

        self.assertEqual(turn.tool_calls[0].args, {})
        self.assertIn("invalid arguments", turn.tool_calls[0].result)

    def test_stops_after_max_steps(self):
        post = FakePost([_tool_call_reply("kb_search", '{"query": "x"}')] * 3)
        agent = self.module.build_agent(post=post, api_key="test-key", max_steps=3)

        turn = agent(Case(id="c3", prompt="loop"))[0]

        self.assertEqual(len(turn.tool_calls), 3)
        self.assertEqual(turn.response, "")

    def test_post_retries_on_rate_limit_then_succeeds(self):
        rate_limited = urllib.error.HTTPError(
            self.module.API_URL, 429, "Too Many Requests", {"Retry-After": "2"}, io.BytesIO(b"{}")
        )
        ok = io.BytesIO(b'{"choices": []}')
        sleeps = []
        with mock.patch.object(self.module.urllib.request, "urlopen", side_effect=[rate_limited, ok]):
            reply = self.module._post_json({"model": "m"}, "k", sleep=sleeps.append)

        self.assertEqual(reply, {"choices": []})
        self.assertEqual(sleeps, [2.0])

    def test_post_gives_up_with_provider_message_after_retries(self):
        def rate_limited(*_args, **_kwargs):
            raise urllib.error.HTTPError(
                self.module.API_URL, 429, "Too Many Requests", {}, io.BytesIO(b'{"error": "rate-limited upstream"}')
            )

        sleeps = []
        with mock.patch.object(self.module.urllib.request, "urlopen", side_effect=rate_limited):
            with self.assertRaisesRegex(RuntimeError, "429.*rate-limited upstream"):
                self.module._post_json({"model": "m"}, "k", sleep=sleeps.append)

        self.assertEqual(len(sleeps), self.module.MAX_RETRIES)

    def test_missing_api_key_fails_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            self.module.build_agent(post=FakePost([]), api_key="", env_file=None)


if __name__ == "__main__":
    unittest.main()
