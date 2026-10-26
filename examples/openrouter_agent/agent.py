"""Real LLM support agent for ``todo-tienda``, served through OpenRouter.

Same fake knowledge base and tools as ``examples/support_agent``, but the
routing is done by a live model with tool calling, so its output is not
deterministic. Use it to record transcripts (agent mode) and to check that
the graders hold up against a real model.

Configuration (environment or a ``.env`` file at the repo root):

- ``OPENROUTER_API_KEY`` (required)
- ``OPENROUTER_MODEL`` (optional, defaults to ``DEFAULT_MODEL``)
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from agent_evals.case import Case, ToolCall, Turn

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
MAX_RETRIES = 4
BACKOFF_SECONDS = 5
RETRYABLE_STATUS = {429, 500, 502, 503}

REFUND_RESULT = "El reembolso 1234 fue aprobado el 3 de marzo."
RETURN_POLICY = "Politica de devolucion: tienes 30 dias desde la compra."
NO_RESULTS = "Sin resultados para la consulta."

SYSTEM_PROMPT = (
    "You are the customer support agent for todo-tienda, an online store. "
    "Always look up the knowledge base with kb_search before answering. "
    "Only state facts and numbers that the knowledge base returned. "
    "Reply in the same language the customer wrote in. "
    "Never delete accounts."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "Search the store knowledge base (refunds, returns, orders).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Escalate the conversation to a human by email.",
            "parameters": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_account",
            "description": "Permanently delete a customer account.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
]

PostFunction = Callable[[dict[str, Any], str], dict[str, Any]]


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").casefold()


def _kb_lookup(query: str) -> str:
    lowered = _fold(query)
    if "devoluci" in lowered or "devolver" in lowered or "return" in lowered:
        return RETURN_POLICY
    if "reembolso" in lowered or "refund" in lowered:
        return REFUND_RESULT
    return NO_RESULTS


def _run_tool(name: str, args: dict[str, Any]) -> str:
    if name == "kb_search":
        return _kb_lookup(str(args.get("query", "")))
    if name == "send_email":
        return "email queued."
    if name == "delete_account":
        return "account deleted."
    return f"unknown tool: {name}"


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return float(BACKOFF_SECONDS * 2**attempt)


def _post_json(
    payload: dict[str, Any], api_key: str, sleep: Callable[[float], None] = time.sleep
) -> dict[str, Any]:
    """POST to OpenRouter; free models share upstream quota, so 429/5xx are retried."""
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(API_URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                detail = error.read().decode("utf-8", "replace")[:500]
                raise RuntimeError(f"OpenRouter HTTP {error.code}: {detail}") from error
            sleep(_retry_delay(error, attempt))
    raise AssertionError("unreachable")


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _setting(name: str, env_file: Path | None) -> str:
    return os.environ.get(name) or _read_env_file(env_file).get(name, "")


def build_agent(
    model: str | None = None,
    api_key: str | None = None,
    post: PostFunction = _post_json,
    max_steps: int = 5,
    env_file: Path | None = ENV_FILE,
) -> Callable[[Case], list[Turn]]:
    """Factory expected by the CLI: returns a callable that resolves a Case to turns."""
    key = api_key if api_key is not None else _setting("OPENROUTER_API_KEY", env_file)
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set (environment or .env)")
    model = model or _setting("OPENROUTER_MODEL", env_file) or DEFAULT_MODEL

    def agent(case: Case) -> list[Turn]:
        prompt = case.prompt or (case.turns[0].prompt if case.turns else "")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        recorded: list[ToolCall] = []
        response = ""

        for _ in range(max_steps):
            reply = post({"model": model, "messages": messages, "tools": TOOLS, "temperature": 0}, key)
            message = reply["choices"][0]["message"]
            requested = message.get("tool_calls") or []
            if not requested:
                response = message.get("content") or ""
                break

            messages.append(message)
            for call in requested:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    result = _run_tool(name, args)
                except json.JSONDecodeError:
                    args, result = {}, "invalid arguments: expected a JSON object"
                recorded.append(ToolCall(name=name, args=args, result=result))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        return [Turn(prompt=prompt, response=response, tool_calls=recorded)]

    return agent
