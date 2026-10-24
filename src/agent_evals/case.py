"""Golden case model and JSON loader for deterministic agent evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation recorded during the agent turn."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: str = ""


@dataclass(frozen=True)
class Turn:
    """One prompt/answer cycle of the agent, with every recorded tool call."""

    prompt: str = ""
    response: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class Case:
    """A golden case: the scenario, the expected behavior, the limits."""

    id: str
    prompt: str = ""
    turns: list[Turn] = field(default_factory=list)
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_tool_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_numbers: tuple[str, ...] = ()
    expected_facts: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expected_language: str | None = None


def case_from_dict(entry: dict[str, Any]) -> Case:
    """Build a Case from a JSON dict; every expectation except the id is optional."""
    prompt = str(entry.get("prompt", ""))
    turns = []
    for raw in entry.get("turns", []):
        calls = [
            ToolCall(
                name=str(call.get("name", "")),
                args=dict(call.get("args", {})),
                result=str(call.get("result", "")),
            )
            for call in raw.get("tool_calls", [])
        ]
        turns.append(
            Turn(
                prompt=str(raw.get("prompt", prompt)),
                response=str(raw.get("response", "")),
                tool_calls=calls,
            )
        )
    return Case(
        id=str(entry["id"]),
        prompt=prompt,
        turns=turns,
        expected_tools=tuple(entry.get("expected_tools", ())),
        forbidden_tools=tuple(entry.get("forbidden_tools", ())),
        expected_tool_args=dict(entry.get("expected_tool_args", {})),
        expected_numbers=tuple(str(n) for n in entry.get("expected_numbers", ())),
        expected_facts=tuple(entry.get("expected_facts", ())),
        forbidden_phrases=tuple(entry.get("forbidden_phrases", ())),
        expected_language=entry.get("expected_language"),
    )


def load_cases(source: str | Path | TextIO) -> list[Case]:
    """Load a list of golden cases from a JSON file (a JSON array of case objects)."""
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        data = json.load(source)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of case objects")
    return [case_from_dict(entry) for entry in data]
