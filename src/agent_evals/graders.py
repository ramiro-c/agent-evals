"""Deterministic graders.

Every grader is a pure function `(case, turns) -> CheckResult`. No network,
no model in the middle: the same input always produces the same verdict.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from .case import Case, Turn

_NUM_RE = re.compile(r"(?<!\w)(?<![A-Za-z]-)[+-]?\d+(?:[.,]\d+)*(?!\w)")

_ES_STOPWORDS = frozenset(
    "a al de del la el le lo las los les un una unos unas y o u en con por para que se su sus tu tus es son esta "
    "este esto estos estas ese eso como mi me mas cuando donde si no ya fue hoy".split())
_EN_STOPWORDS = frozenset(
    "the a an and or of to in on for with at by is are was were be been from as this that these those it its "
    "your you we our us not do does did have has will can should would where when what which how thank thanks "
    "please here there all any more than then so no yes".split())


def normalize_text(text: str) -> str:
    """lowercase, strip accents, collapse whitespace."""
    decomposed = unicodedata.normalize("NFD", text)
    folded = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    folded = folded.casefold()
    return re.sub(r"\s+", " ", folded).strip()


def match_text(text: str) -> str:
    """normalize_text plus punctuation folding, for phrase matching:
    'COMO iA, NO Puedo' matches the forbidden phrase 'como IA no puedo'."""
    stripped = re.sub(r"[^\w\s]", " ", normalize_text(text))
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_number(token: str) -> str:
    """Canonical form of a number token: unify decimal separators, drop thousand separators.

    Rules (deterministic, documented in the README):
    - A token made of groups separated by a single kind of separator where every
      following group has exactly 3 digits (``1,234`` or ``1.234.567``) is thousand-separated.
    - Otherwise the LAST separator is the decimal point and every earlier one is thousands
      (``1,234.50`` -> ``1234.50``; ``42,5`` -> ``42.5``; ``1234567,89`` -> ``1234567.89``).
    """
    digits = token[1:].lstrip() if token and token[0] in "+-" else token
    negative = token[:1] == "-"
    if not any(separator in digits for separator in (".", ",")):
        canonical = digits
    elif re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", digits):
        canonical = re.sub(r"[.,]", "", digits)
    else:
        last = max(digits.rfind("."), digits.rfind(","))
        integer = re.sub(r"[.,]", "", digits[:last])
        decimals = digits[last + 1 :]
        canonical = f"{integer}.{decimals}" if decimals else integer
    return f"-{canonical}" if negative and canonical not in ("", "0") else canonical


def extract_numbers(text: str) -> list[str]:
    """Canonical number tokens inside ``text``; digits glued to letters are ignored."""
    return [normalize_number(match.group()) for match in _NUM_RE.finditer(text)]


def _response_text(turns: list[Turn]) -> str:
    return " ".join(turn.response for turn in turns)


def _tool_results_text(turns: list[Turn]) -> str:
    return " ".join(call.result for turn in turns for call in turn.tool_calls)


class CheckResult:
    """Outcome of one grader against one case."""

    __slots__ = ("check", "passed", "detail")

    def __init__(self, check: str, passed: bool, detail: str = "") -> None:
        self.check = check
        self.passed = passed
        self.detail = detail

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CheckResult(check={self.check!r}, passed={self.passed}, detail={self.detail!r})"


def _calls_in(turns: list[Turn]) -> list[str]:
    return [call.name for turn in turns for call in turn.tool_calls]


def tool_usage(case: Case, turns: list[Turn]) -> CheckResult:
    calls = _calls_in(turns)
    missing = [name for name in case.expected_tools if name not in calls]
    not_allowed = [name for name in case.forbidden_tools if name in calls]
    if not missing and not not_allowed:
        detail = (
            "no tool expectation configured"
            if not case.expected_tools and not case.forbidden_tools
            else "tool usage matches the case"
        )
        return CheckResult("tool_usage", True, detail)
    problems = []
    if missing:
        problems.append("tools not called: " + ", ".join(missing))
    if not_allowed:
        problems.append("forbidden tools called: " + ", ".join(not_allowed))
    return CheckResult("tool_usage", False, "; ".join(problems))


ToolCallArgPair = dict[str, Any]


def _args_match(actual: dict[str, Any], expected_subset: dict[str, Any]) -> bool:
    return all(key in actual and (actual[key] == value or str(actual[key]) == str(value)) for key, value in expected_subset.items())


def tool_args(case: Case, turns: list[Turn]) -> CheckResult:
    if not case.expected_tool_args:
        return CheckResult("tool_args", True, "no argument expectation configured")
    by_tool: dict[str, list[ToolCallArgPair]] = {}
    for turn in turns:
        for call in turn.tool_calls:
            by_tool.setdefault(call.name, []).append(call.args)
    for tool, expected_subset in case.expected_tool_args.items():
        candidates = by_tool.get(tool, [])
        if any(_args_match(actual, expected_subset) for actual in candidates):
            continue
        return CheckResult(
            "tool_args",
            False,
            f"{tool}: no call with args {dict(expected_subset)}",
        )
    return CheckResult("tool_args", True, "argument subsets match")


def numeric_grounding(case: Case, turns: list[Turn]) -> CheckResult:
    supported = set(extract_numbers(_tool_results_text(turns)))
    stated = extract_numbers(_response_text(turns))
    unsupported = [number for number in stated if number not in supported]
    not_stated = [n for n in case.expected_numbers if normalize_number(n) not in stated]
    if not unsupported and not not_stated:
        return CheckResult("numeric_grounding", True, "every stated number is supported")
    problems = []
    if unsupported:
        problems.append("numbers not supported by tool results: " + ", ".join(unsupported))
    if not_stated:
        problems.append("expected numbers absent from the response: " + ", ".join(not_stated))
    return CheckResult("numeric_grounding", False, "; ".join(problems))


def fact_grounding(case: Case, turns: list[Turn]) -> CheckResult:
    response = match_text(_response_text(turns))
    results = match_text(_tool_results_text(turns))
    for phrase in case.expected_facts:
        normalized = match_text(phrase)
        if normalized not in response:
            return CheckResult("fact_grounding", False, f"fact absent from the response: {phrase!r}")
        if normalized not in results:
            return CheckResult("fact_grounding", False, f"fact not backed by any tool result: {phrase!r}")
    return CheckResult("fact_grounding", True, "every expected fact is cited from the source")


def no_forbidden_content(case: Case, turns: list[Turn]) -> CheckResult:
    response = match_text(_response_text(turns))
    for phrase in case.forbidden_phrases:
        if match_text(phrase) in response:
            return CheckResult("no_forbidden_content", False, f"forbidden phrase present: {phrase!r}")
    return CheckResult("no_forbidden_content", True, "no forbidden phrase present")


def language(case: Case, turns: list[Turn]) -> CheckResult:
    if case.expected_language is None:
        return CheckResult("language", True, "language not asserted")
    tokens = re.findall(r"[a-z]+", normalize_text(_response_text(turns)))
    es_hits = sum(1 for token in tokens if token in _ES_STOPWORDS)
    en_hits = sum(1 for token in tokens if token in _EN_STOPWORDS)
    if es_hits == 0 and en_hits == 0:
        return CheckResult("language", False, "ambiguous: no language signal")
    if es_hits == en_hits:
        return CheckResult("language", False, f"ambiguous tie (es={es_hits}, en={en_hits})")
    detected = "es" if es_hits > en_hits else "en"
    if detected != case.expected_language.lower():
        detail = f"expected {case.expected_language!r}, detected {detected!r}"
        return CheckResult("language", False, detail)
    return CheckResult("language", True, f"detected {detected}")


ALL_GRADERS: tuple[Callable[[Case, list[Turn]], CheckResult], ...] = (
    tool_usage,
    tool_args,
    numeric_grounding,
    fact_grounding,
    no_forbidden_content,
    language,
)
GRADER_NAMES: tuple[str, ...] = tuple(grader.__name__ for grader in ALL_GRADERS)
