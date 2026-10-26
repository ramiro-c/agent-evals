"""Grader-level contract tests: one positive and one negative case per grader."""

import unittest

from agent_evals.case import Case, ToolCall, Turn
from agent_evals.graders import (
    fact_grounding,
    language,
    no_forbidden_content,
    numeric_grounding,
    tool_args,
    tool_usage,
)


def _case(**overrides):
    base = dict(id="t", prompt="¿cuál es el estado de mi reembolso?", turns=[])
    base.update(overrides)
    return Case(**base)


class ToolUsageTests(unittest.TestCase):
    def test_passes_when_expected_tool_called(self):
        case = _case(expected_tools=("kb_search",))
        call = ToolCall(name="kb_search", args={"query": "reembolso"}, result="refund 1234 approved")
        result = tool_usage(case, [Turn(prompt=case.prompt, response="ok", tool_calls=[call])])
        self.assertTrue(result.passed)
        self.assertEqual(result.check, "tool_usage")

    def test_fails_when_expected_tool_missing(self):
        case = _case(expected_tools=("kb_search",))
        result = tool_usage(case, [Turn(prompt=case.prompt, response="ok", tool_calls=[])])
        self.assertFalse(result.passed)
        self.assertIn("kb_search", result.detail)

    def test_fails_when_forbidden_tool_called(self):
        case = _case(forbidden_tools=("send_email",))
        bad = ToolCall(name="send_email", args={}, result="sent")
        result = tool_usage(case, [Turn(prompt=case.prompt, response="ok", tool_calls=[bad])])
        self.assertFalse(result.passed)
        self.assertIn("send_email", result.detail)

    def test_passes_without_any_expectation(self):
        result = tool_usage(_case(), [Turn(prompt="p", response="r", tool_calls=[])])
        self.assertTrue(result.passed)


class ToolArgsTests(unittest.TestCase):
    def test_passes_on_superset_arguments(self):
        case = _case(expected_tool_args={"kb_search": {"query": "reembolso"}})
        call = ToolCall(name="kb_search", args={"query": "reembolso", "limit": 3}, result="x")
        result = tool_args(case, [Turn(prompt="p", response="r", tool_calls=[call])])
        self.assertTrue(result.passed)

    def test_fails_on_wrong_argument(self):
        case = _case(expected_tool_args={"kb_search": {"query": "reembolso"}})
        call = ToolCall(name="kb_search", args={"query": "factura"}, result="x")
        result = tool_args(case, [Turn(prompt="p", response="r", tool_calls=[call])])
        self.assertFalse(result.passed)
        self.assertIn("query", result.detail)
        self.assertIn("factura", result.detail)

    def test_contains_matcher_accepts_rephrased_query(self):
        case = _case(expected_tool_args={"kb_search": {"query": {"contains": ["Reembolso", "estado"]}}})
        call = ToolCall(name="kb_search", args={"query": "estado del REEMBOLSO"}, result="x")
        result = tool_args(case, [Turn(prompt="p", response="r", tool_calls=[call])])
        self.assertTrue(result.passed, result.detail)

    def test_contains_matcher_is_accent_insensitive(self):
        case = _case(expected_tool_args={"kb_search": {"query": {"contains": ["devolucion"]}}})
        call = ToolCall(name="kb_search", args={"query": "política de devolución"}, result="x")
        result = tool_args(case, [Turn(prompt="p", response="r", tool_calls=[call])])
        self.assertTrue(result.passed, result.detail)

    def test_contains_matcher_fails_when_a_term_is_missing(self):
        case = _case(expected_tool_args={"kb_search": {"query": {"contains": ["reembolso", "estado"]}}})
        call = ToolCall(name="kb_search", args={"query": "reembolso"}, result="x")
        result = tool_args(case, [Turn(prompt="p", response="r", tool_calls=[call])])
        self.assertFalse(result.passed)

    def test_number_and_string_argument_values_match(self):
        case = _case(expected_tool_args={"calc": {"expression": 4242}})
        call = ToolCall(name="calc", args={"expression": 4242}, result="x")
        self.assertTrue(tool_args(case, [Turn(prompt="p", response="r", tool_calls=[call])]).passed)
        case2 = _case(expected_tool_args={"kb_search": {"query": "reembolso"}})
        call2 = ToolCall(name="kb_search", args={"query": "reembolso"}, result="x")
        self.assertTrue(tool_args(case2, [Turn(prompt="p", response="r", tool_calls=[call2])]).passed)


class NumericGroundingTests(unittest.TestCase):
    def _turns(self, response):
        call = ToolCall(name="kb_search", args={}, result="reembolso 1234: aprobado. Total USD 42.50")
        return [Turn(prompt="p", response=response, tool_calls=[call])]

    def test_passes_when_number_is_supported(self):
        case = _case(expected_numbers=("1234", "42.50"))
        result = numeric_grounding(case, self._turns("Tu reembolso 1234 fue aprobado por USD 42.50."))
        self.assertTrue(result.passed, result.detail)

    def test_fails_on_invented_number(self):
        case = _case(expected_numbers=("1234",))
        result = numeric_grounding(case, self._turns("Tu reembolso 9999 fue aprobado."))
        self.assertFalse(result.passed)
        self.assertIn("9999", result.detail)

    def test_normalizes_thousand_separators_and_decimal_comma(self):
        case = _case()
        result = numeric_grounding(case, self._turns("Acumulado 1,234 en tu cuenta."))
        self.assertTrue(result.passed, result.detail)
        case2 = _case()
        call = ToolCall(name="x", args={}, result="total 1234567,89")
        result2 = numeric_grounding(case2, [Turn(prompt="p", response="total 1234567,89", tool_calls=[call])])
        self.assertTrue(result2.passed, result2.detail)

    def test_expected_number_must_appear_in_response(self):
        case = _case(expected_numbers=("5555",))
        result = numeric_grounding(case, self._turns("Tu reembolso 1234 fue aprobado."))
        self.assertFalse(result.passed)
        self.assertIn("5555", result.detail)

    def test_ignores_numbers_glued_to_letters_or_hyphens(self):
        case = _case()
        call = ToolCall(name="kb_search", args={}, result="reembolso 1234: aprobado")
        turns = [Turn(prompt="p", response="Modelo GPT-4 y h2o detectados, sin cifras sueltas.", tool_calls=[call])]
        result = numeric_grounding(case, turns)
        self.assertTrue(result.passed, result.detail)


class FactGroundingTests(unittest.TestCase):
    def _turns(self, response):
        call = ToolCall(
            name="kb_search", args={}, result="Politica: el reembolso 1234 fue aprobado el 3 de marzo."
        )
        return [Turn(prompt="p", response=response, tool_calls=[call])]

    def test_passes_when_fact_is_cited_from_source(self):
        case = _case(expected_facts=("Reembolso 1234 fue aprobado",))
        result = fact_grounding(case, self._turns("El Reembolso 1234 fue aprobado hoy."))
        self.assertTrue(result.passed, result.detail)

    def test_fails_when_claim_has_no_source(self):
        case = _case(expected_facts=("Reembolso 1234 fue aprobado",))
        result = fact_grounding(case, self._turns("Tu reembolso fue aprobado ayer a las 8."))
        self.assertFalse(result.passed)

    def test_fails_when_fact_in_response_but_never_in_tool_results(self):
        case = _case(expected_facts=("reembolso fue rechazado",))
        result = fact_grounding(case, self._turns("El reembolso fue rechazado."))
        self.assertFalse(result.passed)


class NoForbiddenContentTests(unittest.TestCase):
    def test_passes_when_frase_absente(self):
        case = _case(forbidden_phrases=("como IA no puedo",))
        result = no_forbidden_content(case, [Turn(prompt="p", response="Tu reembolso fue aprobado.", tool_calls=[])])
        self.assertTrue(result.passed)

    def test_fails_case_insensitive_accent_proof(self):
        case = _case(forbidden_phrases=("como IA no puedo",))
        result = no_forbidden_content(case, [Turn(prompt="p", response="COMO iA, NO Puedo ver tus facturas.", tool_calls=[])])
        self.assertFalse(result.passed)


class LanguageTests(unittest.TestCase):
    def test_detects_spanish(self):
        case = _case(expected_language="es")
        turn = Turn(prompt="p", response="Tu reembolso fue aprobado el dia de hoy. Gracias por tu paciencia.", tool_calls=[])
        self.assertTrue(language(case, [turn]).passed, language(case, [turn]).detail)

    def test_fails_when_language_differs(self):
        case = _case(expected_language="es")
        turn = Turn(prompt="p", response="Your refund was approved today. Thank you for your patience while we process the payment.", tool_calls=[])
        self.assertFalse(language(case, [turn]).passed)

    def test_ambiguous_tie_fails_with_reason(self):
        case = _case(expected_language="es")
        turn = Turn(prompt="p", response="OK ok", tool_calls=[])
        result = language(case, [turn])
        self.assertFalse(result.passed)
        self.assertIn("ambig", result.detail.lower())

    def test_no_language_expected_skips(self):
        case = _case()
        turn = Turn(prompt="p", response="ok", tool_calls=[])
        self.assertTrue(language(case, [turn]).passed)


if __name__ == "__main__":
    unittest.main()
