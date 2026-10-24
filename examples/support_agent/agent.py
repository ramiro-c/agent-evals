"""Synthetic support agent for ``todo-tienda`` (a fictional store).

Deterministic by construction: pure rules over the prompt, one fake knowledge
base, no network, no model. Its branches are intentionally imperfect so the
golden suite can show each failure mode the graders catch:

- invented-number      -> numeric_grounding fails (stated number has no source)
- unsupported-claim    -> fact_grounding fails (claim not backed by the KB)
- wrong-tool           -> tool_usage fails (expected tool not called)
- forbidden-phrase     -> no_forbidden_content fails (deflection boilerplate)
- wrong-language       -> language fails (Spanish answer to an English prompt)
- forbidden-tool       -> tool_usage fails (account deletion is never allowed)
- wrong-args           -> tool_args fails (the query misses the expected term)
"""

from __future__ import annotations

import unicodedata
from typing import Callable

from agent_evals.case import Case, ToolCall, Turn

REFUND_RESULT = "El reembolso 1234 fue aprobado el 3 de marzo."
RETURN_POLICY = "Politica de devolucion: tienes 30 dias desde la compra."
NO_RESULTS = "Sin resultados para la consulta."


def _fold(text: str) -> str:
    """lowercase + accent-stripped, so 'borrá' matches branch keyword 'borra'."""
    decomposed = unicodedata.normalize("NFD", text)
    folded = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return folded.casefold()


def _kb_lookup(query: str) -> str:
    lowered = _fold(query)
    if "devoluci" in lowered or "devolver" in lowered:
        return RETURN_POLICY
    if "reembolso" in lowered or "refund" in lowered:
        return REFUND_RESULT
    return NO_RESULTS


def _kb_call(query: str) -> ToolCall:
    return ToolCall(name="kb_search", args={"query": query}, result=_kb_lookup(query))


def handle(prompt: str) -> Turn:
    """Deterministic routing over the prompt: one branch per behavior."""
    lowered = _fold(prompt)

    if "humano" in lowered or "persona real" in lowered:
        # Defect: abandons the expected kb_search for a human escalation email.
        return Turn(
            prompt=prompt,
            response="Derivado tu caso a un agente humano.",
            tool_calls=[
                ToolCall(name="send_email", args={"to": "soporte@todo-tienda.test"}, result="email queued."),
            ],
        )

    if "borrar" in lowered or "borra" in lowered or "eliminar" in lowered:
        # Defect: performs a destructive action that must never be allowed.
        return Turn(
            prompt=prompt,
            response="Listo, tu cuenta fue eliminada.",
            tool_calls=[ToolCall(name="delete_account", args={"user_id": "u-777"}, result="account deleted.")],
        )

    if "factura" in lowered:
        # Defect: deflects with boilerplate instead of using the knowledge base.
        return Turn(
            prompt=prompt,
            response="Como IA no puedo ver tus facturas ni resolver consultas compuestas.",
            tool_calls=[ToolCall(name="kb_search", args={"query": prompt}, result=NO_RESULTS)],
        )

    if "cancelar" in lowered or "contracargo" in lowered:
        # Defect: invents a refund ticket number the KB never gave.
        return Turn(
            prompt=prompt,
            response="Tu reembolso 9999 fue aprobado y demora siete dias habiles.",
            tool_calls=[_kb_call(prompt)],
        )

    if "devolver" in lowered or "devolucion" in lowered:
        # Defect: contradicts the policy on record without citing a source.
        return Turn(
            prompt=prompt,
            response="Podes devolver tu compra hasta mucho despues de lo que dice la politica actual.",
            tool_calls=[_kb_call(prompt)],
        )

    if "refund" in lowered or "where is my" in lowered:
        # Defect: answers an English prompt in Spanish.
        return Turn(
            prompt=prompt,
            response="Tu reembolso 1234 fue aprobado el dia de hoy.",
            tool_calls=[_kb_call(prompt)],
        )

    if "acredita" in lowered:
        # Defect: the KB call misses the wording the case expects.
        return Turn(
            prompt=prompt,
            response="Tu reembolso 1234 se acredita el 3 de marzo.",
            tool_calls=[_kb_call("reembolso total")],
        )

    # Clean path: look up the KB and answer with its facts, in one turn.
    call = _kb_call(prompt)
    return Turn(
        prompt=prompt,
        response="Tu reembolso 1234 fue aprobado y llega como minimo el 3 de marzo. Gracias por tu paciencia.",
        tool_calls=[call],
    )


def build_agent() -> Callable[[Case], list[Turn]]:
    """Factory expected by the CLI: returns a callable that resolves a Case to turns."""

    def agent(case: Case) -> list[Turn]:
        return [handle(case.prompt or (case.turns[0].prompt if case.turns else ""))]

    return agent


if __name__ == "__main__":
    print(handle("¿Cuál es el estado de mi reembolso?"))
