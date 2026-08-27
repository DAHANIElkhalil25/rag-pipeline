"""Contrôle déterministe du périmètre avant génération RAG.

Cette couche ne remplace pas la récupération : elle évite simplement d'appeler
le LLM lorsqu'une bibliothèque explicitement hors corpus est détectée ou
lorsqu'aucun passage n'a été retrouvé. Le seuil numérique reste optionnel afin
de ne pas rejeter des questions françaises valides sur une documentation anglaise
avant une calibration expérimentale dédiée.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping


DEFAULT_REFUSAL_MESSAGE = (
    "Je ne dispose pas d’une source suffisamment fiable dans mon corpus pour répondre à cette question. "
    "Le corpus couvre Python, Scikit-learn et LangChain, mais pas ce sujet."
)


def _normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.casefold()).strip()


def _detect_explicit_out_of_scope_topic(question: str, config: Mapping[str, Any]) -> str | None:
    """Return the named unsupported topic explicitly present in the question."""
    normalised = _normalise_question(question)
    for alias, library in config.get("unsupported_aliases", {}).items():
        if re.search(rf"\b{re.escape(str(alias).casefold())}\s*\.", normalised):
            return str(library)

    for library in config.get("unsupported_libraries", []):
        if re.search(rf"\b{re.escape(str(library).casefold())}\b", normalised):
            return str(library)
    return ""


def explicit_scope_refusal(question: str, config: Mapping[str, Any]) -> Dict[str, Any] | None:
    """Return a refusal before retrieval when a known unsupported library is named."""
    unsupported_topic = _detect_explicit_out_of_scope_topic(question, config)
    if not unsupported_topic:
        return None
    return {
        "allow_answer": False,
        "reason": "unsupported_library",
        "confidence": None,
        "detected_topic": unsupported_topic,
        "message": (
            f"Je ne peux pas répondre de façon fiable à propos de {unsupported_topic}, "
            "car cette bibliothèque n’est pas incluse dans le corpus actuel. "
            "Le corpus couvre Python, Scikit-learn et LangChain."
        ),
    }


def evaluate_scope(
    question: str,
    contexts: List[Dict[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Decide whether evidence is sufficient to call the answer generator.

    The return payload is deliberately structured for the notebook, the Gradio
    UI, and later audit logs. It is deterministic and performs no model call.
    """
    if not config.get("enabled", True):
        return {"allow_answer": True, "reason": "guard_disabled", "confidence": None}

    explicit_refusal = explicit_scope_refusal(question, config)
    if explicit_refusal:
        return explicit_refusal

    if not contexts:
        return {
            "allow_answer": False,
            "reason": "no_retrieved_context",
            "confidence": 0.0,
            "message": DEFAULT_REFUSAL_MESSAGE,
        }

    top_context = contexts[0]
    confidence = top_context.get("reranker_score", top_context.get("retrieval_score"))
    minimum_score = config.get("minimum_top_score")
    if minimum_score is not None and confidence is not None and float(confidence) < float(minimum_score):
        return {
            "allow_answer": False,
            "reason": "retrieval_score_below_threshold",
            "confidence": float(confidence),
            "message": DEFAULT_REFUSAL_MESSAGE,
        }

    return {
        "allow_answer": True,
        "reason": "sufficient_retrieved_context",
        "confidence": float(confidence) if confidence is not None else None,
    }
