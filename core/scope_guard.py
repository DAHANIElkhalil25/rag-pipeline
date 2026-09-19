"""Contrôle déterministe du périmètre avant et après la recherche RAG.

Architecture à double filtre :

  Filtre 1 — Pré-recherche (explicit_scope_refusal)
      Détecte les questions explicitement hors domaine AVANT d'appeler FAISS,
      BM25 ou le reranker. Économise le GPU et le temps de calcul.
      → statut : "refused"

  Filtre 2 — Post-reranking (evaluate_scope)
      Après la recherche hybride + BGE, vérifie que le meilleur passage a un
      score suffisant pour justifier une génération.
      Le score utilisé est `reranker_score` (logit BGE) si le reranking est
      activé, sinon `retrieval_score` (score hybride normalisé [0-1]).
      Un logit BGE < 0.0  ↔  probabilité de pertinence < 50 %.
      Un score hybride < 0.30 ↔  peu de chevauchement sémantique ET lexical.
      → statut : "abstained"

  Résultat normal :
      → statut : "answered"

Structure du payload de retour
-------------------------------
{
    "status"   : "answered" | "refused" | "abstained",
    "reason"   : str,          # code interne (pour les logs)
    "message"  : str | None,   # message à afficher à l'utilisateur
    "confidence": float | None # score du top-1 passage (pour debug)
}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping


# ---------------------------------------------------------------------------
# Messages utilisateur
# ---------------------------------------------------------------------------
_MSG_REFUSED = (
    "Cette question ne concerne pas les domaines couverts par cet assistant. "
    "Le système traite principalement Python, Scikit-learn, LangChain et les "
    "systèmes RAG (FAISS, BM25, embeddings, reranking)."
)

_MSG_ABSTAINED = (
    "Je n'ai pas trouvé suffisamment d'informations pertinentes dans la "
    "documentation disponible pour répondre de manière fiable à cette question."
)

# Message de refus avec mention de la bibliothèque détectée
_MSG_REFUSED_LIBRARY = (
    "Cette question porte sur {library}, qui n'est pas inclus dans le corpus "
    "de cet assistant. Le système couvre Python, Scikit-learn, LangChain et "
    "les systèmes RAG."
)

# Gardé pour compatibilité avec l'ancienne interface
DEFAULT_REFUSAL_MESSAGE = _MSG_ABSTAINED


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Casse et espaces normalisés, pour la comparaison interne."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


# ---------------------------------------------------------------------------
# Filtre 1 — Pré-recherche : détection explicite hors périmètre
# ---------------------------------------------------------------------------

def _detect_out_of_scope(question: str, config: Mapping[str, Any]) -> str | None:
    """Retourne le nom de la bibliothèque/sujet détecté si la question est
    explicitement hors périmètre, sinon None.

    Logique :
      1. Alias pointés  :  pd.xxx, np.xxx, tf.xxx  → pandas, numpy, tensorflow
         (usage API réel, fort signal hors périmètre)
      2. Mots-clés entiers : pandas, django, flask …
         (uniquement en mot entier pour éviter les faux positifs)

    Limites connues :
      - Ne détecte pas les contournements ("librairie de tableaux python").
      - Ne détecte pas les questions de comparaison ("pandas vs polars").
      Ces cas passeront au filtre 2 (score post-reranking).
    """
    normalised = _normalise(question)

    # Alias pointés (ex : pd.head, np.array, tf.keras)
    for alias, library in config.get("unsupported_aliases", {}).items():
        pattern = rf"\b{re.escape(str(alias).casefold())}\s*\."
        if re.search(pattern, normalised):
            return str(library)

    # Mots-clés entiers
    for library in config.get("unsupported_libraries", []):
        pattern = rf"\b{re.escape(str(library).casefold())}\b"
        if re.search(pattern, normalised):
            return str(library)

    return None


def explicit_scope_refusal(
    question: str,
    config: Mapping[str, Any],
) -> Dict[str, Any] | None:
    """Retourne un payload 'refused' si un sujet hors périmètre est détecté,
    sinon None.  Appelé AVANT la recherche FAISS/BM25.
    """
    if not question or not question.strip():
        # Question vide → laisser le filtre 2 gérer (pas de contexte)
        return None

    topic = _detect_out_of_scope(question, config)
    if topic is None:
        return None

    return {
        "status": "refused",
        "reason": "unsupported_library",
        "confidence": None,
        "detected_topic": topic,
        "message": _MSG_REFUSED_LIBRARY.format(library=topic),
        # Compatibilité ancienne interface
        "allow_answer": False,
    }


# ---------------------------------------------------------------------------
# Filtre 2 — Post-reranking : pertinence insuffisante
# ---------------------------------------------------------------------------

def _top_score(contexts: List[Dict[str, Any]]) -> tuple[float | None, str]:
    """Extrait le score le plus fiable du passage top-1 et indique son type.

    Priorité (du plus fiable au moins fiable) :
      1. reranker_score  → logit BGE absolu    (seuil conseillé : 0.0)
      2. raw_cosine_score → cosinus FAISS brut  (seuil conseillé : 0.45)
      3. retrieval_score → hybride normalisé    (NE PAS utiliser comme seuil absolu)

    Retourne (score, type_score).
    """
    if not contexts:
        return None, "none"
    top = contexts[0]

    if top.get("reranker_score") is not None:
        return float(top["reranker_score"]), "reranker"

    if top.get("raw_cosine_score") is not None:
        return float(top["raw_cosine_score"]), "cosine"

    # Fallback : score hybride normalisé (non fiable pour seuillage absolu)
    score = top.get("retrieval_score")
    return (float(score), "hybrid_normalized") if score is not None else (None, "none")


def evaluate_scope(
    question: str,
    contexts: List[Dict[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Décision finale après la recherche + reranking.

    Seuils selon le type de score disponible :
      - reranker (logit BGE)   : config["minimum_top_score"] (défaut 0.0)
      - cosine brut FAISS      : config["minimum_cosine_score"] (défaut 0.45)
      - hybride normalisé      : pas de seuil (non fiable en valeur absolue)

    Retourne un payload structuré : status, reason, message, confidence, allow_answer.
    """
    # Garde désactivée → réponse libre
    if not config.get("enabled", True):
        return {
            "status": "answered",
            "reason": "guard_disabled",
            "confidence": None,
            "score_type": None,
            "message": None,
            "allow_answer": True,
        }

    # Filtre 1 rejoué pour la cohérence (au cas où appelé directement)
    explicit = explicit_scope_refusal(question, config)
    if explicit:
        return explicit

    # Aucun contexte récupéré
    if not contexts:
        return {
            "status": "abstained",
            "reason": "no_retrieved_context",
            "confidence": 0.0,
            "score_type": "none",
            "message": _MSG_ABSTAINED,
            "allow_answer": False,
        }

    # Extraction du score le plus fiable
    confidence, score_type = _top_score(contexts)

    # Seuil adapté au type de score
    threshold = None
    if score_type == "reranker":
        threshold = config.get("minimum_top_score", 0.0)
    elif score_type == "cosine":
        threshold = config.get("minimum_cosine_score", 0.45)
    # hybrid_normalized : pas de seuil absolu fiable → on laisse passer

    if (
        threshold is not None
        and confidence is not None
        and confidence < float(threshold)
    ):
        return {
            "status": "abstained",
            "reason": "retrieval_score_below_threshold",
            "confidence": confidence,
            "score_type": score_type,
            "message": _MSG_ABSTAINED,
            "allow_answer": False,
        }

    # Tout est bon → génération autorisée
    return {
        "status": "answered",
        "reason": "sufficient_retrieved_context",
        "confidence": confidence,
        "score_type": score_type,
        "message": None,
        "allow_answer": True,
    }

