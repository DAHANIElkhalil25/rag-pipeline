"""Tests rapides et reproductibles du scope guard — sans évaluation Ragas.

Ces tests vérifient uniquement la logique de contrôle de périmètre,
sans charger le modèle Mistral ni l'index FAISS. Ils sont exécutables
en quelques secondes sur n'importe quelle machine.

Lancer avec :
    python -m pytest tests/test_scope_guard.py -v
ou directement :
    python tests/test_scope_guard.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.scope_guard import (
    explicit_scope_refusal,
    evaluate_scope,
    _normalise,
    _detect_out_of_scope,
)
from config import SCOPE_GUARD_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CFG = SCOPE_GUARD_CONFIG

# Simule un passage pertinent (logit BGE > 0)
CONTEXT_GOOD = [
    {"chunk_text": "Python est un langage interprété...", "reranker_score": 2.5,
     "retrieval_score": 0.85, "doc_title": "Python Tutorial", "doc_source": "python"}
]

# Simule un passage non pertinent (logit BGE < 0)
CONTEXT_BAD = [
    {"chunk_text": "Liste de noms aléatoires...", "reranker_score": -1.2,
     "retrieval_score": 0.15, "doc_title": "Doc inconnue", "doc_source": "python"}
]


# ---------------------------------------------------------------------------
# Catégorie 1 — Question clairement dans le périmètre
# ---------------------------------------------------------------------------

def test_in_scope_python():
    """Question valide sur Python → pas de refus."""
    q = "Comment fonctionne un décorateur Python ?"
    assert explicit_scope_refusal(q, CFG) is None, "Ne doit pas refuser une question Python."

def test_in_scope_sklearn():
    """Question valide sur Scikit-learn → pas de refus."""
    q = "Comment faire une validation croisée avec scikit-learn ?"
    assert explicit_scope_refusal(q, CFG) is None

def test_in_scope_langchain():
    """Question valide sur LangChain → pas de refus."""
    q = "Comment créer un retriever personnalisé dans LangChain ?"
    assert explicit_scope_refusal(q, CFG) is None

def test_in_scope_rag():
    """Question sur FAISS/RAG → dans le périmètre."""
    q = "Quelle est la différence entre FAISS et BM25 ?"
    assert explicit_scope_refusal(q, CFG) is None


# ---------------------------------------------------------------------------
# Catégorie 2 — Question clairement hors périmètre
# ---------------------------------------------------------------------------

def test_out_of_scope_pandas_alias():
    """Alias pd. → pandas détecté avant recherche."""
    q = "Comment utiliser pd.head() pour afficher un DataFrame ?"
    result = explicit_scope_refusal(q, CFG)
    assert result is not None, "Doit détecter pandas via alias pd."
    assert result["status"] == "refused"
    assert "pandas" in result["detected_topic"].lower()

def test_out_of_scope_django():
    """Mot-clé django → refus."""
    q = "Comment créer un modèle dans Django ?"
    result = explicit_scope_refusal(q, CFG)
    assert result is not None
    assert result["status"] == "refused"

def test_out_of_scope_react():
    """Mot-clé React → refus."""
    q = "Quelle est la différence entre useState et useEffect dans React ?"
    result = explicit_scope_refusal(q, CFG)
    assert result is not None
    assert result["status"] == "refused"

def test_out_of_scope_tensorflow_alias():
    """Alias tf. → tensorflow détecté."""
    q = "Comment compiler un modèle avec tf.keras.Model ?"
    result = explicit_scope_refusal(q, CFG)
    assert result is not None
    assert "tensorflow" in result["detected_topic"].lower()

def test_out_of_scope_numpy_alias():
    """Alias np. → numpy détecté."""
    q = "Que fait np.array() ?"
    result = explicit_scope_refusal(q, CFG)
    assert result is not None
    assert "numpy" in result["detected_topic"].lower()


# ---------------------------------------------------------------------------
# Catégorie 3 — Question vide
# ---------------------------------------------------------------------------

def test_empty_question_no_refusal():
    """Question vide → pas de refus explicite (géré par le filtre 2)."""
    assert explicit_scope_refusal("", CFG) is None
    assert explicit_scope_refusal("   ", CFG) is None

def test_empty_question_abstained():
    """Question vide → abstention (pas de contexte) dans evaluate_scope."""
    result = evaluate_scope("", [], CFG)
    assert result["status"] == "abstained"
    assert result["reason"] == "no_retrieved_context"


# ---------------------------------------------------------------------------
# Catégorie 4 — Question ambiguë
# ---------------------------------------------------------------------------

def test_ambiguous_pandas_mention():
    """Mention de pandas dans un contexte comparatif → doit refuser
    car le mot-clé entier est présent."""
    q = "Est-ce que Python est plus rapide que pandas pour le traitement de données ?"
    result = explicit_scope_refusal(q, CFG)
    assert result is not None, "Le mot 'pandas' doit déclencher le filtre."
    assert result["status"] == "refused"

def test_ambiguous_general_question():
    """Question générale sur la programmation → pas de refus explicite."""
    q = "Quelle est la meilleure façon d'organiser un projet Python ?"
    assert explicit_scope_refusal(q, CFG) is None


# ---------------------------------------------------------------------------
# Catégorie 5 — Dans le périmètre mais absent du corpus
# ---------------------------------------------------------------------------

def test_in_scope_low_score_abstained():
    """Question Python valide mais contexte non pertinent → abstention."""
    q = "Comment utiliser le module asyncio pour des tâches parallèles ?"
    # On simule un mauvais score de reranking
    result = evaluate_scope(q, CONTEXT_BAD, CFG)
    assert result["status"] == "abstained", (
        f"Attendu 'abstained', obtenu '{result['status']}'. "
        f"Score : {result.get('confidence')}"
    )

def test_in_scope_good_score_answered():
    """Question Python avec bon contexte → réponse autorisée."""
    q = "C'est quoi Python ?"
    result = evaluate_scope(q, CONTEXT_GOOD, CFG)
    assert result["status"] == "answered"
    assert result["allow_answer"] is True


# ---------------------------------------------------------------------------
# Catégorie 6 — Mot technique mais sujet non couvert
# ---------------------------------------------------------------------------

def test_technical_word_but_out_of_scope():
    """Contient 'modèle' (technique) mais parle de pytorch → refus."""
    q = "Comment entraîner un modèle avec PyTorch ?"
    result = explicit_scope_refusal(q, CFG)
    assert result is not None
    assert result["status"] == "refused"

def test_technical_word_in_scope():
    """Contient 'modèle' et parle de sklearn → dans le périmètre."""
    q = "Comment sauvegarder un modèle Scikit-learn avec joblib ?"
    assert explicit_scope_refusal(q, CFG) is None


# ---------------------------------------------------------------------------
# Exécution standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_in_scope_python,
        test_in_scope_sklearn,
        test_in_scope_langchain,
        test_in_scope_rag,
        test_out_of_scope_pandas_alias,
        test_out_of_scope_django,
        test_out_of_scope_react,
        test_out_of_scope_tensorflow_alias,
        test_out_of_scope_numpy_alias,
        test_empty_question_no_refusal,
        test_empty_question_abstained,
        test_ambiguous_pandas_mention,
        test_ambiguous_general_question,
        test_in_scope_low_score_abstained,
        test_in_scope_good_score_answered,
        test_technical_word_but_out_of_scope,
        test_technical_word_in_scope,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [OK]  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__} : {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERR]  {test.__name__} : {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passés.")
    sys.exit(0 if failed == 0 else 1)
