"""Script de calibration du seuil d'abstention.

Lance uniquement la recherche hybride FAISS/BM25 (pas de LLM, pas de Mistral).
Affiche le score top-1 pour chaque question et la décision prise.

Lancer avec :
    python tests/calibrate_threshold.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import VECTORSTORE_DIR, SCOPE_GUARD_CONFIG, logger
from core.scope_guard import explicit_scope_refusal, evaluate_scope
from core.search import SimpleBM25

# ─────────────────────────────────────────────
# Questions de test (couvrant tous les cas)
# ─────────────────────────────────────────────
QUESTIONS = [
    # ── Dans le périmètre → score attendu > 0.3 (answered) ──
    ("c'est quoi Python ?",                         "IN-SCOPE",  "answered"),
    ("comment fonctionne un décorateur Python ?",   "IN-SCOPE",  "answered"),
    ("qu'est-ce que FAISS ?",                       "IN-SCOPE",  "answered"),
    ("différence entre fit() et fit_transform() ?", "IN-SCOPE",  "answered"),
    ("comment créer une chain dans LangChain ?",    "IN-SCOPE",  "answered"),

    # ── Hors périmètre → bloqué AVANT la recherche (refused) ──
    ("fait quoi pd.head ?",                         "PRE-FILTER", "refused"),
    ("comment créer un modèle Django ?",            "PRE-FILTER", "refused"),
    ("faire un graphe avec matplotlib",             "PRE-FILTER", "refused"),

    # ── Hors domaine → score attendu < 0.3 (abstained) ──
    ("qui est Lamine Yamal ?",                      "OUT-DOMAIN", "abstained"),
    ("comment cuisiner des pâtes carbonara ?",      "OUT-DOMAIN", "abstained"),
    ("quelle est la capitale de la France ?",       "OUT-DOMAIN", "abstained"),
    ("c'est quoi un moteur diesel ?",               "OUT-DOMAIN", "abstained"),
    ("qui a gagné la coupe du monde 2022 ?",        "OUT-DOMAIN", "abstained"),
    ("comment fonctionne un moteur à réaction ?",   "OUT-DOMAIN", "abstained"),
    ("recette du gâteau au chocolat",               "OUT-DOMAIN", "abstained"),
]

THRESHOLD = SCOPE_GUARD_CONFIG.get("minimum_top_score", 0.3)
ALPHA = 0.7
CANDIDATE_K = 20


def load_components():
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    chunks_path = VECTORSTORE_DIR / "chunks_metadata.json"

    print(f"\nChargement de l'index FAISS : {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"Chargement des chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Auto-detection du modele depuis la dimension de l'index
    _DIM_TO_MODEL = {
        384: ("all-MiniLM-L6-v2", ""),
        768: ("intfloat/multilingual-e5-base", "query: "),
    }
    embedding_model_name, query_prefix = _DIM_TO_MODEL.get(
        index.d, ("all-MiniLM-L6-v2", "")
    )

    print(f"  [dim] Index FAISS      : {index.d}")
    print(f"  [dim] Modele selectionne : {embedding_model_name}")

    print(f"Chargement du modele d'embedding : {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    test_emb = embedding_model.encode(["test"], convert_to_numpy=True)
    emb_dim = test_emb.shape[-1]
    if emb_dim != index.d:
        print(f"  [!] Mismatch residuel : modele={emb_dim}d, index={index.d}d")
    else:
        print(f"  [OK] Dimensions coherentes ({index.d}d).")

    texts = [c.get("chunk_text", "") for c in chunks]
    bm25 = SimpleBM25(texts)

    return index, chunks, embedding_model, bm25, query_prefix


def hybrid_top1_score(question, index, chunks, embedding_model, bm25, query_prefix=""):
    """Retourne le score cosinus brut du passage top-1 (avant normalisation)."""
    n = len(chunks)

    query_emb = embedding_model.encode(
        [query_prefix + question], convert_to_numpy=True
    ).astype(np.float32)

    # Garantir la forme (1, d) attendue par FAISS
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    # Tronquer si necessaire (securite)
    if query_emb.shape[1] != index.d:
        query_emb = query_emb[:, :index.d]

    faiss.normalize_L2(query_emb)

    n_candidates = min(n, 50)
    sem_scores_all, sem_indices_all = index.search(query_emb, n_candidates)
    sem_full = np.zeros(n)
    for sc, idx in zip(sem_scores_all[0], sem_indices_all[0]):
        if 0 <= idx < n:
            sem_full[idx] = sc

    bm25_full = bm25.score(question)

    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    hybrid = ALPHA * _norm(sem_full) + (1 - ALPHA) * _norm(bm25_full)
    top1_idx = int(np.argmax(hybrid))
    return float(hybrid[top1_idx])


def main():
    print("\n" + "=" * 70)
    print("  CALIBRATION DU SEUIL D'ABSTENTION")
    print(f"  Seuil configure : {THRESHOLD}")
    print("=" * 70)

    index, chunks, embedding_model, bm25, query_prefix = load_components()

    print(f"\n{'Question':<50} {'Categorie':<12} {'Score':>7}  {'Decision':>10}  {'OK?':>5}")
    print("-" * 95)

    correct = 0
    total = 0

    for question, category, expected in QUESTIONS:
        total += 1

        # Filtre 1 : pre-recherche
        pre_check = explicit_scope_refusal(question, SCOPE_GUARD_CONFIG)
        if pre_check is not None:
            decision = "refused"
            score_display = " (bloque)"
            ok = "[OK]" if decision == expected else "[FAIL]"
            if decision == expected:
                correct += 1
            label = question[:48] + ".." if len(question) > 48 else question
            print(f"{label:<50} {category:<12} {score_display:>7}  {decision:>10}  {ok:>6}")
            continue

        # Filtre 2 : score cosinus brut top-1
        score = hybrid_top1_score(question, index, chunks, embedding_model, bm25, query_prefix)

        # Contexte simule avec raw_cosine_score pour que evaluate_scope utilise le bon seuil
        fake_ctx = [{"raw_cosine_score": score}]
        scope = evaluate_scope(question, fake_ctx, SCOPE_GUARD_CONFIG)
        decision = scope["status"]

        ok = "[OK]  " if decision == expected else "[FAIL]"
        if decision == expected:
            correct += 1

        label = question[:48] + ".." if len(question) > 48 else question
        print(f"{label:<50} {category:<12} {score:>7.4f}  {decision:>10}  {ok:>6}")

    print("-" * 95)
    print(f"\n  Resultat : {correct}/{total} questions classees correctement")
    print(f"  Seuil utilise : {THRESHOLD}")
    if correct == total:
        print("  [OK] Le seuil se generalise correctement a toutes les questions testees.")
    else:
        failed = total - correct
        print(f"  [!] {failed} question(s) mal classee(s) - ajuster le seuil dans config.py.")
    print()


if __name__ == "__main__":
    main()
