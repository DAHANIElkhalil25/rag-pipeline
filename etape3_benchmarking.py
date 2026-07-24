"""
etape3_benchmarking.py — Benchmarking et justification des choix
=================================================================
Étape 3 du pipeline RAG : comparer rigoureusement les paramètres du
pipeline pour justifier chaque choix technique, soit par la littérature,
soit par une expérimentation sur notre corpus.

Comparaisons réalisées :
  1. Stratégie de chunking     : taille (256, 400, 512) × overlap (30, 50, 80)
  2. Modèle d'embedding        : MiniLM-L6, MiniLM-L12-multilingual, mpnet-base
  3. Méthode de recherche       : sémantique pure vs hybride (sémantique + BM25)

Chaque comparaison produit un tableau de résultats avec les métriques :
  - Precision@k, Recall@k, MRR (Mean Reciprocal Rank)
  - Temps d'indexation et de recherche

Références bibliographiques citées dans le code :
  [1] Lewis et al. (2020) - RAG: Retrieval-Augmented Generation
  [2] Reimers & Gurevych (2019) - Sentence-BERT
  [3] Johnson et al. (2019) - FAISS: Billion-scale similarity search
  [4] Robertson & Zaragoza (2009) - BM25 and Beyond
  [5] Wang et al. (2022) - Text Embeddings by Weakly-Supervised Learning
  [6] Gao et al. (2024) - Retrieval-Augmented Generation: A Survey
  [7] Barnett et al. (2024) - Seven Failure Points in RAG Systems
  [8] Es et al. (2024) - RAGAS: Automated Evaluation of RAG

Entrée :
    rag_project/data/processed/cleaned/{python,sklearn,langchain}/*.json

Sortie :
    rag_project/data/benchmarks/
        ├── benchmark_chunking.csv
        ├── benchmark_embeddings.csv
        ├── benchmark_search.csv
        ├── benchmark_report.json
        └── eval_questions.json          ← Jeu de test réutilisable

Usage:
    python etape3_benchmarking.py
"""

import json
import re
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CLEAN_DIR, BENCHMARK_DIR, METADATA_DIR, logger
from core.tokenizer import TokenCounter
from core.chunker import DocumentChunker
from core.loaders import load_cleaned_documents, check_ml_dependencies
from core.search import SimpleBM25


# ============================================================
# VÉRIFICATION DES DÉPENDANCES
# ============================================================
# → Déplacée dans core/loaders.py : check_ml_dependencies()


# ============================================================
# JEU DE QUESTIONS D'ÉVALUATION
# ============================================================
# Ce jeu de test couvre les trois sources et différents types de questions.
# Chaque question a une réponse attendue et des mots-clés qui DOIVENT
# apparaître dans les chunks pertinents récupérés.
#
# Réf. [8] Es et al. (2024) recommandent au minimum 50 paires QA
# pour une évaluation fiable. Ici on en utilise 20 pour le benchmarking
# (rapide) et on complétera à 50+ dans l'étape d'évaluation finale.

EVAL_QUESTIONS = [
    # ── Python (questions factuelles sur l'API) ──
    {
        "question": "How to list all files in a directory in Python?",
        "expected_keywords": ["listdir", "os", "scandir", "iterdir", "pathlib"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "How to read a JSON file in Python?",
        "expected_keywords": ["json", "load", "open", "read"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "What is a Python decorator and how to use it?",
        "expected_keywords": ["decorator", "wrapper", "function", "@"],
        "source_filter": "python",
        "type": "conceptual",
    },
    {
        "question": "How to handle exceptions in Python?",
        "expected_keywords": ["try", "except", "raise", "exception", "error"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "How to use regular expressions in Python?",
        "expected_keywords": ["re", "match", "search", "pattern", "compile"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "What is the difference between a list and a tuple?",
        "expected_keywords": ["list", "tuple", "mutable", "immutable", "sequence"],
        "source_filter": "python",
        "type": "conceptual",
    },
    {
        "question": "How to use the subprocess module?",
        "expected_keywords": ["subprocess", "run", "Popen", "call", "pipe"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "How to write data to a CSV file in Python?",
        "expected_keywords": ["csv", "writer", "writerow", "DictWriter", "open"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "How does Python garbage collection work?",
        "expected_keywords": ["gc", "reference", "count", "collect", "cycle"],
        "source_filter": "python",
        "type": "conceptual",
    },
    {
        "question": "How to create and use a virtual environment?",
        "expected_keywords": ["venv", "virtualenv", "activate", "pip", "install"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "How to use the logging module in Python?",
        "expected_keywords": ["logging", "logger", "handler", "level", "debug"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "What are context managers and the with statement?",
        "expected_keywords": ["with", "context", "enter", "exit", "manager"],
        "source_filter": "python",
        "type": "conceptual",
    },
    {
        "question": "How to use asyncio for asynchronous programming?",
        "expected_keywords": ["asyncio", "async", "await", "coroutine", "event_loop"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "How to manipulate file paths with pathlib?",
        "expected_keywords": ["pathlib", "Path", "exists", "resolve", "parent"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "What is the Global Interpreter Lock (GIL)?",
        "expected_keywords": ["GIL", "thread", "lock", "interpreter", "concurrent"],
        "source_filter": "python",
        "type": "conceptual",
    },
    {
        "question": "How to use dataclasses in Python?",
        "expected_keywords": ["dataclass", "field", "init", "repr", "frozen"],
        "source_filter": "python",
        "type": "factual",
    },
    {
        "question": "How to sort a list with a custom key function?",
        "expected_keywords": ["sort", "sorted", "key", "lambda", "reverse"],
        "source_filter": "python",
        "type": "factual",
    },
    # ── Scikit-learn (questions ML) ──
    {
        "question": "How to train a random forest classifier?",
        "expected_keywords": ["RandomForest", "fit", "predict", "n_estimators", "classifier"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "What is cross-validation and how to use it in scikit-learn?",
        "expected_keywords": ["cross_val", "fold", "validation", "score", "cv"],
        "source_filter": "sklearn",
        "type": "conceptual",
    },
    {
        "question": "How to scale features before training a model?",
        "expected_keywords": ["scaler", "StandardScaler", "normalize", "fit_transform", "preprocessing"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "How to evaluate a classification model?",
        "expected_keywords": ["accuracy", "precision", "recall", "f1", "confusion_matrix"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "What is a pipeline in scikit-learn?",
        "expected_keywords": ["Pipeline", "steps", "transform", "estimator"],
        "source_filter": "sklearn",
        "type": "conceptual",
    },
    {
        "question": "How to perform dimensionality reduction with PCA?",
        "expected_keywords": ["PCA", "components", "variance", "decomposition", "n_components"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "How to use GridSearchCV for hyperparameter tuning?",
        "expected_keywords": ["GridSearchCV", "param_grid", "best_params", "cv", "scoring"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "How to handle missing values in scikit-learn?",
        "expected_keywords": ["imputer", "SimpleImputer", "missing", "NaN", "strategy"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "What is the difference between bagging and boosting?",
        "expected_keywords": ["bagging", "boosting", "ensemble", "GradientBoosting", "AdaBoost"],
        "source_filter": "sklearn",
        "type": "conceptual",
    },
    {
        "question": "How to perform text classification with scikit-learn?",
        "expected_keywords": ["TfidfVectorizer", "CountVectorizer", "text", "classification", "pipeline"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "How to use K-Means for clustering?",
        "expected_keywords": ["KMeans", "cluster", "centroid", "n_clusters", "fit_predict"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "How to train a Support Vector Machine (SVM)?",
        "expected_keywords": ["SVC", "SVM", "kernel", "fit", "support_vectors"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "How to split data into training and test sets?",
        "expected_keywords": ["train_test_split", "test_size", "random_state", "stratify", "split"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    {
        "question": "What is the bias-variance tradeoff?",
        "expected_keywords": ["bias", "variance", "overfit", "underfit", "generalization"],
        "source_filter": "sklearn",
        "type": "conceptual",
    },
    {
        "question": "How to encode categorical features?",
        "expected_keywords": ["OneHotEncoder", "LabelEncoder", "categorical", "encoding", "ordinal"],
        "source_filter": "sklearn",
        "type": "factual",
    },
    # ── LangChain (questions RAG/LLM) ──
    {
        "question": "How to create a retrieval chain in LangChain?",
        "expected_keywords": ["retrieval", "chain", "retriever", "vector", "document"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "What are embeddings in LangChain and how to use them?",
        "expected_keywords": ["embedding", "vector", "model", "encode", "similarity"],
        "source_filter": "langchain",
        "type": "conceptual",
    },
    {
        "question": "How to use a vector store in LangChain?",
        "expected_keywords": ["vector", "store", "FAISS", "similarity", "search"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "What is an agent in LangChain?",
        "expected_keywords": ["agent", "tool", "action", "reasoning", "LLM"],
        "source_filter": "langchain",
        "type": "conceptual",
    },
    {
        "question": "How to split documents for a RAG pipeline?",
        "expected_keywords": ["split", "chunk", "text", "document", "recursive"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "How to use tools with a LangChain agent?",
        "expected_keywords": ["tool", "agent", "function", "call", "bind"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "What is LangGraph and how does it relate to LangChain?",
        "expected_keywords": ["graph", "state", "node", "agent", "workflow"],
        "source_filter": "langchain",
        "type": "conceptual",
    },
    {
        "question": "How to load documents from a PDF in LangChain?",
        "expected_keywords": ["loader", "PDF", "document", "page", "text"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "How to create a conversational memory in LangChain?",
        "expected_keywords": ["memory", "conversation", "buffer", "history", "chat"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "How to use output parsers in LangChain?",
        "expected_keywords": ["parser", "output", "structured", "format", "pydantic"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "How to create a custom prompt template?",
        "expected_keywords": ["prompt", "template", "variable", "format", "ChatPromptTemplate"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "expected_keywords": ["retrieval", "augmented", "generation", "context", "knowledge"],
        "source_filter": "langchain",
        "type": "conceptual",
    },
    {
        "question": "How to use callbacks for monitoring in LangChain?",
        "expected_keywords": ["callback", "handler", "trace", "monitor", "event"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "How to chain multiple LLM calls together?",
        "expected_keywords": ["chain", "sequential", "LCEL", "pipe", "invoke"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "How to create a multi-modal chain with images and text?",
        "expected_keywords": ["image", "multimodal", "vision", "message", "content"],
        "source_filter": "langchain",
        "type": "factual",
    },
    {
        "question": "What is LangChain Expression Language (LCEL)?",
        "expected_keywords": ["LCEL", "runnable", "pipe", "chain", "expression"],
        "source_filter": "langchain",
        "type": "conceptual",
    },
    {
        "question": "How to stream responses from an LLM in LangChain?",
        "expected_keywords": ["stream", "token", "callback", "chunk", "async"],
        "source_filter": "langchain",
        "type": "factual",
    },
]


# ============================================================
# CHARGEMENT DES DONNÉES NETTOYÉES
# ============================================================
# → Déplacé dans core/loaders.py : load_cleaned_documents()


# ============================================================
# TOKENIZER
# ============================================================
# → Déplacé dans core/tokenizer.py : TokenCounter


# ============================================================
# CHUNKER PARAMÉTRABLE
# ============================================================
# → Déplacé dans core/chunker.py : DocumentChunker
# Usage dans les benchmarks : DocumentChunker(chunk_size=X, chunk_overlap=Y)


# ============================================================
# MÉTRIQUES D'ÉVALUATION DU RETRIEVAL
# ============================================================
# Réf. [8] Es et al. (2024) — RAGAS propose faithfulness, relevancy,
# context_precision, context_recall. Pour le benchmarking du retrieval
# seul (sans LLM), on utilise les métriques IR classiques.

def evaluate_retrieval(
    questions: List[Dict],
    chunks: List[Dict],
    index,
    model,
    k: int = 5
) -> Dict:
    """
    Évalue la qualité du retrieval sur le jeu de questions.

    Métriques calculées :
    - hit_rate@k : % de questions pour lesquelles au moins un chunk
      pertinent est dans le top-k (≡ Recall@k binaire)
    - mrr@k : Mean Reciprocal Rank — position moyenne du premier
      résultat pertinent (plus c'est haut, mieux c'est)
    - avg_precision@k : Precision@k moyenne

    Un chunk est considéré pertinent s'il contient au moins 2 des
    mots-clés attendus (critère souple mais objectif).
    """
    import faiss

    hits, reciprocal_ranks, precisions = [], [], []

    for q in questions:
        query_vec = model.encode([q["question"]], normalize_embeddings=True)
        query_vec = query_vec.astype(np.float32)
        scores, indices = index.search(query_vec, k)

        # Vérifier la pertinence de chaque résultat
        relevant_in_topk = 0
        first_relevant_rank = None
        keywords = [kw.lower() for kw in q["expected_keywords"]]

        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk_text = chunks[idx]["chunk_text"].lower()
            # Un chunk est pertinent s'il contient >= 2 mots-clés
            # Utilisation de word boundary (\b) pour éviter les faux positifs
            # (ex: "re" ne matchera plus "return" ou "represent")
            matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', chunk_text))
            if matches >= 2:
                relevant_in_topk += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank + 1  # 1-indexed

        # Hit rate : au moins un résultat pertinent
        hits.append(1 if relevant_in_topk > 0 else 0)

        # MRR
        if first_relevant_rank is not None:
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Precision@k
        precisions.append(relevant_in_topk / k)

    return {
        "hit_rate": round(np.mean(hits), 4),
        "mrr": round(np.mean(reciprocal_ranks), 4),
        "precision_at_k": round(np.mean(precisions), 4),
        "k": k,
        "n_questions": len(questions),
    }


# ============================================================
# BENCHMARK 1 : CHUNKING
# ============================================================

def benchmark_chunking(documents: List[Dict]) -> pd.DataFrame:
    """
    Compare différentes configurations de chunking.

    Justification littérature :
    - Réf. [6] Gao et al. (2024) : "chunk size is the most critical
      hyperparameter affecting retrieval quality in RAG systems"
    - Réf. [7] Barnett et al. (2024) : chunks trop petits → perte de
      contexte ; chunks trop grands → dilution de l'information pertinente
    - La plage 256-512 tokens est le standard dans la littérature.
      On teste les bornes et le milieu.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print("\n" + "=" * 65)
    print("📊  BENCHMARK 1 — STRATÉGIE DE CHUNKING")
    print("=" * 65)
    print("\n  Réf. Gao et al. (2024) : la taille du chunk est le paramètre")
    print("  le plus impactant sur la qualité du retrieval dans un RAG.")
    print("  On compare 3 tailles × 3 overlaps = 9 configurations.\n")

    # Modèle fixe pour isoler l'effet du chunking
    model = SentenceTransformer("all-MiniLM-L6-v2")

    configs = [
        {"chunk_size": 256, "overlap": 30,  "label": "256 / 30"},
        {"chunk_size": 256, "overlap": 50,  "label": "256 / 50"},
        {"chunk_size": 256, "overlap": 80,  "label": "256 / 80"},
        {"chunk_size": 400, "overlap": 30,  "label": "400 / 30"},
        {"chunk_size": 400, "overlap": 50,  "label": "400 / 50"},
        {"chunk_size": 400, "overlap": 80,  "label": "400 / 80"},
        {"chunk_size": 512, "overlap": 30,  "label": "512 / 30"},
        {"chunk_size": 512, "overlap": 50,  "label": "512 / 50"},
        {"chunk_size": 512, "overlap": 80,  "label": "512 / 80"},
    ]

    results = []
    for cfg in tqdm(configs, desc="✂️  Configs chunking"):
        chunker = DocumentChunker(cfg["chunk_size"], cfg["overlap"])

        # Chunking
        t0 = time.time()
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker.chunk_document(doc))
        chunk_time = time.time() - t0

        if len(all_chunks) < 10:
            logger.warning(f"Config {cfg['label']}: seulement {len(all_chunks)} chunks")
            continue

        # Embedding
        texts = [c["chunk_text"] for c in all_chunks]
        t0 = time.time()
        embeddings = model.encode(texts, batch_size=64,
                                   normalize_embeddings=True, show_progress_bar=False)
        embed_time = time.time() - t0

        # Index FAISS
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))

        # Évaluation
        metrics = evaluate_retrieval(EVAL_QUESTIONS, all_chunks, index, model, k=5)

        results.append({
            "chunk_size": cfg["chunk_size"],
            "overlap": cfg["overlap"],
            "label": cfg["label"],
            "n_chunks": len(all_chunks),
            "avg_tokens": round(np.mean([c["token_count"] for c in all_chunks]), 1),
            "hit_rate@5": metrics["hit_rate"],
            "mrr@5": metrics["mrr"],
            "precision@5": metrics["precision_at_k"],
            "chunk_time_s": round(chunk_time, 2),
            "embed_time_s": round(embed_time, 2),
        })

        print(f"  {cfg['label']:<10} → {len(all_chunks):>6} chunks  "
              f"HR@5={metrics['hit_rate']:.3f}  MRR={metrics['mrr']:.3f}  "
              f"P@5={metrics['precision_at_k']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(BENCHMARK_DIR / "benchmark_chunking.csv", index=False)

    # Meilleur
    if len(df) > 0:
        best = df.loc[df["mrr@5"].idxmax()]
        print(f"\n  🏆 Meilleure config : taille={int(best['chunk_size'])}, "
              f"overlap={int(best['overlap'])} "
              f"(MRR={best['mrr@5']:.3f}, HR@5={best['hit_rate@5']:.3f})")

    return df


# ============================================================
# BENCHMARK 2 : MODÈLES D'EMBEDDING
# ============================================================

def benchmark_embeddings(documents: List[Dict]) -> pd.DataFrame:
    """
    Compare différents modèles d'embedding.

    Justification littérature :
    - Réf. [2] Reimers & Gurevych (2019) : Sentence-BERT est le standard
      pour les embeddings de phrases, avec des variantes optimisées pour
      différents usages (vitesse, multilingue, précision).
    - Réf. [5] Wang et al. (2022) : les modèles plus récents (E5, GTE)
      surpassent les modèles SBERT classiques, mais sont plus lourds.

    Modèles comparés :
    - all-MiniLM-L6-v2 : léger (80Mo), rapide, 384 dim. Standard pour le prototypage.
    - paraphrase-multilingual-MiniLM-L12-v2 : multilingue (50+ langues), 384 dim.
      Pertinent si les questions ou documents sont en français.
    - all-mpnet-base-v2 : plus lourd (420Mo), 768 dim. Meilleur en précision
      selon le benchmark MTEB, mais 3x plus lent.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print("\n" + "=" * 65)
    print("📊  BENCHMARK 2 — MODÈLES D'EMBEDDING")
    print("=" * 65)
    print("\n  Réf. Reimers & Gurevych (2019) : Sentence-BERT.")
    print("  On compare 3 modèles de complexité croissante.\n")

    # Chunking fixe (config par défaut) pour isoler l'effet du modèle
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk_document(doc))

    texts = [c["chunk_text"] for c in all_chunks]
    logger.info(f"Chunks fixés : {len(all_chunks)} (400 tokens, overlap 50)")

    models_to_test = [
        {
            "name": "all-MiniLM-L6-v2",
            "description": "Léger, rapide, anglais (SBERT standard)",
            "dim": 384,
        },
        {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "description": "Multilingue (50+ langues), 384 dim",
            "dim": 384,
        },
        {
            "name": "all-mpnet-base-v2",
            "description": "Plus précis (MTEB top), 768 dim, plus lent",
            "dim": 768,
        },
    ]

    results = []
    for model_cfg in models_to_test:
        model_name = model_cfg["name"]
        print(f"  Chargement : {model_name}...")

        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning(f"Impossible de charger {model_name}: {e}")
            print(f"  ⚠️  {model_name} — échec de chargement, ignoré.")
            continue

        # Embedding
        t0 = time.time()
        embeddings = model.encode(texts, batch_size=64,
                                   normalize_embeddings=True, show_progress_bar=False)
        embed_time = time.time() - t0

        # Index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))

        # Évaluation
        t0 = time.time()
        metrics = evaluate_retrieval(EVAL_QUESTIONS, all_chunks, index, model, k=5)
        search_time = time.time() - t0

        results.append({
            "model": model_name,
            "description": model_cfg["description"],
            "embedding_dim": embeddings.shape[1],
            "hit_rate@5": metrics["hit_rate"],
            "mrr@5": metrics["mrr"],
            "precision@5": metrics["precision_at_k"],
            "embed_time_s": round(embed_time, 2),
            "search_time_s": round(search_time, 4),
            "model_size_mb": "~80" if "MiniLM" in model_name else "~420",
        })

        print(f"  {model_name:<45} HR@5={metrics['hit_rate']:.3f}  "
              f"MRR={metrics['mrr']:.3f}  P@5={metrics['precision_at_k']:.3f}  "
              f"({embed_time:.1f}s)")

        # Libérer la mémoire
        del model, embeddings, index

    df = pd.DataFrame(results)
    df.to_csv(BENCHMARK_DIR / "benchmark_embeddings.csv", index=False)

    if len(df) > 0:
        best = df.loc[df["mrr@5"].idxmax()]
        print(f"\n  🏆 Meilleur modèle : {best['model']} "
              f"(MRR={best['mrr@5']:.3f}, HR@5={best['hit_rate@5']:.3f})")

    return df


# ============================================================
# BENCHMARK 3 : MÉTHODE DE RECHERCHE
# ============================================================

def benchmark_search(documents: List[Dict]) -> pd.DataFrame:
    """
    Compare recherche sémantique pure vs recherche hybride (sémantique + BM25).

    Justification littérature :
    - Réf. [4] Robertson & Zaragoza (2009) : BM25 est le standard
      pour la recherche lexicale depuis 30 ans.
    - Réf. [6] Gao et al. (2024) : "hybrid search combining dense and
      sparse retrieval consistently outperforms either method alone,
      especially for queries containing technical terms."
    - L'intuition : "os.listdir" est mieux trouvé par BM25 (correspondance
      exacte) tandis que "how to list files" est mieux trouvé par la
      recherche sémantique (paraphrase).

    Implémentation BM25 simplifiée (sans dépendance externe) pour éviter
    les problèmes d'installation.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print("\n" + "=" * 65)
    print("📊  BENCHMARK 3 — MÉTHODE DE RECHERCHE")
    print("=" * 65)
    print("\n  Réf. Gao et al. (2024) : la recherche hybride (dense + sparse)")
    print("  surpasse systématiquement chaque méthode seule.\n")

    # Chunking et embedding fixes
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk_document(doc))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["chunk_text"] for c in all_chunks]

    embeddings = model.encode(texts, batch_size=64,
                               normalize_embeddings=True, show_progress_bar=False)

    # Index sémantique
    sem_index = faiss.IndexFlatIP(embeddings.shape[1])
    sem_index.add(embeddings.astype(np.float32))

    # Index BM25 (implémentation légère)
    bm25_index = SimpleBM25(texts)

    results = []

    # ── Test 1 : Sémantique pure ──
    metrics_sem = evaluate_retrieval(EVAL_QUESTIONS, all_chunks, sem_index, model, k=5)
    results.append({
        "method": "Sémantique pure (FAISS)",
        "description": "Similarité cosinus sur embeddings denses",
        "hit_rate@5": metrics_sem["hit_rate"],
        "mrr@5": metrics_sem["mrr"],
        "precision@5": metrics_sem["precision_at_k"],
        "reference": "Reimers & Gurevych (2019)",
    })
    print(f"  Sémantique pure     : HR@5={metrics_sem['hit_rate']:.3f}  "
          f"MRR={metrics_sem['mrr']:.3f}  P@5={metrics_sem['precision_at_k']:.3f}")

    # ── Test 2 : BM25 pure ──
    metrics_bm25 = evaluate_bm25(EVAL_QUESTIONS, all_chunks, bm25_index, k=5)
    results.append({
        "method": "BM25 pure (lexicale)",
        "description": "Correspondance de termes pondérée par TF-IDF",
        "hit_rate@5": metrics_bm25["hit_rate"],
        "mrr@5": metrics_bm25["mrr"],
        "precision@5": metrics_bm25["precision_at_k"],
        "reference": "Robertson & Zaragoza (2009)",
    })
    print(f"  BM25 pure           : HR@5={metrics_bm25['hit_rate']:.3f}  "
          f"MRR={metrics_bm25['mrr']:.3f}  P@5={metrics_bm25['precision_at_k']:.3f}")

    # ── Test 3 : Hybride (sémantique + BM25) ──
    metrics_hybrid = evaluate_hybrid(
        EVAL_QUESTIONS, all_chunks, sem_index, bm25_index, model,
        alpha=0.7, k=5  # 70% sémantique, 30% BM25
    )
    results.append({
        "method": "Hybride (70% sém. + 30% BM25)",
        "description": "Fusion pondérée des scores normalisés",
        "hit_rate@5": metrics_hybrid["hit_rate"],
        "mrr@5": metrics_hybrid["mrr"],
        "precision@5": metrics_hybrid["precision_at_k"],
        "reference": "Gao et al. (2024)",
    })
    print(f"  Hybride (α=0.7)     : HR@5={metrics_hybrid['hit_rate']:.3f}  "
          f"MRR={metrics_hybrid['mrr']:.3f}  P@5={metrics_hybrid['precision_at_k']:.3f}")

    # ── Test 4 : Hybride 50/50 ──
    metrics_h50 = evaluate_hybrid(
        EVAL_QUESTIONS, all_chunks, sem_index, bm25_index, model,
        alpha=0.5, k=5
    )
    results.append({
        "method": "Hybride (50% sém. + 50% BM25)",
        "description": "Fusion pondérée équilibrée",
        "hit_rate@5": metrics_h50["hit_rate"],
        "mrr@5": metrics_h50["mrr"],
        "precision@5": metrics_h50["precision_at_k"],
        "reference": "Gao et al. (2024)",
    })
    print(f"  Hybride (α=0.5)     : HR@5={metrics_h50['hit_rate']:.3f}  "
          f"MRR={metrics_h50['mrr']:.3f}  P@5={metrics_h50['precision_at_k']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(BENCHMARK_DIR / "benchmark_search.csv", index=False)

    if len(df) > 0:
        best = df.loc[df["mrr@5"].idxmax()]
        print(f"\n  🏆 Meilleure méthode : {best['method']} "
              f"(MRR={best['mrr@5']:.3f})")

    del model, embeddings, sem_index
    return df


# ============================================================
# BM25 SIMPLIFIÉ (SANS DÉPENDANCE EXTERNE)
# ============================================================
# → Déplacé dans core/search.py : SimpleBM25


def evaluate_bm25(questions, chunks, bm25_index, k=5):
    """Évalue le retrieval BM25 sur le jeu de questions."""
    hits, reciprocal_ranks, precisions = [], [], []

    for q in questions:
        scores, indices = bm25_index.search(q["question"], k)
        keywords = [kw.lower() for kw in q["expected_keywords"]]
        relevant_in_topk = 0
        first_relevant_rank = None

        for rank, idx in enumerate(indices):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk_text = chunks[idx]["chunk_text"].lower()
            matches = sum(1 for kw in keywords if kw in chunk_text)
            if matches >= 2:
                relevant_in_topk += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank + 1

        hits.append(1 if relevant_in_topk > 0 else 0)
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)
        precisions.append(relevant_in_topk / k)

    return {
        "hit_rate": round(np.mean(hits), 4),
        "mrr": round(np.mean(reciprocal_ranks), 4),
        "precision_at_k": round(np.mean(precisions), 4),
    }


def evaluate_hybrid(questions, chunks, sem_index, bm25_index, model, alpha=0.7, k=5):
    """
    Évalue la recherche hybride (fusion de scores sémantiques et BM25).

    alpha : poids de la recherche sémantique (1-alpha = poids BM25).
    Les scores sont normalisés min-max avant fusion.
    """
    import faiss
    n = len(chunks)
    hits, reciprocal_ranks, precisions = [], [], []

    for q in questions:
        # Scores sémantiques
        query_vec = model.encode([q["question"]], normalize_embeddings=True).astype(np.float32)
        sem_scores_topk, sem_idx = sem_index.search(query_vec, min(n, 50))
        sem_full = np.zeros(n)
        for sc, idx in zip(sem_scores_topk[0], sem_idx[0]):
            if 0 <= idx < n:
                sem_full[idx] = sc

        # Scores BM25
        bm25_full = bm25_index.score(q["question"])

        # Normalisation min-max
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-10)

        sem_norm = norm(sem_full)
        bm25_norm = norm(bm25_full)

        # Fusion
        hybrid = alpha * sem_norm + (1 - alpha) * bm25_norm
        top_k_idx = np.argsort(hybrid)[::-1][:k]

        keywords = [kw.lower() for kw in q["expected_keywords"]]
        relevant_in_topk = 0
        first_relevant_rank = None

        for rank, idx in enumerate(top_k_idx):
            chunk_text = chunks[idx]["chunk_text"].lower()
            matches = sum(1 for kw in keywords if kw in chunk_text)
            if matches >= 2:
                relevant_in_topk += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank + 1

        hits.append(1 if relevant_in_topk > 0 else 0)
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)
        precisions.append(relevant_in_topk / k)

    return {
        "hit_rate": round(np.mean(hits), 4),
        "mrr": round(np.mean(reciprocal_ranks), 4),
        "precision_at_k": round(np.mean(precisions), 4),
    }


# ============================================================
# BENCHMARK COMPLET — GRID SEARCH (108 CONFIGURATIONS)
# ============================================================

def benchmark_full_grid(documents: List[Dict]) -> pd.DataFrame:
    """
    Évalue le produit cartésien complet de toutes les configurations.

    Espace de recherche :
      9 configs chunking × 3 modèles embedding × 4 méthodes recherche = 108

    Justification littérature :
    - Réf. [6] Gao et al. (2024) : les interactions entre chunk_size et
      modèle d'embedding sont significatives — un grid search complet
      est nécessaire pour identifier l'optimum global.

    Optimisation : les embeddings sont pré-calculés par paire
    (chunk_config, model), puis les 4 méthodes de recherche sont
    évaluées sur le même index → 27 passes coûteuses + 108 évaluations.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print("\n" + "=" * 65)
    print("📊  GRID SEARCH COMPLET — 108 CONFIGURATIONS")
    print("=" * 65)
    print("\n  Produit cartésien : 9 chunking × 3 embeddings × 4 recherche")
    print("  Réf. Gao et al. (2024) : évaluation exhaustive recommandée.\n")

    chunk_configs = [
        {"chunk_size": 256, "overlap": 30},
        {"chunk_size": 256, "overlap": 50},
        {"chunk_size": 256, "overlap": 80},
        {"chunk_size": 400, "overlap": 30},
        {"chunk_size": 400, "overlap": 50},
        {"chunk_size": 400, "overlap": 80},
        {"chunk_size": 512, "overlap": 30},
        {"chunk_size": 512, "overlap": 50},
        {"chunk_size": 512, "overlap": 80},
    ]

    model_names = [
        "all-MiniLM-L6-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-mpnet-base-v2",
    ]

    search_methods = [
        {"name": "Sémantique pure", "type": "semantic"},
        {"name": "BM25 pure", "type": "bm25"},
        {"name": "Hybride α=0.7", "type": "hybrid", "alpha": 0.7},
        {"name": "Hybride α=0.5", "type": "hybrid", "alpha": 0.5},
    ]

    total_configs = len(chunk_configs) * len(model_names) * len(search_methods)
    results = []
    config_count = 0

    for model_name in model_names:
        print(f"\n  🧠 Chargement modèle : {model_name}...")
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning(f"Impossible de charger {model_name}: {e}")
            continue

        for cfg in chunk_configs:
            # ── Chunking ──
            chunker = DocumentChunker(cfg["chunk_size"], cfg["overlap"])
            all_chunks = []
            for doc in documents:
                all_chunks.extend(chunker.chunk_document(doc))

            if len(all_chunks) < 10:
                logger.warning(f"Config {cfg}: seulement {len(all_chunks)} chunks, ignoré")
                config_count += len(search_methods)
                continue

            texts = [c["chunk_text"] for c in all_chunks]

            # ── Embedding (coûteux — une seule fois par paire) ──
            embeddings = model.encode(
                texts, batch_size=64,
                normalize_embeddings=True, show_progress_bar=False
            )

            # ── Index FAISS ──
            sem_index = faiss.IndexFlatIP(embeddings.shape[1])
            sem_index.add(embeddings.astype(np.float32))

            # ── Index BM25 ──
            bm25_index = SimpleBM25(texts)

            # ── Évaluer les 4 méthodes de recherche ──
            for search_cfg in search_methods:
                config_count += 1
                search_type = search_cfg["type"]

                if search_type == "semantic":
                    metrics = evaluate_retrieval(
                        EVAL_QUESTIONS, all_chunks, sem_index, model, k=5
                    )
                elif search_type == "bm25":
                    metrics = evaluate_bm25(
                        EVAL_QUESTIONS, all_chunks, bm25_index, k=5
                    )
                elif search_type == "hybrid":
                    metrics = evaluate_hybrid(
                        EVAL_QUESTIONS, all_chunks, sem_index, bm25_index,
                        model, alpha=search_cfg["alpha"], k=5
                    )
                else:
                    continue

                results.append({
                    "chunk_size": cfg["chunk_size"],
                    "overlap": cfg["overlap"],
                    "model": model_name,
                    "search_method": search_cfg["name"],
                    "n_chunks": len(all_chunks),
                    "embedding_dim": embeddings.shape[1],
                    "hit_rate@5": metrics["hit_rate"],
                    "mrr@5": metrics["mrr"],
                    "precision@5": metrics["precision_at_k"],
                })

                print(f"  [{config_count:>3}/{total_configs}] "
                      f"chunk={cfg['chunk_size']}/{cfg['overlap']}  "
                      f"model={model_name.split('/')[-1][:15]:<15}  "
                      f"search={search_cfg['name']:<20}  "
                      f"MRR={metrics['mrr']:.3f}  HR={metrics['hit_rate']:.3f}")

            # Libérer l'index pour cette config
            del sem_index, bm25_index

        # Libérer le modèle
        del model

    df = pd.DataFrame(results)
    df.to_csv(BENCHMARK_DIR / "benchmark_full_grid.csv", index=False)

    if len(df) > 0:
        best = df.loc[df["mrr@5"].idxmax()]
        print(f"\n  {'=' * 60}")
        print(f"  🏆 CONFIGURATION OPTIMALE GLOBALE :")
        print(f"     Chunking   : taille={int(best['chunk_size'])}, "
              f"overlap={int(best['overlap'])}")
        print(f"     Embedding  : {best['model']}")
        print(f"     Recherche  : {best['search_method']}")
        print(f"     MRR@5      : {best['mrr@5']:.4f}")
        print(f"     HR@5       : {best['hit_rate@5']:.4f}")
        print(f"     P@5        : {best['precision@5']:.4f}")
        print(f"  {'=' * 60}")

    return df


# ============================================================
# RAPPORT FINAL
# ============================================================

def generate_report(df_chunk, df_embed, df_search, df_grid, elapsed):
    """Génère le rapport consolidé du benchmarking."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_eval_questions": len(EVAL_QUESTIONS),
        "methodology": "full_grid_search" if len(df_grid) > 0 else "OFAT",
        "n_configurations_tested": len(df_grid) if len(df_grid) > 0 else (
            len(df_chunk) + len(df_embed) + len(df_search)),
        "literature_references": {
            "RAG": "Lewis et al. (2020) - Retrieval-Augmented Generation",
            "SBERT": "Reimers & Gurevych (2019) - Sentence-BERT",
            "FAISS": "Johnson et al. (2019) - Billion-scale similarity search",
            "BM25": "Robertson & Zaragoza (2009) - BM25 and Beyond",
            "RAG_survey": "Gao et al. (2024) - Retrieval-Augmented Generation: A Survey",
            "RAG_failures": "Barnett et al. (2024) - Seven Failure Points in RAG",
            "RAGAS": "Es et al. (2024) - RAGAS: Automated Evaluation of RAG",
        },
        "recommendations": {},
    }

    # Recommandations depuis le grid search (prioritaire)
    if len(df_grid) > 0:
        best_global = df_grid.loc[df_grid["mrr@5"].idxmax()]
        report["recommendations"]["chunking"] = {
            "chunk_size": int(best_global["chunk_size"]),
            "overlap": int(best_global["overlap"]),
            "mrr": float(best_global["mrr@5"]),
            "justification": (
                f"Configuration optimale identifiée par grid search exhaustif "
                f"sur {len(df_grid)} configurations. "
                f"Réf. Gao et al. (2024)."
            ),
        }
        report["recommendations"]["embedding_model"] = {
            "model": best_global["model"],
            "mrr": float(best_global["mrr@5"]),
            "justification": (
                f"Meilleur MRR@5 en combinaison avec chunk_size="
                f"{int(best_global['chunk_size'])} et {best_global['search_method']}."
            ),
        }
        report["recommendations"]["search_method"] = {
            "method": best_global["search_method"],
            "mrr": float(best_global["mrr@5"]),
            "justification": (
                f"Meilleur MRR@5 en combinaison optimale globale. "
                f"Réf. Gao et al. (2024)."
            ),
        }
    else:
        # Fallback sur les benchmarks OFAT individuels
        if len(df_chunk) > 0:
            best_chunk = df_chunk.loc[df_chunk["mrr@5"].idxmax()]
            report["recommendations"]["chunking"] = {
                "chunk_size": int(best_chunk["chunk_size"]),
                "overlap": int(best_chunk["overlap"]),
                "mrr": float(best_chunk["mrr@5"]),
                "justification": "Meilleur MRR@5 sur le jeu de test."
            }

        if len(df_embed) > 0:
            best_embed = df_embed.loc[df_embed["mrr@5"].idxmax()]
            report["recommendations"]["embedding_model"] = {
                "model": best_embed["model"],
                "mrr": float(best_embed["mrr@5"]),
                "justification": "Meilleur MRR@5. "
                    "Réf. Reimers & Gurevych (2019)."
            }

        if len(df_search) > 0:
            best_search = df_search.loc[df_search["mrr@5"].idxmax()]
            report["recommendations"]["search_method"] = {
                "method": best_search["method"],
                "mrr": float(best_search["mrr@5"]),
                "justification": "Meilleur MRR@5. "
                    "Réf. Gao et al. (2024)."
            }

    with open(BENCHMARK_DIR / "benchmark_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Sauvegarder aussi les questions de test pour réutilisation
    with open(BENCHMARK_DIR / "eval_questions.json", 'w', encoding='utf-8') as f:
        json.dump(EVAL_QUESTIONS, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("📋  RECOMMANDATIONS FINALES")
    print("=" * 65)
    for key, rec in report["recommendations"].items():
        if "model" in rec:
            print(f"\n  {key} : {rec['model']} (MRR={rec['mrr']:.3f})")
        elif "method" in rec:
            print(f"\n  {key} : {rec['method']} (MRR={rec['mrr']:.3f})")
        elif "chunk_size" in rec:
            print(f"\n  {key} : taille={rec['chunk_size']}, overlap={rec['overlap']} "
                  f"(MRR={rec['mrr']:.3f})")
        print(f"    → {rec['justification']}")

    print(f"\n📁  Fichiers générés :")
    for f in sorted(BENCHMARK_DIR.glob("*")):
        print(f"    {f.name}")

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("📊  ÉTAPE 3 — BENCHMARKING ET JUSTIFICATION DES CHOIX")
    print("=" * 65)
    print("\n  Cette étape compare rigoureusement les paramètres du pipeline")
    print("  pour justifier chaque choix technique par l'expérimentation.\n")

    start = datetime.now()

    # Vérifications
    check_ml_dependencies()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Charger les données
    print("\n" + "─" * 50)
    print("📂  Chargement des documents nettoyés (étape 2)")
    print("─" * 50)
    documents = load_cleaned_documents()
    if not documents:
        print(f"\n❌ Aucun document trouvé dans {CLEAN_DIR}")
        print(f"   Exécutez d'abord : python main.py --etape 2")
        sys.exit(1)
    print(f"  → {len(documents)} documents chargés.\n")

    # Benchmarks individuels (diagnostic rapide)
    df_chunk = benchmark_chunking(documents)
    df_embed = benchmark_embeddings(documents)
    df_search = benchmark_search(documents)

    # Grid search complet (108 configurations)
    print("\n" + "─" * 50)
    print("🔬  Lancement du grid search complet (108 configs)")
    print("─" * 50)
    df_grid = benchmark_full_grid(documents)

    # Rapport
    elapsed = (datetime.now() - start).total_seconds()
    generate_report(df_chunk, df_embed, df_search, df_grid, elapsed)

    print("\n" + "=" * 65)
    print("🎉  Étape 3 terminée !")
    print(f"    → {len(df_grid)} configurations évaluées (grid search)")
    print(f"    → Benchmarks sauvegardés dans {BENCHMARK_DIR}")
    print(f"    → Utilisez les recommandations pour l'étape 4 (indexation).")
    print(f"    ⏱️  Durée totale : {elapsed/60:.1f} minutes")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
