"""
config.py — Configuration centralisée du projet RAG
====================================================
Ce fichier contient tous les chemins, constantes et paramètres
partagés entre les étapes du pipeline. Modifier ici = modifier partout.

Usage:
    from config import *
    ou
    from config import PROJECT_ROOT, CONFIG
"""

import logging
import os
from pathlib import Path

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("rag_pipeline")

# ============================================================
# CHEMINS DU PROJET
# ============================================================
PROJECT_ROOT = Path(os.getenv("RAG_PROJECT_ROOT", Path(__file__).resolve().parent / "rag_project"))
# Kaggle can redirect generated artifacts outside the source checkout so that
# re-cloning the repository does not erase the corpus or experiment results.
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", PROJECT_ROOT / "data"))

# Données brutes (sortie étape 1)
RAW_DIR = DATA_DIR / "raw"
PYTHON_RAW = RAW_DIR / "python_docs"
SKLEARN_RAW = RAW_DIR / "sklearn_docs"
LANGCHAIN_RAW = RAW_DIR / "langchain_docs"

# Données nettoyées (sortie étape 2)
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = PROCESSED_DIR / "cleaned"

# Base vectorielle (sortie étape 4)
VECTORSTORE_DIR = PROCESSED_DIR / "vectorstore"

# Benchmarking (sortie étape 3)
BENCHMARK_DIR = DATA_DIR / "benchmarks"

# Métadonnées
METADATA_DIR = DATA_DIR / "metadata"

# Évaluation (sortie étape 6)
EVALUATION_DIR = DATA_DIR / "evaluation"
EVALUATION_DATASETS_DIR = EVALUATION_DIR / "datasets"
EVALUATION_RUNS_DIR = EVALUATION_DIR / "runs"

# ============================================================
# CONFIGURATION DU LLM (ÉTAPE 5)
# ============================================================
LLM_CONFIG = {
    # ── Modèle de génération ──
    # Mistral 7B Instruct : meilleur ratio performance/taille
    # Réf. Jiang et al. (2023) : surpasse LLaMA 2 13B
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",

    # ── Quantification ──
    # 4-bit (NF4) via bitsandbytes : réduit VRAM de 14GB → ~5GB
    # Compatible avec Kaggle T4 (16GB VRAM)
    "quantization": "4bit",

    # ── Paramètres de génération ──
    "max_input_tokens": 3072,
    "max_new_tokens": 256,
    "temperature": 0.1,      # Basse pour des réponses factuelles
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "request_timeout_seconds": 180,
    "max_context_chars_per_chunk": 2600,

    # ── Retrieval ──
    "top_k_retrieval": 5,    # Nombre de passages à retrouver

    # ── Ollama fallback (si pas de GPU) ──
    "ollama_url": "http://localhost:11434",
    "ollama_model": "mistral",
}

# Paramètres de l'évaluation : les échecs de jugement sont conservés comme
# invalides et ne sont jamais remplacés silencieusement par 0.5.
EVALUATION_CONFIG = {
    "max_claims_per_answer": 8,
    "max_contexts_per_sample": 5,
    "judge_temperature": 0.0,
    "cache_judgments": True,
}

# Configuration de l'évaluation finale. La baseline historique est conservée
# séparément : ce bloc ne doit être utilisé que pour les expériences finales.
RAGAS_CONFIG = {
    "ragas_version": "0.4.3",
    "metrics": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "factual_correctness",
    ],
    "batch_size": 10,
    "checkpoint_every": 10,
    "default_split": "dev",
    "judge_mode": "external_or_adapter",
}

# Controlled retrieval profiles. Set RAG_RETRIEVAL_PROFILE before indexing;
# the selected profile is persisted in the index manifest for reproducibility.
RETRIEVAL_PROFILES = {
    "baseline": {
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_query_prefix": "",
        "embedding_passage_prefix": "",
        "search_method": "hybrid",
        "alpha": 0.7,
        "candidate_k": 5,
        "final_k": 5,
        "reranker": {"enabled": False},
    },
    "multilingual_hybrid": {
        "embedding_model": "intfloat/multilingual-e5-base",
        "embedding_query_prefix": "query: ",
        "embedding_passage_prefix": "passage: ",
        "search_method": "hybrid",
        "alpha": 0.7,
        "candidate_k": 20,
        "final_k": 5,
        "reranker": {"enabled": False},
    },
    "multilingual_hybrid_rerank": {
        "embedding_model": "intfloat/multilingual-e5-base",
        "embedding_query_prefix": "query: ",
        "embedding_passage_prefix": "passage: ",
        "search_method": "hybrid",
        "alpha": 0.7,
        "candidate_k": 20,
        "final_k": 5,
        "reranker": {
            "enabled": True,
            "model_name": "BAAI/bge-reranker-v2-m3",
            "max_length": 512,
            "unload_after_query": False,
        },
    },
}
DEFAULT_RETRIEVAL_PROFILE = "multilingual_hybrid_rerank"

# Configuration de l'interface Gradio destinée à Kaggle.
UI_CONFIG = {
    "share": True,
    "server_name": "0.0.0.0",
    "server_port": 7860,
}

# ============================================================
# CONFIGURATION DES SOURCES
# ============================================================
CONFIG = {
    # ── Python Documentation ──
    "python": {
        "repo_url": "https://github.com/python/cpython.git",
        "branch": "main",
        "doc_subdir": "Doc",
        "version": "3.13",
        "extensions": [".rst"],
        "exclude_dirs": ["_build", "_static", "_templates", "tools"],
    },

    # ── Scikit-learn Documentation ──
    "sklearn": {
        "repo_url": "https://github.com/scikit-learn/scikit-learn.git",
        "branch": "main",
        "doc_subdir": "doc",
        "version": "1.5",
        "extensions": [".rst"],
        "exclude_dirs": ["_build", "_static", "_templates", "themes", "images"],
    },

    # ── LangChain Documentation (clone GitHub) ──
    "langchain": {
        "repo_url": "https://github.com/langchain-ai/docs.git",
        "branch": "main",
        "version": "latest",
        "target_dirs": [
            "src/oss/python/langchain",
            "src/oss/python/langgraph",
            "src/oss/python/integrations",
        ],
        "extensions": [".mdx", ".md"],
        "exclude_patterns": [
            "node_modules", "_build", ".git", "__pycache__",
            "javascript", "typescript",
        ],
    },
}

# ============================================================
# INITIALISATION DE L'ARBORESCENCE
# ============================================================
def init_directories():
    """Crée tous les dossiers nécessaires au projet."""
    dirs = [
        PYTHON_RAW / "json_docs",
        SKLEARN_RAW / "json_docs",
        LANGCHAIN_RAW / "json_docs",
        CLEAN_DIR / "python",
        CLEAN_DIR / "sklearn",
        CLEAN_DIR / "langchain",
        VECTORSTORE_DIR,
        BENCHMARK_DIR,
        METADATA_DIR,
        EVALUATION_DIR,
        EVALUATION_DATASETS_DIR,
        EVALUATION_RUNS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Arborescence initialisée sous {PROJECT_ROOT.resolve()}")


if __name__ == "__main__":
    init_directories()
    print("✅ Configuration validée. Arborescence créée.")
    print(f"   Racine : {PROJECT_ROOT.resolve()}")
