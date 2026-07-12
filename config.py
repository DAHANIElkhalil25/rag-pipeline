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
PROJECT_ROOT = Path(__file__).resolve().parent / "rag_project"
DATA_DIR = PROJECT_ROOT / "data"

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
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Arborescence initialisée sous {PROJECT_ROOT.resolve()}")


if __name__ == "__main__":
    init_directories()
    print("✅ Configuration validée. Arborescence créée.")
    print(f"   Racine : {PROJECT_ROOT.resolve()}")
