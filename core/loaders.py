"""
core/loaders.py — Chargement de données et vérification des dépendances
=========================================================================
Fonctions partagées pour charger les documents nettoyés et vérifier
que les packages ML nécessaires sont installés.

Utilisé par : etape3_benchmarking.py, etape4_indexation.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

from config import CLEAN_DIR, logger


def load_cleaned_documents() -> List[Dict]:
    """
    Charge tous les documents nettoyés (sortie étape 2).

    Parcourt les trois sous-dossiers de CLEAN_DIR et charge
    chaque fichier JSON.

    Returns
    -------
    list[dict]
        Liste de documents nettoyés avec clés : cleaned_content,
        text_only, code_blocks, segments, metadata.
    """
    documents = []

    for source_dir in ["python", "sklearn", "langchain"]:
        dir_path = CLEAN_DIR / source_dir
        if not dir_path.exists():
            logger.warning(f"Dossier non trouvé : {dir_path}")
            continue

        json_files = list(dir_path.glob("*.json"))
        logger.info(f"Chargement de {len(json_files)} documents {source_dir}...")

        for fp in json_files:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                documents.append(doc)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Erreur lecture {fp.name}: {e}")

    logger.info(f"Total : {len(documents)} documents nettoyés chargés.")
    return documents


def check_ml_dependencies():
    """
    Vérifie que les packages ML nécessaires sont installés.

    Packages vérifiés : faiss-cpu, sentence-transformers, tiktoken.
    Quitte le programme avec un message clair si un package manque.
    """
    errors = []

    for pkg, imp in [
        ("faiss-cpu", "faiss"),
        ("sentence-transformers", "sentence_transformers"),
        ("tiktoken", "tiktoken"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            errors.append(f"{pkg}  →  pip install {pkg}")

    if errors:
        print("❌ Packages manquants :\n")
        for e in errors:
            print(f"   • {e}")
        print(f"\n   Installation rapide :")
        print(f"   pip install faiss-cpu sentence-transformers tiktoken")
        sys.exit(1)

    print("✅ Toutes les dépendances ML sont présentes.")
