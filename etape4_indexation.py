"""
etape4_indexation.py — Chunking et Indexation Vectorielle
==========================================================
Étape 4 du pipeline RAG : découper les documents nettoyés en chunks,
les convertir en vecteurs (embeddings), et les stocker dans un index FAISS.

Si l'étape 3 (benchmarking) a été exécutée, cette étape utilise
automatiquement les paramètres optimaux recommandés. Sinon, elle
utilise une configuration par défaut raisonnable.

Pipeline :
  Document nettoyé JSON (étape 2)
    → Chunking (découpage intelligent par section + taille)
    → Embedding (Sentence-Transformers → vecteurs 384d)
    → Indexation (FAISS → recherche de similarité)
  Base vectorielle FAISS (sortie)

Entrée :
    rag_project/data/processed/cleaned/{python,sklearn,langchain}/*.json
    rag_project/data/benchmarks/benchmark_report.json  (optionnel, étape 3)

Sortie :
    rag_project/data/vectorstore/
        ├── faiss_index.bin        ← Index FAISS (vecteurs)
        ├── chunks_metadata.json   ← Métadonnées des chunks
        └── chunking_report.json   ← Rapport de l'étape

Usage:
    python etape4_indexation.py

Prérequis:
    pip install sentence-transformers faiss-cpu tiktoken numpy pandas tqdm
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CLEAN_DIR, PROCESSED_DIR, METADATA_DIR, BENCHMARK_DIR, logger
from core.chunker import DocumentChunker
from core.loaders import load_cleaned_documents, check_ml_dependencies

# ── Chemins de sortie (étape 4) ──
VECTORSTORE_DIR = PROCESSED_DIR / "vectorstore"

FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
CHUNKS_META_PATH = VECTORSTORE_DIR / "chunks_metadata.json"
REPORT_PATH = METADATA_DIR / "chunking_report.json"


# ============================================================
# CONFIGURATION DU CHUNKING
# ============================================================

# Configuration par défaut (utilisée si le benchmarking n'a pas été exécuté)
DEFAULT_CHUNK_CONFIG = {
    # ── Taille des chunks ──
    # 400 tokens est un bon compromis pour la doc technique :
    #   - Assez long pour contenir une explication complète
    #   - Assez court pour être spécifique au retrieval
    #   - Compatible avec la limite de 512 tokens de la plupart des modèles d'embedding
    "chunk_size_tokens": 400,

    # ── Overlap ──
    # 50 tokens de recouvrement entre chunks consécutifs garantissent
    # qu'une information à cheval sur deux chunks ne sera pas perdue.
    "chunk_overlap_tokens": 50,

    # ── Taille minimale ──
    # Les chunks trop courts sont du bruit : un titre seul ou une ligne
    # isolée ne sont pas utiles pour le retrieval.
    "min_chunk_tokens": 30,

    # ── Modèle d'embedding ──
    # all-MiniLM-L6-v2 : modèle léger (80 Mo), rapide, 384 dimensions.
    # Bon compromis qualité/vitesse pour un prototype de stage.
    # Alternative multilingue : paraphrase-multilingual-MiniLM-L12-v2
    "embedding_model": "all-MiniLM-L6-v2",

    # ── Type d'index FAISS ──
    # FlatIP = recherche exacte par produit scalaire (après normalisation L2
    # cela équivaut à la similarité cosinus). Parfait pour < 500k vecteurs.
    "faiss_index_type": "FlatIP",

    # ── Batch size pour l'embedding ──
    # Nombre de chunks encodés simultanément. Ajuster selon la RAM disponible.
    "embedding_batch_size": 64,
}


def load_config_from_benchmark() -> Dict:
    """
    Charge la configuration optimale depuis le rapport de benchmarking (étape 3).

    Si le fichier benchmark_report.json existe et contient des recommandations,
    on les utilise pour paramétrer l'indexation. Sinon, on utilise la
    configuration par défaut.

    Returns
    -------
    dict
        Configuration du chunking et de l'embedding.
    """
    config = DEFAULT_CHUNK_CONFIG.copy()
    report_path = BENCHMARK_DIR / "benchmark_report.json"

    if not report_path.exists():
        logger.info("Pas de rapport de benchmarking trouvé → config par défaut.")
        return config

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        recommendations = report.get("recommendations", {})

        # Appliquer les recommandations du benchmarking
        if "chunking" in recommendations:
            chunk_rec = recommendations["chunking"]
            config["chunk_size_tokens"] = chunk_rec.get("chunk_size", config["chunk_size_tokens"])
            config["chunk_overlap_tokens"] = chunk_rec.get("overlap", config["chunk_overlap_tokens"])
            logger.info(f"Config chunking depuis benchmark : "
                       f"taille={config['chunk_size_tokens']}, overlap={config['chunk_overlap_tokens']}")

        if "embedding_model" in recommendations:
            embed_rec = recommendations["embedding_model"]
            config["embedding_model"] = embed_rec.get("model", config["embedding_model"])
            logger.info(f"Modèle d'embedding depuis benchmark : {config['embedding_model']}")

        print("  ✅ Configuration chargée depuis le benchmarking (étape 3).")

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Erreur lecture du rapport de benchmark : {e}")
        logger.info("Utilisation de la configuration par défaut.")

    return config


# ============================================================
# EMBEDDING ET INDEXATION
# ============================================================

def create_embeddings(chunks: List[Dict], config: Dict) -> np.ndarray:
    """
    Convertit les chunks en vecteurs d'embedding.

    Utilise Sentence-Transformers pour encoder chaque chunk_text
    en un vecteur dense. Les vecteurs sont normalisés L2 pour que
    le produit scalaire FAISS équivaille à la similarité cosinus.

    Parameters
    ----------
    chunks : list[dict]
        Chunks produits par le chunker.
    config : dict
        Configuration (modèle, batch_size).

    Returns
    -------
    np.ndarray
        Matrice d'embeddings (n_chunks × embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    model_name = config["embedding_model"]
    batch_size = config["embedding_batch_size"]

    logger.info(f"Chargement du modèle d'embedding : {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"  Dimension des embeddings : {embedding_dim}")

    texts = [chunk["chunk_text"] for chunk in chunks]
    logger.info(f"Encoding de {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalisation L2 intégrée
    )

    logger.info(f"✅ Embeddings créés : {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """
    Construit un index FAISS à partir des embeddings.

    Utilise IndexFlatIP (produit scalaire) car les embeddings sont
    normalisés L2, donc IP ≡ similarité cosinus. C'est l'index
    le plus simple et le plus fiable — pas d'approximation.

    Parameters
    ----------
    embeddings : np.ndarray
        Matrice d'embeddings normalisés (n × d).

    Returns
    -------
    faiss.Index
        Index FAISS prêt pour la recherche.
    """
    import faiss

    dim = embeddings.shape[1]

    # IndexFlatIP : recherche exacte par produit scalaire
    # Avec des vecteurs normalisés L2, IP = cosine similarity
    index = faiss.IndexFlatIP(dim)

    # Ajouter les vecteurs à l'index
    # FAISS attend du float32
    embeddings_f32 = embeddings.astype(np.float32)
    index.add(embeddings_f32)

    logger.info(f"✅ Index FAISS créé : {index.ntotal} vecteurs, dim={dim}")
    return index


def save_index(index, chunks_metadata: List[Dict]):
    """
    Sauvegarde l'index FAISS et les métadonnées des chunks.

    Les deux fichiers sont nécessaires ensemble :
    - faiss_index.bin contient les vecteurs (sans les textes)
    - chunks_metadata.json contient les textes et métadonnées
      (même ordre que les vecteurs dans l'index)
    """
    import faiss

    # Sauvegarder l'index FAISS
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"Index FAISS sauvegardé : {FAISS_INDEX_PATH}")

    # Sauvegarder les métadonnées (texte + infos de chaque chunk)
    with open(CHUNKS_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Métadonnées sauvegardées : {CHUNKS_META_PATH}")


# ============================================================
# TEST DE RECHERCHE (VALIDATION)
# ============================================================

def test_search(index, chunks: List[Dict], config: Dict, n_tests: int = 5):
    """
    Valide l'index avec des requêtes de test.

    Encode des questions types et vérifie que les résultats sont
    cohérents. C'est un smoke test, pas une évaluation formelle
    (celle-ci a été faite à l'étape 3 avec le benchmarking).
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(config["embedding_model"])

    test_queries = [
        "How to read a file in Python?",
        "What is a random forest classifier?",
        "How to create a retrieval chain in LangChain?",
        "What is cross-validation in scikit-learn?",
        "How to use the os.path module?",
    ]

    print("\n" + "=" * 65)
    print("🔍  VALIDATION PAR RECHERCHE")
    print("=" * 65)

    for query in test_queries[:n_tests]:
        # Encoder la requête
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = query_vec.astype(np.float32)

        # Rechercher les 3 chunks les plus proches
        scores, indices = index.search(query_vec, 3)

        print(f"\n  📝 Question : {query}")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            title = chunk["doc_title"][:40]
            source = chunk["doc_source"]
            preview = chunk["chunk_text"][:80].replace('\n', ' ')
            print(f"     #{rank+1} [{source}] (score={score:.3f}) {title}")
            print(f"         → {preview}...")

    print()


# ============================================================
# RAPPORT
# ============================================================

def generate_report(chunks: List[Dict], embeddings: np.ndarray,
                    config: Dict, elapsed_seconds: float) -> Dict:
    """Génère et affiche le rapport complet de l'étape 4."""
    df = pd.DataFrame(chunks)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "config_source": "benchmark" if (BENCHMARK_DIR / "benchmark_report.json").exists() else "default",
        "elapsed_seconds": round(elapsed_seconds, 1),
        "total_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape),
        "sources": {},
        "token_stats": {
            "mean": round(df["token_count"].mean(), 1),
            "median": round(df["token_count"].median(), 1),
            "min": int(df["token_count"].min()),
            "max": int(df["token_count"].max()),
            "std": round(df["token_count"].std(), 1),
        },
        "chunks_with_code": int(df["has_code"].sum()),
        "output_files": {
            "faiss_index": str(FAISS_INDEX_PATH),
            "chunks_metadata": str(CHUNKS_META_PATH),
        },
    }

    for src in df["doc_source"].unique():
        sub = df[df["doc_source"] == src]
        report["sources"][src] = {
            "chunks": len(sub),
            "documents": sub["doc_filepath"].nunique(),
            "total_tokens": int(sub["token_count"].sum()),
        }

    # Affichage
    print("\n" + "=" * 65)
    print("📊  RAPPORT — ÉTAPE 4 : CHUNKING ET INDEXATION")
    print("=" * 65)

    print(f"\n  ⚙️  Configuration (source: {report['config_source']}) :")
    print(f"     Taille chunk    : {config['chunk_size_tokens']} tokens")
    print(f"     Overlap         : {config['chunk_overlap_tokens']} tokens")
    print(f"     Min chunk       : {config['min_chunk_tokens']} tokens")
    print(f"     Modèle          : {config['embedding_model']}")

    print(f"\n  📦  Résultats :")
    print(f"     Chunks totaux   : {len(chunks):>8,}")
    print(f"     Dimension emb.  : {embeddings.shape[1]:>8}")
    print(f"     Avec code       : {int(df['has_code'].sum()):>8} "
          f"({df['has_code'].mean()*100:.1f}%)")

    print(f"\n  📂  Par source :")
    print(f"     {'Source':<15} {'Chunks':>8} {'Docs':>8} {'Tokens':>10}")
    print(f"     {'─'*45}")
    for src, info in report["sources"].items():
        print(f"     {src:<15} {info['chunks']:>8,} "
              f"{info['documents']:>8,} {info['total_tokens']:>10,}")

    print(f"\n  📏  Taille des chunks (tokens) :")
    print(f"     Moyenne  : {report['token_stats']['mean']:>6}")
    print(f"     Médiane  : {report['token_stats']['median']:>6}")
    print(f"     Min      : {report['token_stats']['min']:>6}")
    print(f"     Max      : {report['token_stats']['max']:>6}")

    print(f"\n  ⏱️   Durée : {elapsed_seconds:.1f} secondes")

    # Sauvegarder le rapport
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  📋  Rapport sauvegardé : {REPORT_PATH}")

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("🔗  ÉTAPE 4 — CHUNKING ET INDEXATION VECTORIELLE")
    print("=" * 65)

    start_time = datetime.now()

    # 0. Vérifier les dépendances
    check_ml_dependencies()

    # 0.5 Créer les répertoires de sortie
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger la configuration (depuis benchmark ou par défaut)
    print("\n" + "─" * 50)
    print("⚙️   1/5 — Chargement de la configuration")
    print("─" * 50)
    config = load_config_from_benchmark()

    # 2. Charger les documents nettoyés (sortie étape 2)
    print("\n" + "─" * 50)
    print("📂  2/5 — Chargement des documents nettoyés")
    print("─" * 50)
    documents = load_cleaned_documents()

    if not documents:
        print("\n❌ Aucun document trouvé dans le dossier de l'étape 2.")
        print(f"   Dossier attendu : {CLEAN_DIR}")
        print(f"   Exécutez d'abord : python main.py --etape 2")
        sys.exit(1)

    # 3. Chunking
    print("\n" + "─" * 50)
    print("✂️   3/5 — Chunking des documents")
    print("─" * 50)
    chunker = DocumentChunker(
        chunk_size=config["chunk_size_tokens"],
        chunk_overlap=config["chunk_overlap_tokens"],
        min_chunk=config["min_chunk_tokens"],
    )

    all_chunks = []
    docs_without_chunks = 0

    for doc in tqdm(documents, desc="✂️  Chunking"):
        doc_chunks = chunker.chunk_document(doc)
        if doc_chunks:
            all_chunks.extend(doc_chunks)
        else:
            docs_without_chunks += 1

    logger.info(f"Chunking terminé : {len(all_chunks)} chunks "
                f"à partir de {len(documents)} documents")
    if docs_without_chunks > 0:
        logger.info(f"  ({docs_without_chunks} documents ignorés — trop courts)")

    if not all_chunks:
        print("\n❌ Aucun chunk produit. Vérifiez le contenu de l'étape 2.")
        sys.exit(1)

    # 4. Embedding
    print("\n" + "─" * 50)
    print("🧠  4/5 — Création des embeddings")
    print("─" * 50)
    embeddings = create_embeddings(all_chunks, config)

    # 5. Indexation FAISS
    print("\n" + "─" * 50)
    print("📦  5/5 — Construction et sauvegarde de l'index FAISS")
    print("─" * 50)
    index = build_faiss_index(embeddings)
    save_index(index, all_chunks)

    # 6. Validation par recherche
    test_search(index, all_chunks, config)

    # 7. Rapport
    elapsed = (datetime.now() - start_time).total_seconds()
    generate_report(all_chunks, embeddings, config, elapsed)

    print("\n" + "=" * 65)
    print("🎉  Étape 4 terminée !")
    print(f"    → {len(all_chunks)} chunks indexés dans FAISS")
    print(f"    → Index : {FAISS_INDEX_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
