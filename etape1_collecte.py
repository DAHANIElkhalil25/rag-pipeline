"""
etape1_collecte.py — Collecte et acquisition des données
=========================================================
Étape 1 du pipeline RAG : télécharger les documentations techniques
depuis leurs dépôts GitHub officiels et les stocker en JSON standardisé.

Sources :
  - Python 3.13     → clone cpython, dossier Doc/ (fichiers .rst)
  - Scikit-learn 1.5 → clone scikit-learn, dossier doc/ (fichiers .rst)
  - LangChain       → clone langchain-ai/docs, dossier src/ (fichiers .mdx)

Usage:
    python etape1_collecte.py

Sortie:
    rag_project/data/raw/{python,sklearn,langchain}_docs/json_docs/*.json
    rag_project/data/metadata/corpus_index.csv
"""

import re
import shutil
from pathlib import Path

import git
import pandas as pd
from tqdm import tqdm

from config import (
    CONFIG, PROJECT_ROOT, RAW_DIR,
    PYTHON_RAW, SKLEARN_RAW, LANGCHAIN_RAW,
    METADATA_DIR, logger, init_directories,
)
from utils import (
    create_document_record, save_document,
    extract_rst_title, extract_mdx_title, classify_section,
)


# ============================================================
# 1. COLLECTE PYTHON (clone cpython → Doc/*.rst)
# ============================================================

def clone_cpython_docs() -> Path:
    """Clone partiel du dépôt CPython (dossier Doc/ uniquement)."""
    config = CONFIG["python"]
    clone_dir = PYTHON_RAW / "cpython_clone"
    doc_dir = clone_dir / config["doc_subdir"]

    if doc_dir.exists() and any(doc_dir.rglob("*.rst")):
        n = len(list(doc_dir.rglob("*.rst")))
        logger.info(f"Clone CPython existant ({n} fichiers .rst)")
        return doc_dir

    logger.info("Clone partiel de CPython (Doc/ uniquement)...")
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    repo = git.Repo.clone_from(
        config["repo_url"], clone_dir,
        depth=1, no_checkout=True, branch=config["branch"]
    )
    repo.git.sparse_checkout("init", "--cone")
    repo.git.sparse_checkout("set", config["doc_subdir"])
    repo.git.checkout()

    n = len(list(doc_dir.rglob("*.rst")))
    logger.info(f"✅ CPython cloné : {n} fichiers .rst")
    return doc_dir


def collect_rst_docs(source_name: str, doc_dir: Path, config: dict, output_dir: Path) -> list:
    """
    Fonction générique pour extraire et indexer les fichiers .rst.

    Factorise la logique commune entre Python et Scikit-learn.

    Parameters
    ----------
    source_name : str
        Nom de la source ('python', 'sklearn').
    doc_dir : Path
        Répertoire contenant les fichiers .rst.
    config : dict
        Configuration de la source (version, exclude_dirs, url_base).
    output_dir : Path
        Répertoire de sortie pour les fichiers JSON.

    Returns
    -------
    list
        Liste des métadonnées des documents collectés.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude = set(config.get("exclude_dirs", []))
    records = []

    rst_files = [
        f for f in doc_dir.rglob("*.rst")
        if not any(ex in f.parts for ex in exclude)
    ]
    logger.info(f"Traitement de {len(rst_files)} fichiers .rst {source_name}...")

    for fp in tqdm(rst_files, desc=f"📄 {source_name}"):
        try:
            content = fp.read_text(encoding='utf-8', errors='replace')
            if len(content.strip()) < 100:
                continue

            rel_path = str(fp.relative_to(doc_dir))
            url_base = config.get("url_base", "")
            record = create_document_record(
                content=content,
                source=source_name.lower(),
                filepath=rel_path,
                title=extract_rst_title(content),
                section=classify_section(rel_path, source_name.lower()),
                version=config["version"],
                doc_format="rst",
                url=f"{url_base}{rel_path.replace('.rst', '.html')}" if url_base else None,
            )
            save_document(record, output_dir)
            records.append(record["metadata"])
        except (UnicodeDecodeError, IOError, OSError) as e:
            logger.exception(f"Erreur lors du traitement de {fp.name}")

    logger.info(f"✅ {len(records)} documents {source_name} collectés.")
    return records


def collect_python_docs(doc_dir: Path) -> list:
    """Extrait et indexe les fichiers .rst de la doc Python."""
    config = CONFIG["python"]
    config_with_url = {**config, "url_base": "https://docs.python.org/3/"}
    return collect_rst_docs("python", doc_dir, config_with_url, PYTHON_RAW / "json_docs")


# ============================================================
# 2. COLLECTE SCIKIT-LEARN (clone scikit-learn → doc/*.rst)
# ============================================================

def clone_sklearn_docs() -> Path:
    """Clone partiel du dépôt Scikit-learn (dossier doc/ uniquement)."""
    config = CONFIG["sklearn"]
    clone_dir = SKLEARN_RAW / "sklearn_clone"
    doc_dir = clone_dir / config["doc_subdir"]

    if doc_dir.exists() and any(doc_dir.rglob("*.rst")):
        n = len(list(doc_dir.rglob("*.rst")))
        logger.info(f"Clone Scikit-learn existant ({n} fichiers .rst)")
        return doc_dir

    logger.info("Clone partiel de Scikit-learn (doc/ uniquement)...")
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    repo = git.Repo.clone_from(
        config["repo_url"], clone_dir,
        depth=1, no_checkout=True, branch=config["branch"]
    )
    repo.git.sparse_checkout("init", "--cone")
    repo.git.sparse_checkout("set", config["doc_subdir"])
    repo.git.checkout()

    n = len(list(doc_dir.rglob("*.rst")))
    logger.info(f"✅ Scikit-learn cloné : {n} fichiers .rst")
    return doc_dir


def collect_sklearn_docs(doc_dir: Path) -> list:
    """Extrait et indexe les fichiers .rst de la doc Scikit-learn."""
    config = CONFIG["sklearn"]
    config_with_url = {**config, "url_base": "https://scikit-learn.org/stable/"}
    return collect_rst_docs("sklearn", doc_dir, config_with_url, SKLEARN_RAW / "json_docs")


# ============================================================
# 3. COLLECTE LANGCHAIN (clone langchain-ai/docs → src/*.mdx)
# ============================================================

def clone_langchain_docs() -> Path:
    """Clone du dépôt langchain-ai/docs (documentation unifiée)."""
    config = CONFIG["langchain"]
    clone_dir = LANGCHAIN_RAW / "langchain_docs_clone"

    if clone_dir.exists() and (clone_dir / ".git").exists():
        mdx_count = len(list(clone_dir.rglob("*.mdx")))
        if mdx_count > 50:
            logger.info(f"Clone LangChain existant ({mdx_count} fichiers .mdx)")
            return clone_dir
        logger.warning("Clone incomplet, re-clonage...")
        shutil.rmtree(clone_dir)

    logger.info("Clone de langchain-ai/docs (--depth 1)...")
    git.Repo.clone_from(
        config["repo_url"], clone_dir,
        depth=1, branch=config["branch"]
    )

    mdx_count = len(list(clone_dir.rglob("*.mdx")))
    md_count = len(list(clone_dir.rglob("*.md")))
    logger.info(f"✅ LangChain cloné : {mdx_count} .mdx, {md_count} .md")
    return clone_dir


def collect_langchain_docs(repo_dir: Path) -> list:
    """Extrait et indexe les fichiers .mdx/.md du dépôt LangChain."""
    config = CONFIG["langchain"]
    output_dir = LANGCHAIN_RAW / "json_docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nettoyer les anciens fichiers (d'un éventuel scraping raté)
    for old in output_dir.glob("*.json"):
        old.unlink()

    exclude = set(config["exclude_patterns"])
    all_files = []

    # Chercher dans les dossiers cibles
    for target_dir in config["target_dirs"]:
        target_path = repo_dir / target_dir
        if not target_path.exists():
            # Essayer sans le préfixe src/
            alt = repo_dir / target_dir.replace("src/", "")
            if alt.exists():
                target_path = alt
            else:
                logger.warning(f"Dossier non trouvé : {target_dir}")
                continue

        for ext in config["extensions"]:
            found = [
                f for f in target_path.rglob(f"*{ext}")
                if not any(ex in str(f) for ex in exclude)
            ]
            all_files.extend(found)

    # Fallback : recherche élargie si rien trouvé
    if not all_files:
        logger.warning("Aucun fichier dans les cibles. Recherche élargie...")
        for ext in config["extensions"]:
            all_files.extend([
                f for f in repo_dir.rglob(f"*{ext}")
                if not any(ex in str(f) for ex in exclude)
                and "node_modules" not in str(f)
            ])

    all_files = list(set(all_files))
    records = []
    logger.info(f"Traitement de {len(all_files)} fichiers LangChain...")

    for fp in tqdm(all_files, desc="🔗 LangChain"):
        try:
            content = fp.read_text(encoding='utf-8', errors='replace')
            if len(content.strip()) < 100:
                continue
            if fp.name in ('docs.json', 'mint.json', '_meta.json'):
                continue

            rel_path = str(fp.relative_to(repo_dir))

            # Construire l'URL docs.langchain.com
            url_path = rel_path
            for prefix in ['src/', 'build/']:
                if url_path.startswith(prefix):
                    url_path = url_path[len(prefix):]
            url_path = re.sub(r'\.(mdx|md)$', '', url_path)
            url = f"https://docs.langchain.com/{url_path}"

            record = create_document_record(
                content=content,
                source="langchain",
                filepath=rel_path,
                title=extract_mdx_title(content),
                section=classify_section(rel_path, "langchain"),
                version=config["version"],
                doc_format="mdx" if fp.suffix == ".mdx" else "md",
                url=url,
            )
            save_document(record, output_dir)
            records.append(record["metadata"])
        except (UnicodeDecodeError, IOError, OSError) as e:
            logger.exception(f"Erreur lors du traitement de {fp.name}")

    logger.info(f"✅ {len(records)} documents LangChain collectés.")
    return records


# ============================================================
# 4. CONSOLIDATION ET EXPORT
# ============================================================

def consolidate(python_meta, sklearn_meta, langchain_meta):
    """Fusionne les métadonnées et génère l'index + le rapport."""
    all_meta = python_meta + sklearn_meta + langchain_meta
    df = pd.DataFrame(all_meta)

    # Sauvegarde de l'index
    index_path = METADATA_DIR / "corpus_index.csv"
    df.to_csv(index_path, index=False, encoding='utf-8')

    # Rapport
    print("\n" + "=" * 65)
    print("📊  RAPPORT DE COLLECTE — ÉTAPE 1")
    print("=" * 65)
    print(f"\n{'Source':<20} {'Documents':>10} {'Mots':>12}")
    print("-" * 45)
    for src in ['python', 'sklearn', 'langchain']:
        sub = df[df['source'] == src]
        print(f"  {src:<18} {len(sub):>10,} {sub['word_count'].sum():>12,}")
    print("-" * 45)
    print(f"  {'TOTAL':<18} {len(df):>10,} {df['word_count'].sum():>12,}")

    print(f"\n📁  Répartition par section :")
    print(df.groupby(['source', 'section']).size()
          .unstack(fill_value=0).to_string())

    print(f"\n📋  Index sauvegardé : {index_path}")
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("🚀  ÉTAPE 1 — COLLECTE DES DONNÉES")
    print("=" * 65)

    # Initialiser l'arborescence
    init_directories()

    # 1. Python
    print("\n" + "─" * 50)
    print("📦  1/3 — Documentation Python")
    print("─" * 50)
    python_dir = clone_cpython_docs()
    python_meta = collect_python_docs(python_dir)

    # 2. Scikit-learn
    print("\n" + "─" * 50)
    print("📦  2/3 — Documentation Scikit-learn")
    print("─" * 50)
    sklearn_dir = clone_sklearn_docs()
    sklearn_meta = collect_sklearn_docs(sklearn_dir)

    # 3. LangChain
    print("\n" + "─" * 50)
    print("📦  3/3 — Documentation LangChain")
    print("─" * 50)
    langchain_dir = clone_langchain_docs()
    langchain_meta = collect_langchain_docs(langchain_dir)

    # 4. Consolidation
    df = consolidate(python_meta, sklearn_meta, langchain_meta)

    print("\n🎉  Étape 1 terminée !")
    print(f"    → {len(df)} documents prêts pour l'étape 2.\n")


if __name__ == "__main__":
    main()
