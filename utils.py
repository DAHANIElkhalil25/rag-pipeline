"""
utils.py — Fonctions utilitaires partagées
===========================================
Fonctions transversales utilisées par les étapes 1 et 2 :
- Création d'enregistrements documents standardisés
- Sauvegarde / chargement JSON
- Extraction de titres (RST, MDX)
- Classification par section
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


# ============================================================
# CRÉATION DE DOCUMENTS STANDARDISÉS
# ============================================================

def create_document_record(
    content: str,
    source: str,
    filepath: str,
    title: str,
    section: str,
    version: str,
    doc_format: str,
    url: Optional[str] = None
) -> Dict:
    """
    Crée un enregistrement standardisé pour chaque document collecté.

    Cette structure uniforme permet de traiter tous les documents
    de la même manière dans les étapes suivantes, quelle que soit
    leur source d'origine.

    Parameters
    ----------
    content : str
        Contenu brut du document (avec balisage d'origine).
    source : str
        Identifiant de la source ('python', 'sklearn', 'langchain').
    filepath : str
        Chemin relatif du fichier dans le dépôt ou URL.
    title : str
        Titre du document.
    section : str
        Catégorie (tutorial, api_reference, guide, etc.).
    version : str
        Version de la documentation.
    doc_format : str
        Format d'origine ('rst', 'md', 'mdx', 'html').
    url : str, optional
        URL d'origine.

    Returns
    -------
    dict
        Enregistrement standardisé.
    """
    return {
        "content": content,
        "metadata": {
            "source": source,
            "filepath": filepath,
            "title": title,
            "section": section,
            "version": version,
            "format": doc_format,
            "url": url,
            "collected_at": datetime.now().isoformat(),
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "char_count": len(content),
            "word_count": len(content.split()),
        }
    }


def save_document(record: Dict, output_dir: Path) -> Path:
    """Sauvegarde un enregistrement document en JSON."""
    filepath = record["metadata"]["filepath"]
    safe_name = re.sub(r'[^\w\-.]', '_', filepath)[:100]
    file_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
    output_path = output_dir / f"{safe_name}_{file_hash}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return output_path


def load_raw_documents(json_dir: Path) -> List[Dict]:
    """Charge tous les fichiers JSON d'un dossier de données brutes."""
    documents = []
    if not json_dir.exists():
        return documents

    for filepath in sorted(json_dir.glob("*.json")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            doc["_json_path"] = str(filepath)
            documents.append(doc)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            import logging
            logging.getLogger("rag_pipeline").warning(f"Erreur lecture {filepath.name}: {e}")

    return documents


# ============================================================
# EXTRACTION DE TITRES
# ============================================================

def extract_rst_title(content: str) -> str:
    """
    Extrait le titre d'un fichier reStructuredText.

    Convention RST : le titre est la première ligne suivie d'une ligne
    de caractères de soulignement (===, ---, ~~~, etc.).
    """
    lines = content.strip().split('\n')
    for i, line in enumerate(lines):
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and len(set(next_line)) == 1 and next_line[0] in '=-~^"#*+':
                if len(next_line) >= len(line.strip()) * 0.5:
                    title = line.strip()
                    if title and not all(c in '=-~^"#*+' for c in title):
                        return title
    return "Sans titre"


def extract_mdx_title(content: str) -> str:
    """
    Extrait le titre d'un fichier .mdx / .md.

    Cherche dans l'ordre :
    1. Le champ 'title' dans le frontmatter YAML
    2. Le premier titre Markdown (# Titre)
    """
    # Frontmatter YAML
    fm_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if fm_match:
        frontmatter = fm_match.group(1)
        title_match = re.search(
            r'^title:\s*["\']?(.+?)["\']?\s*$', frontmatter, re.MULTILINE
        )
        if title_match:
            return title_match.group(1).strip()

    # Premier titre Markdown
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    return "Sans titre"


# ============================================================
# CLASSIFICATION PAR SECTION
# ============================================================

def classify_section(filepath: str, source: str) -> str:
    """
    Classifie un document dans une catégorie sémantique.

    Cette classification permet le filtrage lors du retrieval.
    """
    fp_lower = filepath.lower()

    # Patterns communs
    if any(kw in fp_lower for kw in ['tutorial', 'quickstart', 'getting_started', 'getting-started']):
        return "tutorial"
    if any(kw in fp_lower for kw in ['api', 'reference', 'library']):
        return "api_reference"
    if any(kw in fp_lower for kw in ['howto', 'how_to', 'how-to', 'guide', 'user_guide']):
        return "guide"
    if any(kw in fp_lower for kw in ['example', 'auto_examples', 'gallery']):
        return "example"
    if 'faq' in fp_lower:
        return "faq"

    # Patterns spécifiques par source
    if source == "python":
        if "library/" in fp_lower:
            return "api_reference"
        if "reference/" in fp_lower:
            return "guide"

    elif source == "sklearn":
        if "modules/" in fp_lower:
            return "guide"

    elif source == "langchain":
        if "integrations/" in fp_lower:
            return "integrations"
        if "langgraph/" in fp_lower:
            return "langgraph"
        if any(kw in fp_lower for kw in ['retrieval', 'rag', 'vector']):
            return "retrieval"
        if any(kw in fp_lower for kw in ['model', 'llm', 'chat']):
            return "models"
        if any(kw in fp_lower for kw in ['tool', 'agent']):
            return "agents"
        return "langchain_core"

    return "other"
