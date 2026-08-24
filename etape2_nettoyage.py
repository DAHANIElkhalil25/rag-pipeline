"""
etape2_nettoyage.py — Nettoyage et préparation des données
===========================================================
Étape 2 du pipeline RAG : convertir les documents bruts (RST, MDX, HTML)
en texte propre, séparer code/texte, dédoublonner.

Pipeline par document :
  Document brut JSON (étape 1)
    → Conversion de format (RST / MDX / HTML → texte)
    → Nettoyage transversal (encodage, Unicode, URLs)
    → Séparation code / texte narratif
  Document nettoyé JSON (sortie)

Usage:
    python etape2_nettoyage.py

Entrée:
    rag_project/data/raw/*/json_docs/*.json

Sortie:
    rag_project/data/processed/cleaned/{python,sklearn,langchain}/*.json
    rag_project/data/processed/cleaned/corpus_cleaned_index.csv
"""

import re
import json
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

from config import (
    PYTHON_RAW, SKLEARN_RAW, LANGCHAIN_RAW,
    CLEAN_DIR, METADATA_DIR, CONFIG, logger,
)
from utils import load_raw_documents


# ============================================================
# CONVERTISSEUR RST → TEXTE
# ============================================================

class RSTCleaner:
    """Convertisseur reStructuredText → texte propre."""

    def __init__(self):
        self.content_directives = [
            'note', 'warning', 'tip', 'important', 'hint',
            'attention', 'caution', 'danger', 'error',
            'seealso', 'admonition', 'topic',
        ]
        self.remove_directives = [
            'toctree', 'index', 'only', 'tabularcolumns',
            'currentmodule', 'sectionauthor', 'highlight',
            'default-domain', 'module', 'testsetup',
            'testcleanup', 'doctest', 'productionlist', 'glossary',
        ]
        self.role_pattern = re.compile(
            r':(?:func|class|mod|meth|attr|data|const|exc|obj|ref|doc|term|'
            r'keyword|option|envvar|pep|rfc|command|program|file|samp|'
            r'py:func|py:class|py:mod|py:meth|py:attr|py:data|py:exc|py:obj):'
            r'`([^`]*)`'
        )

    def clean(self, text: str) -> str:
        text = self._handle_code_blocks(text)
        text = self._remove_directives_block(text)
        text = self._convert_content_directives(text)
        text = self._handle_remaining_directives(text)
        text = self._convert_roles(text)
        text = self._clean_inline_markup(text)
        text = self._remove_title_underlines(text)
        text = self._clean_references(text)
        text = self._clean_field_lists(text)
        text = self._normalize_whitespace(text)
        return text

    def _handle_code_blocks(self, text):
        def replace_cb(match):
            lang = match.group(1).strip() if match.group(1) else ""
            code = match.group(2)
            lines = code.split('\n')
            non_empty = [l for l in lines if l.strip()]
            if non_empty:
                mi = min(len(l) - len(l.lstrip()) for l in non_empty)
                lines = [l[mi:] if len(l) > mi else l for l in lines]
            return f"\n[CODE:{lang}]\n{''.join(l + chr(10) for l in lines).strip()}\n[/CODE]\n"

        text = re.sub(
            r'\.\.\s+(?:code-block|sourcecode|code|highlight)::[ ]*([^\n]*)\n'
            r'(?:[ ]*:[^\n]*\n)*\n((?:[ ]{3,}[^\n]*\n?)+)',
            replace_cb, text
        )
        text = re.sub(
            r'::\s*\n\n((?:[ ]{3,}[^\n]*\n?)+)',
            lambda m: f"\n[CODE:]\n{m.group(1).strip()}\n[/CODE]\n", text
        )
        return text

    def _remove_directives_block(self, text):
        for d in self.remove_directives:
            text = re.sub(
                rf'\.\.\s+{re.escape(d)}::.*?(?=\n\S|\n\n\S|\Z)',
                '', text, flags=re.DOTALL
            )
        return text

    def _convert_content_directives(self, text):
        for d in self.content_directives:
            def repl(match, dn=d):
                c = match.group(1) if match.group(1) else ""
                lines = c.split('\n')
                ne = [l for l in lines if l.strip()]
                if ne:
                    mi = min(len(l) - len(l.lstrip()) for l in ne)
                    lines = [l[mi:] if len(l) >= mi else l for l in lines]
                clean = '\n'.join(lines).strip()
                return f"\n{dn.capitalize()} : {clean}\n" if clean else ""
            text = re.sub(
                rf'\.\.\s+{re.escape(d)}::[ ]*[^\n]*\n((?:[ ]+[^\n]*\n?)*)',
                repl, text
            )
        return text

    def _handle_remaining_directives(self, text):
        text = re.sub(r'\.\.\s+[a-zA-Z0-9_-]+::[ ]*[^\n]*\n', '\n', text)
        text = re.sub(r'^\.\.\s+[^\n]*$', '', text, flags=re.MULTILINE)
        return text

    def _convert_roles(self, text):
        def role_repl(m):
            c = m.group(1)
            if '<' in c and '>' in c:
                return c.split('<')[0].strip()
            if c.startswith('~'):
                return c[1:].split('.')[-1]
            return c
        text = self.role_pattern.sub(role_repl, text)
        text = re.sub(r':[a-zA-Z_]+:`([^`]*)`', r'\1', text)
        return text

    def _clean_inline_markup(self, text):
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'``([^`]+)``', r'\1', text)
        text = re.sub(r'\|([^|]+)\|', r'\1', text)
        return text

    def _remove_title_underlines(self, text):
        return re.sub(r'^[=\-~`#"^+*]{3,}\s*$', '', text, flags=re.MULTILINE)

    def _clean_references(self, text):
        text = re.sub(r'`([^`]+)`_+', r'\1', text)
        text = re.sub(r'`([^<]+)\s*<[^>]+>`_+', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        return text

    def _clean_field_lists(self, text):
        text = re.sub(r':param\s+(\w+):', r'Paramètre \1 :', text)
        text = re.sub(r':type\s+(\w+):', r'Type \1 :', text)
        text = re.sub(r':returns?:', 'Retourne :', text)
        text = re.sub(r':rtype:', 'Type de retour :', text)
        text = re.sub(r':raises?\s+(\w+):', r'Lève \1 :', text)
        return text

    def _normalize_whitespace(self, text):
        text = text.replace('\t', '    ')
        text = re.sub(r'[ ]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()


# ============================================================
# CONVERTISSEUR MDX → TEXTE
# ============================================================

class MDXCleaner:
    """Convertisseur MDX (Markdown + composants React) → texte propre."""

    def clean(self, text: str) -> str:
        # Supprimer le frontmatter YAML
        text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)

        # Supprimer les imports React/MDX
        text = re.sub(r'^import\s+.*$', '', text, flags=re.MULTILINE)

        # Convertir les blocs de code Markdown → marqueurs
        def code_repl(m):
            lang = m.group(1) or ""
            code = m.group(2).strip()
            return f"\n[CODE:{lang}]\n{code}\n[/CODE]\n"
        text = re.sub(r'```(\w*)\n(.*?)```', code_repl, text, flags=re.DOTALL)

        # Supprimer les composants React auto-fermants <Component />
        text = re.sub(r'<[A-Z]\w+\s*/>', '', text)

        # Convertir les composants React avec contenu : garder le contenu
        text = re.sub(r'<[A-Z]\w+[^>]*>', '', text)
        text = re.sub(r'</[A-Z]\w+>', '', text)

        # Balises HTML résiduelles
        text = re.sub(r'<(?:br|hr)\s*/?>', '\n', text)
        text = re.sub(r'</?(?:div|span|p|a|em|strong|b|i|u|code|pre)[^>]*>', '', text)

        # Nettoyage Markdown inline
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Images
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', text)

        # Normalisation
        text = re.sub(r'[ ]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()


# ============================================================
# CONVERTISSEUR HTML → TEXTE
# ============================================================

class HTMLCleaner:
    """Convertisseur HTML → texte propre via BeautifulSoup."""

    def clean(self, html_content: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback regex si BS4 absent
            text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()

        import html as html_module
        soup = BeautifulSoup(html_content, 'html.parser')

        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        for el in soup.find_all(class_=re.compile(r'sidebar|menu|nav|breadcrumb')):
            el.decompose()

        # Marquer les blocs de code
        for pre in soup.find_all('pre'):
            code_tag = pre.find('code')
            lang = ""
            if code_tag:
                for cls in code_tag.get('class', []):
                    if cls.startswith('language-'):
                        lang = cls.split('-', 1)[1]
                        break
                pre.string = f"\n[CODE:{lang}]\n{code_tag.get_text()}\n[/CODE]\n"
            else:
                pre.string = f"\n[CODE:]\n{pre.get_text()}\n[/CODE]\n"

        text = soup.get_text(separator='\n', strip=True)
        text = html_module.unescape(text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()


# ============================================================
# NETTOYEUR DE TEXTE UNIFIÉ
# ============================================================

class TextCleaner:
    """Pipeline de nettoyage transversal (encodage, Unicode, URLs, espaces)."""

    def clean(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        text = self._fix_encoding(text)
        text = self._normalize_unicode(text)
        text = self._clean_special_chars(text)
        text = self._clean_urls(text)
        text = self._clean_doc_artifacts(text)
        text = self._normalize_whitespace(text)
        return text

    def _fix_encoding(self, text):
        return ftfy.fix_text(text) if HAS_FTFY else text

    def _normalize_unicode(self, text):
        return unicodedata.normalize('NFC', text)

    def _clean_special_chars(self, text):
        replacements = {
            '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
            '\u2013': '-', '\u2014': ' - ', '\u2026': '...',
            '\u00A0': ' ', '\u200B': '', '\uFEFF': '',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    def _clean_urls(self, text):
        return re.sub(r'https?://\S{50,}', '[URL]', text)

    def _clean_doc_artifacts(self, text):
        text = re.sub(r'(?:New|Changed|Added)\s+in\s+version\s+([\d.]+)',
                       r'(depuis version \1)', text, flags=re.IGNORECASE)
        text = re.sub(r'Deprecated\s+since\s+version\s+([\d.]+)',
                       r'(déprécié depuis version \1)', text, flags=re.IGNORECASE)
        text = re.sub(r'Source code:?\s*\S+\.py\s*', '', text)
        return text

    def _normalize_whitespace(self, text):
        text = text.replace('\t', '    ')
        text = re.sub(r'([^ \n]) +', r'\1 ', text)
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()


# ============================================================
# SÉPARATEUR CODE / TEXTE
# ============================================================

class CodeSeparator:
    """Sépare les blocs [CODE]...[/CODE] du texte narratif."""

    PATTERN = re.compile(r'\[CODE:([^\]]*)\]\n(.*?)\n\[/CODE\]', re.DOTALL)

    def separate(self, text: str) -> Dict:
        segments, code_blocks, text_parts = [], [], []
        last_end = 0

        for m in self.PATTERN.finditer(text):
            before = text[last_end:m.start()].strip()
            if before:
                segments.append({"type": "text", "content": before})
                text_parts.append(before)

            lang = m.group(1).strip()
            code = m.group(2).strip()
            if code:
                cb = {"type": "code", "language": lang, "content": code}
                segments.append(cb)
                code_blocks.append(cb)
            last_end = m.end()

        remaining = text[last_end:].strip()
        if remaining:
            segments.append({"type": "text", "content": remaining})
            text_parts.append(remaining)

        total = len(text)
        code_chars = sum(len(cb["content"]) for cb in code_blocks)

        return {
            "segments": segments,
            "text_only": "\n\n".join(text_parts),
            "code_blocks": code_blocks,
            "code_ratio": round(code_chars / total, 3) if total else 0,
            "n_code_blocks": len(code_blocks),
        }


# ============================================================
# PIPELINE INTÉGRÉ
# ============================================================

# Instanciation des composants
rst_cleaner = RSTCleaner()
mdx_cleaner = MDXCleaner()
html_cleaner = HTMLCleaner()
text_cleaner = TextCleaner()
code_separator = CodeSeparator()


def clean_document(doc: Dict) -> Optional[Dict]:
    """Applique le pipeline complet à un document brut."""
    content = doc.get("content", "")
    metadata = doc.get("metadata", {})
    doc_format = metadata.get("format", "unknown")

    if not content or not content.strip():
        return None

    original_length = len(content)

    # Conversion de format
    if doc_format == "rst":
        converted = rst_cleaner.clean(content)
    elif doc_format in ("mdx", "md"):
        converted = mdx_cleaner.clean(content)
    elif doc_format == "html":
        converted = html_cleaner.clean(content)
    else:
        converted = content

    # Nettoyage transversal
    cleaned = text_cleaner.clean(converted)
    if not cleaned or len(cleaned.strip()) < 50:
        return None

    # Séparation code/texte
    separated = code_separator.separate(cleaned)

    source_key = str(metadata.get("source", "unknown")).lower()
    content_sha256 = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    path_hint = re.sub(r"[^a-z0-9]+", "_", str(metadata.get("filepath", metadata.get("title", "document"))).lower()).strip("_")[:64] or "document"
    document_id = f"{source_key}_{path_hint}_{content_sha256[:12]}"

    return {
        "cleaned_content": cleaned,
        "text_only": separated["text_only"],
        "code_blocks": separated["code_blocks"],
        "segments": separated["segments"],
        "metadata": {
            **metadata,
            "cleaned_at": datetime.now().isoformat(),
            "original_char_count": original_length,
            "cleaned_char_count": len(cleaned),
            "text_only_char_count": len(separated["text_only"]),
            "reduction_ratio": round(1 - len(cleaned) / original_length, 3) if original_length else 0,
            "code_ratio": separated["code_ratio"],
            "n_code_blocks": separated["n_code_blocks"],
            "cleaned_word_count": len(cleaned.split()),
            "text_only_word_count": len(separated["text_only"].split()),
            "content_hash": hashlib.md5(cleaned.encode()).hexdigest(),
            "content_sha256": content_sha256,
            "document_id": document_id,
            "source_version": metadata.get("source_version", CONFIG.get(source_key, {}).get("version", "unknown")),
        }
    }


def process_source(source_name: str, json_dir: Path, output_dir: Path) -> Tuple[List[Dict], Dict]:
    """Nettoie tous les documents d'une source."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Une réexécution doit repartir d'un corpus nettoyé cohérent et ne pas
    # conserver des fichiers périmés d'une collecte précédente.
    for old_file in output_dir.glob("clean_*.json"):
        old_file.unlink()
    documents = load_raw_documents(json_dir)
    meta_list, stats = [], {"input": len(documents), "output": 0, "filtered": 0, "errors": 0}

    logger.info(f"Nettoyage de {len(documents)} documents {source_name}...")

    for doc in tqdm(documents, desc=f"🧹 {source_name}"):
        try:
            cleaned = clean_document(doc)
            if cleaned is None:
                stats["filtered"] += 1
                continue

            fp = doc.get("metadata", {}).get("filepath", "unknown")
            safe = re.sub(r'[^\w\-.]', '_', fp)[:100]
            fh = hashlib.md5(fp.encode()).hexdigest()[:8]
            out_path = output_dir / f"clean_{safe}_{fh}.json"
            cleaned["metadata"]["cleaned_file"] = str(out_path)

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned, f, ensure_ascii=False, indent=2)

            meta_list.append(cleaned["metadata"])
            stats["output"] += 1
        except (UnicodeDecodeError, IOError, ValueError, KeyError) as e:
            logger.exception(f"Erreur nettoyage pour {doc.get('metadata', {}).get('filepath', 'unknown')}")
            stats["errors"] += 1

    print(f"  {source_name}: {stats['input']} → {stats['output']} "
          f"(filtrés: {stats['filtered']}, erreurs: {stats['errors']})")
    return meta_list, stats


def deduplicate(metadata_list: List[Dict]) -> Tuple[List[Dict], int, int]:
    """Dédoublonnage par hash exact + quasi-doublons par titre."""
    # Passe 1 : doublons exacts
    seen_hashes, unique = {}, []
    exact_dups = 0
    for m in metadata_list:
        h = m.get("content_hash", "")
        if h in seen_hashes:
            exact_dups += 1
        else:
            seen_hashes[h] = True
            unique.append(m)

    # Passe 2 : quasi-doublons
    final, seen_titles = [], []
    near_dups = 0
    for m in unique:
        title, src = m.get("title", ""), m.get("source", "")
        is_dup = False
        for st, ss in seen_titles:
            if src == ss and SequenceMatcher(None, title.lower(), st.lower()).ratio() > 0.9:
                is_dup = True
                break
        if is_dup:
            near_dups += 1
        else:
            final.append(m)
            seen_titles.append((title, src))

    return final, exact_dups, near_dups


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("🧹  ÉTAPE 2 — NETTOYAGE ET PRÉPARATION")
    print("=" * 65)

    all_meta = []

    # Traitement par source
    sources = [
        ("PYTHON",     PYTHON_RAW / "json_docs",    CLEAN_DIR / "python"),
        ("SKLEARN",    SKLEARN_RAW / "json_docs",   CLEAN_DIR / "sklearn"),
        ("LANGCHAIN",  LANGCHAIN_RAW / "json_docs", CLEAN_DIR / "langchain"),
    ]

    stats_all = {}
    for name, in_dir, out_dir in sources:
        meta, stats = process_source(name, in_dir, out_dir)
        all_meta.extend(meta)
        stats_all[name] = stats

    # Dédoublonnage
    unique_meta, exact_dups, near_dups = deduplicate(all_meta)

    # Le rapport et le corpus indexé doivent désigner exactement le même jeu
    # de documents. Les quasi-doublons sont donc supprimés physiquement.
    retained_files = {meta.get("cleaned_file") for meta in unique_meta}
    removed_files = 0
    for meta in all_meta:
        candidate = meta.get("cleaned_file")
        if candidate and candidate not in retained_files:
            path = Path(candidate)
            if path.exists():
                path.unlink()
                removed_files += 1

    # Export
    df = pd.DataFrame(unique_meta)
    index_path = CLEAN_DIR / "corpus_cleaned_index.csv"
    df.to_csv(index_path, index=False, encoding='utf-8')
    logger.info(f"Dédoublonnage appliqué au corpus : {removed_files} fichiers supprimés.")

    # Rapport
    print("\n" + "=" * 65)
    print("📊  RAPPORT DE NETTOYAGE — ÉTAPE 2")
    print("=" * 65)
    print(f"\n  Documents en entrée       : {sum(s['input'] for s in stats_all.values()):>6}")
    print(f"  Documents nettoyés        : {sum(s['output'] for s in stats_all.values()):>6}")
    print(f"  Filtrés (vides/courts)    : {sum(s['filtered'] for s in stats_all.values()):>6}")
    print(f"  Doublons exacts supprimés : {exact_dups:>6}")
    print(f"  Quasi-doublons supprimés  : {near_dups:>6}")
    print(f"  Documents finaux uniques  : {len(unique_meta):>6}")

    if len(df) > 0:
        print(f"\n📂  Par source :")
        print(df.groupby('source').agg(
            docs=('source', 'count'),
            mots=('cleaned_word_count', 'sum'),
        ).to_string())

    print(f"\n📋  Index sauvegardé : {index_path}")
    print(f"\n🎉  Étape 2 terminée !")
    print(f"    → {len(unique_meta)} documents prêts pour l'étape 3.\n")


if __name__ == "__main__":
    main()
