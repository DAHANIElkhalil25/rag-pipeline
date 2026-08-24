"""
core/chunker.py — Chunker unifié du pipeline RAG
==================================================
Découpe les documents nettoyés en chunks optimisés pour le RAG.

Stratégie hybride :
1. D'abord, tenter de découper par SECTIONS (titres Markdown #, ##, ###).
   La doc technique est naturellement organisée en sections thématiques ;
   les respecter produit des chunks plus cohérents.
2. Si une section est trop longue, la sous-découper par PARAGRAPHES
   (double saut de ligne).
3. Si un paragraphe est encore trop long, le découper par TAILLE FIXE
   avec overlap.
4. Toujours ignorer les chunks trop courts (< min_tokens).

Cette cascade section → paragraphe → taille fixe est plus intelligente
qu'un découpage purement mécanique car elle respecte la structure
sémantique du document.

Réf. [7] Barnett et al. (2024) identifient le chunking comme le
premier point de défaillance des systèmes RAG ("Missing Content"
et "Wrong Granularity").

Réf. [6] Gao et al. (2024) recommandent des chunks de 256-512 tokens
pour la documentation technique, avec un overlap de 10-15%.

Utilisé par : etape3_benchmarking.py, etape4_indexation.py
"""

import re
import hashlib
from typing import Dict, List, Tuple

from core.tokenizer import TokenCounter


def _identifier(value: str, fallback: str) -> str:
    """Construit un identifiant lisible et stable à partir d'un texte."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return normalized[:64] or fallback


class DocumentChunker:
    """
    Chunker paramétrable pour le benchmarking et l'indexation.

    Parameters
    ----------
    chunk_size : int
        Taille maximale d'un chunk en tokens.
    chunk_overlap : int
        Nombre de tokens de recouvrement entre chunks consécutifs.
    min_chunk : int
        Taille minimale d'un chunk en tokens (en dessous, il est ignoré).
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, min_chunk: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk = min_chunk
        self.token_counter = TokenCounter()
        self.section_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    def chunk_document(self, doc: Dict) -> List[Dict]:
        """
        Découpe un document nettoyé en chunks avec métadonnées.

        Parameters
        ----------
        doc : dict
            Document nettoyé (sortie de l'étape 2).
            Clés attendues : cleaned_content, text_only, metadata, segments

        Returns
        -------
        list[dict]
            Liste de chunks avec chunk_text, chunk_id, doc_source,
            doc_title, doc_filepath, doc_url, section_title,
            chunk_index, token_count, has_code.
        """
        content = doc.get("cleaned_content", "")
        metadata = doc.get("metadata", {})

        if not content or self.token_counter.count(content) < self.min_chunk:
            return []

        # Phase 1 : Découpage par sections
        sections = self._split_by_sections(content)

        # Phase 2 : Pour chaque section, découper si trop longue
        raw_chunks = []
        for section_title, section_text in sections:
            section_tokens = self.token_counter.count(section_text)

            if section_tokens <= self.chunk_size:
                if section_tokens >= self.min_chunk:
                    raw_chunks.append((section_title, section_text))
            else:
                sub_chunks = self._split_by_paragraphs(section_text)
                for sub_text in sub_chunks:
                    sub_tokens = self.token_counter.count(sub_text)
                    if sub_tokens <= self.chunk_size:
                        if sub_tokens >= self.min_chunk:
                            raw_chunks.append((section_title, sub_text))
                    else:
                        for piece in self._split_by_size(sub_text):
                            raw_chunks.append((section_title, piece))

        # Phase 3 : Construire les chunks avec métadonnées complètes
        content_sha256 = metadata.get("content_sha256") or hashlib.sha256(content.encode("utf-8")).hexdigest()
        source = _identifier(metadata.get("source", "unknown"), "unknown")
        path_hint = _identifier(metadata.get("filepath", metadata.get("title", "document")), "document")
        document_id = metadata.get("document_id") or f"{source}_{path_hint}_{content_sha256[:12]}"

        chunks = []
        for idx, (section_title, chunk_text) in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            token_count = self.token_counter.count(chunk_text)
            if token_count < self.min_chunk:
                continue

            chunks.append({
                "chunk_text": chunk_text,
                "chunk_id": f"{document_id}_chunk_{idx:04d}",
                "document_id": document_id,
                "doc_source": metadata.get("source", "unknown"),
                "source_version": metadata.get("source_version", "unknown"),
                "doc_title": metadata.get("title", "Sans titre"),
                "doc_filepath": metadata.get("filepath", ""),
                "doc_url": metadata.get("url", ""),
                "doc_section": metadata.get("section", "other"),
                "section_title": section_title or "",
                "chunk_index": idx,
                "total_chunks_in_doc": -1,
                "content_sha256": content_sha256,
                "chunk_content_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "token_count": token_count,
                "has_code": bool(re.search(r'\[CODE:', chunk_text)),
            })

        for chunk in chunks:
            chunk["total_chunks_in_doc"] = len(chunks)

        return chunks

    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Découpe le texte par titres de section.

        Retourne une liste de (titre_section, contenu_section).
        Si pas de titre trouvé, retourne le texte entier avec titre vide.
        """
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            return [("", text)]

        sections = []

        pre_text = text[:matches[0].start()].strip()
        if pre_text and self.token_counter.count(pre_text) >= self.min_chunk:
            sections.append(("", pre_text))

        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text:
                sections.append((title, section_text))

        return sections if sections else [("", text)]

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Découpe par paragraphes, puis fusionne les paragraphes courts.

        Deux paragraphes courts consécutifs sont fusionnés s'ils tiennent
        ensemble dans un chunk — ça évite de produire des micro-chunks
        qui perdent leur contexte.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text]

        merged = []
        buffer = ""

        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            candidate_tokens = self.token_counter.count(candidate)

            if candidate_tokens <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer)
                if self.token_counter.count(para) <= self.chunk_size:
                    buffer = para
                else:
                    for piece in self._split_by_size(para):
                        merged.append(piece)
                    buffer = ""

        if buffer:
            merged.append(buffer)

        return merged

    def _split_by_size(self, text: str) -> List[str]:
        """
        Découpage mécanique par taille fixe avec overlap.

        Dernier recours quand le texte ne contient ni sections ni paragraphes.
        Découpe par phrases pour éviter de couper au milieu d'un mot.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.token_counter.count(sentence)

            if current_tokens + sent_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if self.token_counter.count(chunk_text) >= self.min_chunk:
                        chunks.append(chunk_text)

                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_tok = self.token_counter.count(s)
                    if overlap_tokens + s_tok <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tok
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sent_tokens

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.token_counter.count(chunk_text) >= self.min_chunk:
                chunks.append(chunk_text)

        return chunks
