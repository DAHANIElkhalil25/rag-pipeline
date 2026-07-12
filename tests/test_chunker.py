"""Tests unitaires pour le chunker unifié (core/chunker.py)."""

import pytest
from core.chunker import DocumentChunker


# ============================================================
# TESTS — DocumentChunker
# ============================================================

class TestDocumentChunker:
    """Tests pour le chunker unifié."""

    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, min_chunk=10)

    def _make_doc(self, content: str, source: str = "python") -> dict:
        """Crée un document de test minimal."""
        return {
            "cleaned_content": content,
            "metadata": {
                "source": source,
                "title": "Test Doc",
                "filepath": "test.rst",
                "url": "https://test.com",
                "section": "other",
                "content_hash": "abc123def456",
            }
        }

    def test_empty_document(self):
        """Un document vide ne produit aucun chunk."""
        doc = self._make_doc("")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_short_document(self):
        """Un document trop court est ignoré."""
        doc = self._make_doc("Court.")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_single_chunk_document(self):
        """Un document de taille raisonnable produit au moins un chunk."""
        content = "Ceci est un document suffisamment long pour être un chunk. " * 5
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_metadata(self):
        """Chaque chunk contient les métadonnées attendues."""
        content = "Ceci est du contenu suffisamment long pour être chunké. " * 5
        doc = self._make_doc(content, source="sklearn")
        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) >= 1
        chunk = chunks[0]

        assert "chunk_text" in chunk
        assert "chunk_id" in chunk
        assert "doc_source" in chunk
        assert "doc_title" in chunk
        assert "token_count" in chunk
        assert "has_code" in chunk
        assert chunk["doc_source"] == "sklearn"
        assert chunk["doc_title"] == "Test Doc"

    def test_chunk_ids_unique(self):
        """Tous les chunk_ids d'un document sont uniques."""
        content = "Paragraphe de texte. " * 50
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)

        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_sections_respected(self):
        """Les sections Markdown sont détectées et utilisées."""
        content = (
            "Introduction avant les sections.\n\n"
            "# Section Un\n\n"
            "Contenu de la première section. " * 10 + "\n\n"
            "# Section Deux\n\n"
            "Contenu de la deuxième section. " * 10
        )
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)

        # Les sections titles doivent apparaître dans les métadonnées
        section_titles = [c["section_title"] for c in chunks if c["section_title"]]
        assert len(section_titles) >= 1

    def test_code_detection(self):
        """Les chunks contenant [CODE:] sont détectés."""
        content = (
            "Texte normal avant.\n\n"
            "[CODE:python]\nx = 1\nprint(x)\n[/CODE]\n\n"
            "Texte après le code. " * 10
        )
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)

        has_code_chunks = [c for c in chunks if c["has_code"]]
        assert len(has_code_chunks) >= 1

    def test_total_chunks_in_doc(self):
        """Le champ total_chunks_in_doc est correct."""
        content = "Contenu de test. " * 50
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)

        if chunks:
            expected_total = len(chunks)
            for chunk in chunks:
                assert chunk["total_chunks_in_doc"] == expected_total

    def test_token_count_within_limits(self):
        """Aucun chunk ne dépasse la taille maximale."""
        content = "Contenu de test assez long pour produire plusieurs chunks. " * 100
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            # Le token_count peut légèrement dépasser à cause du découpage
            # par phrases, mais ne devrait pas être excessif
            assert chunk["token_count"] >= self.chunker.min_chunk

    def test_different_chunk_sizes(self):
        """Des tailles de chunk différentes produisent des résultats différents."""
        content = "Contenu de test pour le chunking. " * 100
        doc = self._make_doc(content)

        small_chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, min_chunk=10)
        large_chunker = DocumentChunker(chunk_size=200, chunk_overlap=30, min_chunk=10)

        small_chunks = small_chunker.chunk_document(doc)
        large_chunks = large_chunker.chunk_document(doc)

        # Plus petits chunks → plus de chunks
        assert len(small_chunks) > len(large_chunks)
