"""Tests unitaires pour les cleaners (etape2_nettoyage.py)."""

import pytest
from etape2_nettoyage import RSTCleaner, MDXCleaner, TextCleaner


# ============================================================
# TESTS — RSTCleaner
# ============================================================

class TestRSTCleaner:
    """Tests pour le nettoyeur RST."""

    def setup_method(self):
        self.cleaner = RSTCleaner()

    def test_removes_title_underlines(self):
        text = "Mon Titre\n=========\n\nContenu."
        result = self.cleaner.clean(text)
        assert "=========" not in result
        assert "Mon Titre" in result

    def test_converts_roles(self):
        text = "Utilisez :func:`os.listdir` pour lister."
        result = self.cleaner.clean(text)
        assert ":func:" not in result
        assert "os.listdir" in result

    def test_removes_toctree(self):
        text = "Avant\n\n.. toctree::\n   :maxdepth: 2\n\n   page1\n   page2\n\nAprès"
        result = self.cleaner.clean(text)
        assert "toctree" not in result
        assert "Avant" in result

    def test_cleans_inline_markup(self):
        text = "Du texte **gras** et *italique* et ``code``."
        result = self.cleaner.clean(text)
        assert "**" not in result
        assert "``" not in result
        assert "gras" in result
        assert "code" in result

    def test_handles_code_blocks(self):
        text = "Exemple:\n\n.. code-block:: python\n\n   x = 1\n   print(x)\n\nSuite."
        result = self.cleaner.clean(text)
        assert "[CODE:" in result
        assert "[/CODE]" in result

    def test_converts_field_lists(self):
        text = ":param name: Le nom du fichier.\n:returns: Le contenu."
        result = self.cleaner.clean(text)
        assert "Paramètre name" in result
        assert "Retourne" in result

    def test_empty_input(self):
        result = self.cleaner.clean("")
        assert result == ""


# ============================================================
# TESTS — MDXCleaner
# ============================================================

class TestMDXCleaner:
    """Tests pour le nettoyeur MDX."""

    def setup_method(self):
        self.cleaner = MDXCleaner()

    def test_removes_frontmatter(self):
        text = '---\ntitle: "Test"\nslug: test\n---\n\nContenu réel.'
        result = self.cleaner.clean(text)
        assert "---" not in result
        assert "Contenu réel" in result

    def test_removes_imports(self):
        text = 'import { Tabs } from "@components"\n\nContenu.'
        result = self.cleaner.clean(text)
        assert "import" not in result
        assert "Contenu" in result

    def test_handles_code_blocks(self):
        text = "Exemple:\n\n```python\nx = 1\n```\n\nSuite."
        result = self.cleaner.clean(text)
        assert "[CODE:" in result
        assert "[/CODE]" in result

    def test_removes_react_components(self):
        text = "Avant <SomeComponent /> après."
        result = self.cleaner.clean(text)
        assert "<SomeComponent" not in result
        assert "Avant" in result

    def test_cleans_inline_markdown(self):
        text = "Du **gras** et `code` et [lien](http://example.com)."
        result = self.cleaner.clean(text)
        assert "**" not in result
        assert "gras" in result
        assert "lien" in result
        assert "http://example.com" not in result


# ============================================================
# TESTS — TextCleaner
# ============================================================

class TestTextCleaner:
    """Tests pour le nettoyeur de texte transversal."""

    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_normalizes_unicode(self):
        text = "café"  # NFC normalization
        result = self.cleaner.clean(text)
        assert "café" in result or "cafe" in result

    def test_replaces_smart_quotes(self):
        text = "\u201CBonjour\u201D et \u2018test\u2019"
        result = self.cleaner.clean(text)
        assert '"' in result
        assert "'" in result

    def test_cleans_long_urls(self):
        text = "Voir https://example.com/" + "a" * 60 + " pour plus."
        result = self.cleaner.clean(text)
        assert "[URL]" in result

    def test_keeps_short_urls(self):
        text = "Voir https://example.com pour plus."
        result = self.cleaner.clean(text)
        assert "https://example.com" in result

    def test_handles_version_notes(self):
        text = "New in version 3.10 : cette fonctionnalité."
        result = self.cleaner.clean(text)
        assert "depuis version 3.10" in result

    def test_normalizes_whitespace(self):
        text = "Ligne 1\n\n\n\n\n\nLigne 2"
        result = self.cleaner.clean(text)
        assert "\n\n\n\n" not in result

    def test_empty_input(self):
        result = self.cleaner.clean("")
        assert result == ""

    def test_whitespace_only(self):
        result = self.cleaner.clean("   \n\n  ")
        assert result == ""
