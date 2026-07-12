"""Tests unitaires pour les fonctions utilitaires (utils.py)."""

import pytest
from utils import extract_rst_title, extract_mdx_title, classify_section


# ============================================================
# TESTS — extract_rst_title
# ============================================================

class TestExtractRstTitle:
    """Tests pour l'extraction de titres RST."""

    def test_standard_title(self):
        content = "Mon Titre\n=========\n\nContenu ici."
        assert extract_rst_title(content) == "Mon Titre"

    def test_dashes_underline(self):
        content = "Autre Titre\n-----------\n\nContenu."
        assert extract_rst_title(content) == "Autre Titre"

    def test_tilde_underline(self):
        content = "Sous-Titre\n~~~~~~~~~~\n\nContenu."
        assert extract_rst_title(content) == "Sous-Titre"

    def test_no_title(self):
        content = "Juste du texte sans titre.\nPas de soulignement."
        assert extract_rst_title(content) == "Sans titre"

    def test_empty_content(self):
        content = ""
        assert extract_rst_title(content) == "Sans titre"

    def test_underline_only(self):
        """Un soulignement seul ne devrait pas être considéré comme un titre."""
        content = "==========\n\nContenu."
        assert extract_rst_title(content) == "Sans titre"

    def test_title_with_special_chars(self):
        content = ":mod:`os.path` — Manipulation de chemins\n=========================================\n\nContenu."
        assert extract_rst_title(content) == ":mod:`os.path` — Manipulation de chemins"


# ============================================================
# TESTS — extract_mdx_title
# ============================================================

class TestExtractMdxTitle:
    """Tests pour l'extraction de titres MDX/Markdown."""

    def test_frontmatter_title(self):
        content = '---\ntitle: "Mon Titre"\n---\n\nContenu.'
        assert extract_mdx_title(content) == "Mon Titre"

    def test_frontmatter_title_no_quotes(self):
        content = '---\ntitle: Mon Titre Simple\n---\n\nContenu.'
        assert extract_mdx_title(content) == "Mon Titre Simple"

    def test_h1_title(self):
        content = "# Titre Principal\n\nContenu ici."
        assert extract_mdx_title(content) == "Titre Principal"

    def test_frontmatter_priority_over_h1(self):
        """Le frontmatter a priorité sur le titre H1."""
        content = '---\ntitle: "Titre FM"\n---\n\n# Titre H1\n\nContenu.'
        assert extract_mdx_title(content) == "Titre FM"

    def test_no_title(self):
        content = "Juste du texte sans titre ni frontmatter."
        assert extract_mdx_title(content) == "Sans titre"

    def test_empty_content(self):
        content = ""
        assert extract_mdx_title(content) == "Sans titre"


# ============================================================
# TESTS — classify_section
# ============================================================

class TestClassifySection:
    """Tests pour la classification de sections."""

    def test_tutorial(self):
        assert classify_section("tutorial/intro.rst", "python") == "tutorial"
        assert classify_section("getting_started/install.rst", "python") == "tutorial"

    def test_api_reference(self):
        assert classify_section("api/module.rst", "python") == "api_reference"
        assert classify_section("library/os.rst", "python") == "api_reference"

    def test_guide(self):
        assert classify_section("howto/logging.rst", "python") == "guide"
        assert classify_section("user_guide/intro.rst", "sklearn") == "guide"

    def test_example(self):
        assert classify_section("auto_examples/plot.rst", "sklearn") == "example"

    def test_faq(self):
        assert classify_section("faq.rst", "python") == "faq"

    def test_langchain_integrations(self):
        assert classify_section("integrations/openai.mdx", "langchain") == "integrations"

    def test_langchain_agents(self):
        assert classify_section("tool/custom_tool.mdx", "langchain") == "agents"

    def test_other(self):
        assert classify_section("misc/changelog.rst", "python") == "other"

    def test_sklearn_modules(self):
        assert classify_section("modules/svm.rst", "sklearn") == "guide"
