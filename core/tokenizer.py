"""
core/tokenizer.py — Compteur de tokens unifié
===============================================
Utilise tiktoken (même tokenizer BPE que les LLM) pour un
comptage cohérent avec ce que le modèle verra réellement.

Pourquoi tiktoken et pas len(text.split()) ?
Le comptage par espaces est imprécis : "don't" est 1 mot mais 2 tokens,
"machine-learning" est 1 mot mais 3 tokens.

Utilisé par : etape3_benchmarking.py, etape4_indexation.py
"""


class TokenCounter:
    """Compteur de tokens basé sur tiktoken (BPE, même tokenizer que les LLM)."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.encoder = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Retourne le nombre de tokens dans le texte."""
        return len(self.encoder.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Tronque le texte à max_tokens tokens."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])
