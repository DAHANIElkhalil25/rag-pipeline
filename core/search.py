"""
core/search.py — Recherche BM25 simplifiée
=============================================
Implémentation BM25 légère sans dépendance externe.

Réf. [4] Robertson & Zaragoza (2009) — BM25 and Beyond.
Paramètres standards : k1=1.5 (saturation des termes),
b=0.75 (normalisation par longueur).

Cette implémentation est suffisante pour le benchmarking.
En production, on utiliserait rank_bm25 ou Elasticsearch.

Utilisé par : etape3_benchmarking.py
"""

import re
from typing import List, Tuple
from collections import defaultdict

import numpy as np


class SimpleBM25:
    """
    Implémentation BM25 légère sans dépendance externe.

    Parameters
    ----------
    corpus : list[str]
        Liste des documents à indexer.
    k1 : float
        Paramètre de saturation des termes (défaut: 1.5).
    b : float
        Paramètre de normalisation par longueur (défaut: 0.75).
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        # Tokenizer simple : lowercase + split sur non-alphanumérique
        self.tokenized = [self._tokenize(doc) for doc in corpus]

        # Longueur moyenne des documents
        self.avgdl = np.mean([len(doc) for doc in self.tokenized])

        # Document frequency (DF) pour chaque terme
        self.df = defaultdict(int)
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] += 1

        # IDF pré-calculé
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = np.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def score(self, query: str) -> np.ndarray:
        """Calcule le score BM25 de la requête contre tous les documents."""
        query_terms = self._tokenize(query)
        scores = np.zeros(self.corpus_size)

        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self.tokenized):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = len(doc)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * numerator / denominator

        return scores

    def search(self, query: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les top-k indices et scores."""
        scores = self.score(query)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return scores[top_k_idx], top_k_idx
