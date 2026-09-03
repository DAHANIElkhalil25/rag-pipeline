# 🤖 Pipeline RAG — Documentation Technique

Système de Question-Réponse basé sur **Retrieval-Augmented Generation (RAG)** appliqué à la documentation technique de Python 3.13, Scikit-learn 1.5 et LangChain.

**Projet de Fin d'Année — INSEA, Filière Data Science 2025–2026**
Organisme d'accueil : 3D Smart Factory

---

## Architecture du pipeline (6 étapes)

```
Documentation officielle
        │
   [Étape 1] Collecte        — Clone Git des dépôts officiels (Python, Sklearn, LangChain)
        │
   [Étape 2] Nettoyage       — Parsing RST/MDX, normalisation, dédoublonnage
        │
   [Étape 3] Benchmarking    — Grid search 108 configurations (chunk × embedding × search)
        │
   [Étape 4] Indexation      — Chunking + FAISS avec la config optimale
        │
   [Étape 5] Génération      — Mistral-7B-Instruct-v0.3 (4-bit, GPU T4)
        │
   [Étape 6] Évaluation      — Ragas 0.4.3 (120 questions, juge Mistral)
```

---

## Structure du dépôt

```
rag-pipeline/
│
├── core/                          ← Modules partagés (refactorisés)
│   ├── __init__.py
│   ├── tokenizer.py               ← Compteur de tokens (tiktoken)
│   ├── chunker.py                 ← Chunker unifié (section → paragraphe → taille)
│   ├── loaders.py                 ← Chargement des documents
│   └── search.py                  ← BM25 simplifié
│
├── tests/                         ← Tests unitaires (52 tests, 100% pass)
│   ├── test_chunker.py
│   ├── test_cleaners.py
│   └── test_utils.py
│
├── evaluation/                    ← Artefacts d'évaluation
│   ├── datasets/
│   │   └── eval_questions.json    ← 49 questions de benchmarking
│   ├── runs/
│   │   └── final_120_20260827T154025Z/
│   │       ├── manifest.json      ← Config exacte du run (modèle, retrieval, ragas)
│   │       ├── summary.json       ← Scores RAGAS par domaine
│   │       └── samples.csv        ← Résultats par question
│   ├── manifeste_execution.json   ← Manifeste global du pipeline
│   ├── statistiques_rapport.json  ← Stats agrégées
│   ├── scores_ragas_finaux.png    ← Graphique radar des métriques
│   └── chunks_par_source.png      ← Distribution des chunks
│
├── benchmarks/                    ← Résultats du grid search (étape 3)
│   ├── benchmark_full_grid.csv    ← 108 configurations × 3 métriques
│   ├── benchmark_report.json      ← Config optimale recommandée
│   ├── benchmark_chunking.csv
│   ├── benchmark_embeddings.csv
│   ├── benchmark_search.csv
│   └── chunking_report.json       ← Rapport de découpage du corpus
│
├── notebooks/
│   └── notebook_kaggle_rag.ipynb  ← Notebook Kaggle (template, GPU T4)
│
├── rapport_pfa_latex/
│   └── rapport_stage.tex          ← Source LaTeX du rapport PFA
│
├── etape1_collecte.py             ← Étape 1 : collecte des données
├── etape2_nettoyage.py            ← Étape 2 : nettoyage et préparation
├── etape3_benchmarking.py         ← Étape 3 : grid search 108 configs
├── etape4_indexation.py           ← Étape 4 : chunking et indexation FAISS
├── etape5_generation.py           ← Étape 5 : génération Mistral 7B
├── etape6_evaluation.py           ← Étape 6 : évaluation Ragas
├── main.py                        ← Point d'entrée CLI
├── config.py                      ← Configuration centralisée
├── utils.py                       ← Fonctions utilitaires partagées
└── requirements.txt               ← Dépendances avec versions
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Exécution

```bash
# Pipeline complet (étapes 1 à 6)
python main.py

# Étapes individuelles
python main.py --etape 1    # Collecte
python main.py --etape 2    # Nettoyage
python main.py --etape 3    # Benchmarking (2-4h)
python main.py --etape 4    # Indexation FAISS
python main.py --etape 5    # Génération (nécessite GPU ou Ollama)
python main.py --etape 6    # Évaluation Ragas
```

## Tests

```bash
python -m pytest tests/ -v
```

---

## Résultats expérimentaux

### Benchmarking (étape 3) — Grid search 108 configurations

Exécuté sur Kaggle GPU T4, durée **~1h26**. Configuration optimale identifiée :

| Paramètre | Valeur optimale | MRR@5 |
|-----------|----------------|-------|
| Taille de chunk | 400 tokens | — |
| Chevauchement | 30 tokens | — |
| Méthode de recherche | Hybride α=0.7 | **0.9276** |

> Résultats complets dans [`benchmarks/benchmark_full_grid.csv`](benchmarks/benchmark_full_grid.csv)

### Évaluation finale (étape 6) — 120 questions, Ragas 0.4.3

Système final : `multilingual-e5-base` + BM25 hybride + reranking `bge-reranker-v2-m3` + `Mistral-7B-Instruct-v0.3`

| Métrique | Global | Python | Scikit-learn | LangChain |
|----------|--------|--------|--------------|-----------|
| **Faithfulness** | 0.888 | 0.890 | 0.891 | 0.882 |
| **Answer Relevancy** | 0.913 | 0.910 | 0.924 | 0.907 |
| **Context Precision** | 0.819 | 0.869 | 0.864 | 0.726 |
| **Context Recall** | 0.879 | 0.875 | 0.925 | 0.838 |
| **Factual Correctness** | 0.693 | 0.758 | 0.708 | 0.611 |

> Détails par question dans [`evaluation/runs/final_120_20260827T154025Z/samples.csv`](evaluation/runs/final_120_20260827T154025Z/samples.csv)

---

## Notes techniques

- Le premier lancement télécharge ~200 Mo de dépôts Git (Python + Sklearn + LangChain)
- Les lancements suivants réutilisent les clones existants (idempotence)
- L'étape 3 prend **2–4h** selon le hardware (108 configurations × embeddings)
- L'étape 5 nécessite un GPU ou Ollama local (`http://localhost:11434`)
- Les clés API Ragas sont lues depuis les **secrets Kaggle** (jamais dans le code)
- Testé avec Python 3.10+ (Kaggle : Python 3.12)

## Sortie de l'étape 4 (données volumineuses, non versionnées)

```
rag_project/data/
├── raw/                       ← Documents bruts clonés (~200 Mo)
├── processed/
│   ├── cleaned/               ← Documents nettoyés (~1 050 fichiers JSON)
│   └── vectorstore/
│       ├── faiss_index.bin    ← Index FAISS binaire
│       └── chunks_metadata.json
└── evaluation/                ← Résultats locaux Ragas (non versionnés)
```
