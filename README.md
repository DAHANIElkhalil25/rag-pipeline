# 🤖 Pipeline RAG — Documentation Technique

## Projet
Système de Question-Réponse basé sur RAG (Retrieval-Augmented Generation) appliqué à la documentation technique Python, Scikit-learn et LangChain.

## Structure des fichiers

```
rag_project_src/
│
├── main.py                 ← Point d'entrée (exécuter celui-ci)
├── config.py               ← Configuration : chemins, constantes, paramètres
├── utils.py                ← Fonctions utilitaires partagées
├── etape1_collecte.py      ← Étape 1 : collecte des données (clone GitHub)
├── etape2_nettoyage.py     ← Étape 2 : nettoyage et préparation
├── etape3_benchmarking.py  ← Étape 3 : benchmarking des paramètres
├── etape4_indexation.py    ← Étape 4 : chunking et indexation FAISS
├── requirements.txt        ← Dépendances avec versions
├── core/                   ← Modules partagés (refactorisés)
│   ├── __init__.py
│   ├── tokenizer.py        ← Compteur de tokens (tiktoken)
│   ├── chunker.py          ← Chunker unifié (section → paragraphe → taille)
│   ├── loaders.py          ← Chargement des documents + vérif. dépendances
│   └── search.py           ← BM25 simplifié
├── tests/                  ← Tests unitaires
│   ├── test_utils.py
│   ├── test_cleaners.py
│   └── test_chunker.py
└── README.md               ← Ce fichier
```

## Installation

```bash
pip install -r requirements.txt
```

## Exécution

### Depuis VS Code (terminal intégré)

```bash
# Exécuter tout le pipeline (étapes 1 + 2 + 3 + 4)
python main.py

# Exécuter une seule étape
python main.py --etape 1    # Collecte uniquement
python main.py --etape 2    # Nettoyage uniquement (nécessite étape 1)
python main.py --etape 3    # Benchmarking (nécessite étape 2)
python main.py --etape 4    # Indexation (nécessite étape 2, utilise étape 3 si dispo)
```

### Depuis Jupyter Notebook

Créez un notebook dans le même dossier et exécutez :

```python
# Cellule 1 — Étape 1
%run etape1_collecte.py

# Cellule 2 — Étape 2
%run etape2_nettoyage.py

# Cellule 3 — Étape 3 (optionnel, benchmarking)
%run etape3_benchmarking.py

# Cellule 4 — Étape 4 (indexation)
%run etape4_indexation.py
```

Ou importez les fonctions pour un contrôle plus fin :

```python
# Cellule 1
from etape1_collecte import *
init_directories()

# Cellule 2 — Python uniquement
python_dir = clone_cpython_docs()
python_meta = collect_python_docs(python_dir)

# Cellule 3 — Scikit-learn
sklearn_dir = clone_sklearn_docs()
sklearn_meta = collect_sklearn_docs(sklearn_dir)

# etc.
```

## Tests

```bash
# Exécuter tous les tests
python -m pytest tests/ -v

# Exécuter un fichier de tests spécifique
python -m pytest tests/test_utils.py -v
```

## Sortie

Après exécution, l'arborescence suivante est créée :

```
rag_project/
└── data/
    ├── raw/                          ← Données brutes (étape 1)
    │   ├── python_docs/
    │   │   ├── cpython_clone/Doc/    ← Clone Git
    │   │   └── json_docs/            ← Documents JSON standardisés
    │   ├── sklearn_docs/
    │   │   ├── sklearn_clone/doc/
    │   │   └── json_docs/
    │   └── langchain_docs/
    │       ├── langchain_docs_clone/ ← Clone Git
    │       └── json_docs/
    ├── processed/                    ← Données nettoyées (étape 2)
    │   ├── cleaned/
    │   │   ├── python/
    │   │   ├── sklearn/
    │   │   ├── langchain/
    │   │   └── corpus_cleaned_index.csv
    │   └── vectorstore/              ← Index FAISS (étape 4)
    │       ├── faiss_index.bin
    │       └── chunks_metadata.json
    ├── benchmarks/                   ← Résultats benchmarking (étape 3)
    │   ├── benchmark_chunking.csv
    │   ├── benchmark_embeddings.csv
    │   ├── benchmark_search.csv
    │   ├── benchmark_report.json
    │   └── eval_questions.json
    └── metadata/
        ├── corpus_index.csv
        └── chunking_report.json
```

## Notes

- Le premier lancement télécharge ~200 Mo de dépôts Git (Python + Sklearn + LangChain)
- Les lancements suivants réutilisent les clones existants (idempotence)
- L'étape 1 prend ~5-10 min, l'étape 2 ~2-5 min
- L'étape 3 (benchmarking) prend ~10-30 min selon le hardware
- L'étape 4 utilise automatiquement les paramètres optimaux de l'étape 3 si disponibles
- Testé avec Python 3.10+
