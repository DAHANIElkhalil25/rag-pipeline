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

## Baseline pré-améliorations

La baseline est conservée de manière immuable afin de permettre la comparaison académique avant/après. Les résultats observés dans Kaggle sont enregistrés dans `evaluation/baseline/baseline_metrics_v1.json`, le notebook initial est archivé dans `notebooks/baseline_kaggle_rag_v1.ipynb` et l'ancien évaluateur local est conservé dans `legacy/etape6_baseline_custom.py`.

La baseline utilise 20 questions et un évaluateur local inspiré de Ragas. Elle ne doit pas être écrasée par les expériences du système final.

## Notebook Kaggle baseline

Le notebook `notebooks/baseline_kaggle_rag_v1.ipynb` archive le protocole initial. Il clone une version déterminée du dépôt, sépare le code des données produites, réutilise les artefacts existants et charge une seule instance du pipeline avant la démonstration, l'évaluation et l'interface.

Dans Kaggle, activez Internet et un GPU T4, puis exécutez les cellules dans l'ordre. Les données sont écrites dans `/kaggle/working/rag_data` et les principaux résultats sont exportés sous forme de fichiers JSON, CSV et PNG. L'archive `/kaggle/working/rag_results_bundle.zip` peut être téléchargée depuis l'onglet Files.

Les fichiers d'évaluation comprennent `ragas_report.json`, `ragas_details.csv`, `evaluation_summary.csv`, `evaluation_details_normalized.csv`, `evaluation_by_source.png`, `evaluation_coverage.png` et `run_manifest.json`. Les valeurs invalides restent absentes et sont accompagnées d'un statut et d'un compteur d'erreurs ; elles ne sont pas remplacées arbitrairement par `0.5`.

## Système final et notebook Kaggle séparé

Utilisez `notebooks/final_kaggle_rag_system_v2.ipynb` pour le système final. Ce notebook reconstruit le corpus et l'index final, applique réellement le dédoublonnage, produit un manifeste d'index, sauvegarde les fenêtres de contexte exactes fournies au générateur et lance l'évaluation officielle Ragas v0.4.3 sans modifier la baseline.

Dans Kaggle, activez Internet et un GPU T4. Pour lancer Ragas, créez un Kaggle Secret nommé `OPENAI_API_KEY` ou `MISTRAL_API_KEY`, sélectionnez le provider et le modèle juge dans la cellule de configuration, puis activez `RUN_OFFICIAL_RAGAS`. La clé ne doit jamais être écrite dans le notebook ni dans le dépôt.

Le notebook distingue strictement les jeux suivants :

| Fichier | Rôle |
|---|---|
| `evaluation/datasets/dev_dataset_v1.jsonl` | Les 20 questions historiques, utilisées seulement pendant le développement. |
| `evaluation/datasets/test_dataset_v1_annotation_template.csv` | Le plan équilibré de 120 questions finales à annoter manuellement. |
| `evaluation/datasets/test_dataset_v1.jsonl` | Le test final validé et gelé, créé seulement après revue humaine. |

Après avoir complété le CSV de 120 questions, convertissez-le avec :

```bash
python evaluation/convert_annotations.py \
  --input evaluation/datasets/test_dataset_v1_annotation_template.csv \
  --output evaluation/datasets/test_dataset_v1.jsonl
```

La conversion refuse un test final incomplet, non relu ou dépourvu d'identifiants de chunks de référence. Les règles détaillées se trouvent dans `evaluation/annotation_guidelines.md`.

## Évaluation Ragas officielle du système final

Le module `evaluation/ragas_runner.py` est l'entrée principale du système final. Il charge le jeu JSONL, enregistre les réponses, les chunks récupérés, les identifiants de chunks, les textes exacts fournis au prompt, les erreurs et les métriques par question. Il sauvegarde ensuite `samples.jsonl`, `samples.csv`, `summary.json` et `manifest.json` dans `data/evaluation/runs/<run_id>/`.

Les métriques officielles utilisées sont `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall` et `FactualCorrectness`. Lorsque les `reference_context_ids` ont été annotés, le runner ajoute aussi des métriques déterministes de précision et rappel par identifiants. Les références de compatibilité Ragas sont documentées dans `evaluation/official_ragas_sources.md`.

Exemple CLI :

```bash
export OPENAI_API_KEY="..."
python main.py --final-eval \
  --dataset evaluation/datasets/dev_dataset_v1.jsonl \
  --run-id dev_ragas_v1 \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --api-key-env OPENAI_API_KEY
```

Le générateur local Mistral et le modèle juge Ragas sont volontairement séparés. La baseline locale reste disponible uniquement pour la comparaison historique ; les résultats Ragas du système final sont les métriques principales à utiliser dans la comparaison finale.

## Interface utilisateur

Le module `ui.py` fournit une interface Gradio destinée à Kaggle. Il faut appeler `launch_ui(pipeline, share=True)` après avoir chargé le pipeline. L'interface réutilise la même instance de modèle et affiche la réponse ainsi que les sources récupérées et leurs scores. Elle n'est pas lancée automatiquement par le programme CLI afin de conserver un mode headless compatible avec les notebooks et les environnements CI.

## Interprétation des métriques baseline

L'évaluation de l'étape 6 est une implémentation locale et transparente de métriques inspirées de Ragas. Le rapport distingue la fidélité, la pertinence de la réponse, la précision du contexte et le rappel du contexte. Si une compatibilité stricte avec le paquet officiel Ragas est requise, il faudra ajouter un adaptateur pour le client LLM et le modèle d'embedding avant d'utiliser les métriques officielles.

La précision du contexte est calculée de manière sensible au rang sur les cinq premiers passages, tandis que le rappel du contexte utilise les affirmations de la réponse de référence. Les résultats doivent être interprétés conjointement avec la couverture des scores valides, les erreurs de jugement et les métriques IR du benchmarking.
