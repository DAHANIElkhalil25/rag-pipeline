# Guide simple — Notebook RAG final

Utilisez uniquement le fichier **`notebook_final_rag_fr.ipynb`**. Il contient les étapes 1 à 6 du pipeline et les améliorations techniques.

## Avant de commencer dans Kaggle

Activez **Internet** et **GPU T4**. Importez ensuite le notebook dans Kaggle. Pour le premier lancement, ne modifiez aucune valeur dans la cellule 1 : le mode doit rester `PREPARATION`.

## Premier lancement : préparation du système amélioré

Exécutez les cellules dans l'ordre, de la cellule 1 à la cellule 8, ou utilisez **Run All**.

| Cellule | Ce qu'elle fait | Ce que vous devez faire |
|---|---|---|
| 1 | Définit le mode et les paramètres simples. | Ne changez rien. |
| 2 | Télécharge le dépôt et prépare Kaggle. | Attendez la fin. |
| 3 | Exécute les étapes 1 à 4 : collecte, nettoyage, benchmarking et index. | Attendez la fin. |
| 4 | Exécute l'étape 5 : réponse RAG avec sources. | Lisez la réponse et les sources affichées. |
| 5 | Prépare les 120 questions et les passages candidats. | Téléchargez le fichier `candidats_120_questions.jsonl`. |
| 6 | Sauvegarde les statistiques utiles pour le rapport. | Gardez le dossier `statistiques_rapport`. |
| 7 | Lance l'interface facultative. | Ignorez-la au premier lancement. |
| 8 | Crée l'archive finale. | Téléchargez `resultats_rag_stage.zip`. |

Après le premier lancement, envoyez dans le chat le fichier **`candidats_120_questions.jsonl`**. Il sera utilisé pour préparer le jeu de test final validé.

## Second lancement : évaluation finale Ragas

Effectuez ce lancement seulement après avoir reçu le fichier `test_dataset_v1.jsonl` validé et après avoir créé votre Secret Kaggle `OPENAI_API_KEY`.

Dans la cellule 1, modifiez uniquement :

```python
MODE = "EVALUATION_FINALE"
```

Puis exécutez de nouveau les cellules 1 à 8 dans l'ordre. La cellule 5 exécutera Ragas sur les 120 questions. La cellule 6 sauvegardera les résultats, les graphiques et les statistiques finales.

## Fichiers à conserver pour le rapport

| Fichier ou dossier | Utilité |
|---|---|
| `statistiques_rapport/statistiques_rapport.json` | Chiffres de corpus, index, chunks, configuration et résultats. |
| `statistiques_rapport/chunks_par_source.png` | Graphique de couverture documentaire. |
| `statistiques_rapport/scores_ragas_finaux.png` | Graphique des métriques finales Ragas, disponible après évaluation. |
| `evaluation/runs/<id_run>/summary.json` | Résumé final des métriques Ragas. |
| `evaluation/runs/<id_run>/samples.csv` | Résultats détaillés question par question. |
| `manifeste_execution.json` | Preuve de reproduction : version du code, profil, date et fichiers. |
| `resultats_rag_stage.zip` | Archive complète à sauvegarder. |

> La baseline historique reste séparée. Dans le rapport, elle sert à expliquer les limites initiales ; les résultats Ragas sur 120 questions sont les résultats finaux principaux.
