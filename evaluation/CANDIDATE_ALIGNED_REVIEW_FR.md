# Jeu de revue aligné sur les candidats Kaggle

Le fichier `datasets/test_dataset_v1_candidate_aligned_review.jsonl` est une version de travail complète de **120 questions**. Il a été construit uniquement à partir des identifiants de passages contenus dans les fichiers de candidats déjà produits par Kaggle. Il ne demande aucune nouvelle collecte, indexation ou exécution du notebook.

Les questions dont un passage direct avait été sélectionné lors des deux revues de preuve ont été conservées. Les questions restantes ont été reformulées de façon plus étroite, uniquement lorsque le texte d’un passage candidat permettait d’établir la nouvelle réponse. Cette opération conserve l’équilibre de 40 questions Python, 40 scikit-learn et 40 LangChain.

| Élément | Valeur |
|---|---:|
| Nombre total de questions | 120 |
| Questions conservées | 98 |
| Questions reformulées à partir d’un passage candidat | 22 |
| Identifiants de contexte renseignés | 120/120 |
| Statut | `ai_prevalidated_pending_human_review` |

Le fichier CSV associé, `datasets/test_dataset_v1_candidate_aligned_review.csv`, permet une relecture rapide. Chaque ligne contient la question, la réponse de référence, les identifiants de chunks et les URLs exactes des passages retenus. Le champ `researcher_confirmation` est volontairement vide : il ne faut pas présenter ce jeu comme « validé humainement » tant qu’un relecteur n’a pas donné son accord.

> Ce jeu est **prêt pour une revue humaine courte**, mais n’est pas encore `test_dataset_v1.jsonl`. Le script d’évaluation finale refuse volontairement un statut non validé. Cette séparation protège la crédibilité académique du rapport : une prévalidation assistée n’est jamais confondue avec une validation humaine.

Lorsqu’une validation humaine sera disponible, les 120 lignes pourront être transférées vers le format final avec `review_status="validated"`, le nom du relecteur et la date de validation. À ce moment seulement, l’évaluation Ragas sur les 120 questions doit être exécutée.
