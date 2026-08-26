# Protocole de validation experte assistée par IA

La validation du jeu de 120 questions suit un protocole de preuve directe. Pour chaque ligne, le relecteur examine uniquement la question en français, la réponse de référence, les identifiants de chunks sélectionnés, les URLs de documentation et les extraits associés. Une validation positive exige que la réponse réponde à la question, que ses affirmations importantes soient explicites dans les extraits, que l’URL soit une source documentaire officielle et qu’aucune information externe ne soit ajoutée.

| Verdict | Interprétation | Traitement |
|---|---|---|
| `approve` | La preuve sélectionnée soutient directement la réponse. | La ligne peut être gelée avec le statut de validation experte IA. |
| `revise` | Une formulation plus étroite est nécessaire pour respecter exactement l’extrait. | La question ou la réponse est corrigée, puis contrôlée de nouveau. |
| `reject` | L’extrait ne permet pas de soutenir une formulation correcte. | La ligne est exclue du gel jusqu’à correction. |

Le contrôle automatique ajoute trois garanties supplémentaires : il vérifie le total de 120 questions, la répartition équilibrée de 40 questions par domaine et l’existence de chaque identifiant de chunk dans les candidats Kaggle reçus.

> La sortie constitue une **validation experte assistée par IA**. Dans le rapport de stage, elle doit être décrite comme telle ; elle ne doit pas être présentée comme une annotation indépendante par plusieurs évaluateurs humains.

## Résultat appliqué au jeu final

La première revue de preuves a conservé 98 questions dont les passages soutenaient déjà directement la réponse. Les 22 autres questions ont été reformulées à partir de passages réellement disponibles dans les candidats Kaggle. Un contrôle expert supplémentaire a approuvé 16 formulations et proposé 23 corrections de formulation, qui ont été appliquées. Deux rejets de fond ont été corrigés manuellement à partir de leurs passages directs.

Le service de relecture a ensuite retourné 79 erreurs de disponibilité de crédits. Ces erreurs sont archivées dans `test_dataset_v1_ai_expert_review.jsonl`, mais ne sont **pas** interprétées comme des rejets du contenu. Elles constituent une limite méthodologique explicitement signalée dans `test_dataset_v1_expert_validation_report.json`.

Le fichier final `datasets/test_dataset_v1.jsonl` contient donc 120 questions, 40 par domaine, et un ou plusieurs identifiants de chunks de référence pour chaque ligne. Son statut `validated` est accepté par le runner Ragas, mais ses métadonnées indiquent clairement `Manus AI — validation experte de preuve directe` comme relecteur unique.
