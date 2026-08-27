# Baseline historique — éléments vérifiés pour le rapport PFA

## Identification de l’artefact

| Champ | Valeur vérifiée |
|---|---|
| Nom de la baseline | `baseline_v1_custom_ragas_inspired` |
| Artefact de référence | `evaluation/baseline/baseline_metrics_v1.json` |
| Commit source enregistré | `c88e287160cf17bc503233676900b67f2bfd43e6` |
| Taille du jeu évalué | 20 questions : 7 Python, 6 Scikit-learn, 7 LangChain |
| Évaluateur | Code personnalisé inspiré de Ragas, non équivalent à Ragas 0.4.3 officiel |
| Contextes évalués par question | Au plus 5 |
| Température du juge | 0,0 |
| Contrôle des affirmations | Au plus 8 affirmations par réponse |

## Résultats archivés

| Catégorie | Mesure | Valeur |
|---|---|---:|
| Récupération | Hit rate@5 | 0,9796 |
| Récupération | MRR@5 | 0,9265 |
| Récupération | Precision@5 | 0,8245 |
| Génération / contexte | Faithfulness | 0,7888 |
| Génération / question | Answer Relevancy | 0,7267 |
| Contexte | Context Precision | 0,5619 |
| Contexte | Context Recall | 0,6250 |

## Rôle méthodologique exact

La baseline est le **point de départ diagnostique** du projet. Elle a été exécutée sur un jeu court de 20 questions et a permis de faire apparaître des limites initiales de pertinence, de précision de contexte et de rappel de contexte. Elle est conservée dans une archive immuable afin de ne pas réécrire l’historique expérimental.

Elle n’est cependant pas une référence numérique strictement comparable à l’évaluation finale. Les questions, la méthode d’annotation des références, l’implémentation des métriques et le juge ne sont pas identiques. L’évaluation finale sur 120 questions utilise Ragas 0.4.3, des références documentaires et des identifiants de chunks annotés, ainsi qu’un juge Mistral externe. Une déclaration du type « le score final est supérieur à la baseline de X % » serait donc incorrecte.

## Modèle et configuration de récupération : précision nécessaire

Le code au commit archivé indique une configuration de repli composée de `all-MiniLM-L6-v2` avec recherche sémantique seule et cinq passages. Il pouvait toutefois charger une configuration alternative depuis un fichier de benchmarking lorsqu’un tel fichier était disponible dans l’environnement Kaggle. Aucun manifeste de benchmarking correspondant à cette exécution historique n’a été archivé dans le dépôt.

Par rigueur, le rapport décrit donc la **baseline comme une évaluation personnalisée de la configuration RAG historique** et ne présente pas `all-MiniLM-L6-v2` comme paramètre de fait du run exécuté. Le modèle `all-MiniLM-L6-v2` peut être mentionné comme la configuration de repli présente dans le code source, mais pas comme une valeur expérimentale certifiée.

## Formulation à insérer dans le rapport

> La baseline `baseline_v1_custom_ragas_inspired`, archivée au commit `c88e287`, a évalué une configuration RAG historique sur 20 questions au moyen d’un évaluateur personnalisé inspiré de Ragas. Elle a permis d’identifier les premiers axes d’amélioration et constitue un diagnostic initial. Le code versionné prévoit par défaut une recherche sémantique fondée sur `all-MiniLM-L6-v2`, mais le manifeste précis de l’exécution Kaggle historique n’a pas été conservé. La baseline est donc documentée comme point de départ, sans revendiquer une comparaison numérique stricte avec l’évaluation finale Ragas 0.4.3 sur 120 questions.
