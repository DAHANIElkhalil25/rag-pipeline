# Analyse de l’évaluation finale Ragas — système RAG documentaire

**Exécution analysée :** `final_120_20260827T154025Z`  
**Date de fin :** 27 août 2026, 17:20 UTC  
**Jeu évalué :** 120 enregistrements (40 Python, 40 Scikit-learn, 40 LangChain)  
**Générateur :** `mistralai/Mistral-7B-Instruct-v0.3`  
**Juge Ragas :** `mistral-small-latest` via l’API Mistral compatible OpenAI  
**Récupération :** E5 multilingue + hybride BM25/dense + reranking BGE, cinq passages retournés par question.

## 1. Conclusion immédiate

L’exécution a bien atteint le stade d’évaluation : contrairement au premier essai OpenAI, les cinq métriques Ragas ont été calculées pour l’essentiel du jeu. Aucun échantillon n’est en échec complet ; 114/120 sont entièrement évalués et 6/120 sont partiels à cause de réponses incomplètes du juge ou d’une indisponibilité temporaire du service Mistral.

> Cette exécution est donc un **résultat Ragas exploitable**, mais elle doit être présentée comme **provisoire** : cinq lignes du jeu de test sont mal formées et doivent être corrigées avant de revendiquer un score final définitivement gelé.

| État des 120 échantillons | Nombre | Part |
|---|---:|---:|
| Entièrement évalués | 114 | 95,0 % |
| Partiellement évalués | 6 | 5,0 % |
| Échec complet | 0 | 0,0 % |

## 2. Résultats globaux Ragas

Les métriques Ragas sont sur une échelle de 0 à 1 ; un score plus élevé est meilleur. Faithfulness vérifie si la réponse est justifiée par le contexte récupéré. Answer Relevancy mesure la pertinence de la réponse par rapport à la question. Context Precision et Context Recall évaluent l’utilité et la couverture des passages récupérés. Factual Correctness compare la réponse à la réponse de référence. [1] [2] [3] [4] [5]

| Métrique | Moyenne sur 120 | Couverture | Lecture pédagogique |
|---|---:|---:|---|
| Faithfulness | **0,888** | 116/120 (96,7 %) | Les réponses sont généralement bien appuyées par les passages fournis au générateur. |
| Answer Relevancy | **0,913** | 120/120 (100 %) | Les réponses répondent le plus souvent à la question posée. |
| Context Precision | **0,819** | 119/120 (99,2 %) | Les passages récupérés sont souvent utiles, mais pas toujours ciblés. |
| Context Recall | **0,879** | 120/120 (100 %) | Les éléments nécessaires à la réponse sont fréquemment présents dans les passages récupérés. |
| Factual Correctness | **0,693** | 118/120 (98,3 %) | Les réponses restent le principal axe d’amélioration : elles sont parfois pertinentes et fondées sur le contexte, mais insuffisamment exactes par rapport à la réponse de référence. |

Le profil est cohérent pour un premier système RAG : la récupération et l’ancrage documentaire sont bons, tandis que la fidélité exacte de la formulation finale demeure plus faible. Un score de Faithfulness élevé ne garantit pas, à lui seul, une réponse parfaitement exacte : le modèle peut rester dans le contexte fourni tout en donnant une explication incomplète, trop générale ou légèrement imprécise. C’est précisément pourquoi les métriques doivent être interprétées ensemble.

## 3. Résultats par domaine

| Domaine | Faithfulness | Relevancy | Context Precision | Context Recall | Factual Correctness | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| Python | 0,890 | 0,910 | **0,869** | 0,875 | **0,758** | Domaine le plus solide pour la précision des contextes et l’exactitude finale. |
| Scikit-learn | **0,891** | **0,924** | 0,864 | **0,925** | 0,708 | Très bonne couverture documentaire ; réponses parfois moins exactes que le contexte disponible. |
| LangChain | 0,882 | 0,907 | **0,726** | **0,838** | **0,611** | Domaine prioritaire : récupération moins ciblée et réponses les moins exactes. |

LangChain est le point faible net. Cela correspond à la structure plus vaste et plus évolutive de sa documentation : les pages d’intégration, de retrievers et d’agents peuvent être proches lexicalement tout en répondant à des besoins différents. Il faudra prioriser ce domaine si le système doit être amélioré après l’évaluation de stage.

## 4. Diagnostic de la récupération avec les identifiants de chunks

En plus du jugement LLM, l’exécution calcule des mesures déterministes à partir des identifiants exacts de chunks annotés dans le jeu de test. Elles permettent de vérifier si les passages de référence sont effectivement parmi les cinq passages remis au générateur.

| Indicateur déterministe | Valeur | Interprétation |
|---|---:|---|
| Hit@5 | **0,792** | Au moins un chunk de référence est retrouvé dans le top 5 pour 95 questions sur 120. |
| MRR@5 | **0,633** | Lorsqu’un chunk de référence est trouvé, il apparaît souvent tôt dans le classement, mais pas toujours en première position. |
| Rang 1 | 63 questions | Le bon chunk est premier dans 52,5 % des cas. |
| Rang 2 | 17 questions | Le bon chunk est second dans 14,2 % des cas. |
| Absent du top 5 | 25 questions | Les 20,8 % restants représentent le principal défaut du récupérateur. |

| Domaine | Hit@5 | MRR@5 | Chunks de référence absents du top 5 | Priorité |
|---|---:|---:|---:|---|
| Python | 0,800 | 0,618 | 8/40 | Moyenne |
| Scikit-learn | **0,925** | **0,754** | 3/40 | Faible |
| LangChain | **0,650** | **0,527** | 14/40 | Très élevée |

La précision déterministe par identifiants est de 0,240. Cette valeur ne signifie pas automatiquement que 76 % des passages sont inutiles. Le système retourne systématiquement cinq chunks, alors que 47 questions n’ont qu’un seul chunk de référence annoté : même si ce seul chunk est classé premier, la précision par identifiant vaut mécaniquement 1/5 = 0,20. Il faut donc privilégier l’interprétation conjointe de Hit@5, MRR@5, rappel et métriques Ragas de contexte.

## 5. Cas et erreurs à connaître

Six échantillons sont partiels. Ils n’indiquent pas une erreur du RAG : ils correspondent à des réponses du juge Mistral interrompues par la limite de sortie ou à une indisponibilité temporaire du service. Les autres métriques de ces questions ont été conservées.

| Question | Domaine | Mesure manquante | Cause |
|---|---|---|---|
| `test_python_004` | Python | Factual Correctness | Service Mistral temporairement indisponible (503). |
| `test_scikit_learn_004` | Scikit-learn | Context Precision | Réinitialisation réseau en amont. |
| `test_scikit_learn_017` | Scikit-learn | Faithfulness | Service Mistral temporairement indisponible (503). |
| `test_scikit_learn_033` | Scikit-learn | Faithfulness | Sortie du juge incomplète (limite de tokens). |
| `test_langchain_027` | LangChain | Faithfulness | Sortie du juge incomplète (limite de tokens). |
| `test_langchain_039` | LangChain | Faithfulness, Factual Correctness | Sortie du juge incomplète (limite de tokens). |

Les cas faibles les plus révélateurs sont : `test_langchain_011` (différence LangChain/LangGraph : pertinence et précision de contexte faibles), `test_langchain_037` (compression du contexte pour Deep Agents : Faithfulness nulle), `test_python_029` (annotation d’un ensemble : Faithfulness nulle), `test_scikit_learn_009` (scaler et valeurs aberrantes : Faithfulness nulle) et `test_scikit_learn_025` (choix d’une métrique : réponse visiblement hors sujet). Ces exemples servent de base à une analyse qualitative dans le rapport, plutôt qu’à une conclusion fondée seulement sur les moyennes.

## 6. Limite du jeu de test à corriger

Le contrôle structurel des artefacts a identifié cinq enregistrements mal formés : dans ces lignes, le champ `user_input` contient une phrase déclarative proche d’une réponse, alors que le champ `reference` contient une URL de documentation au lieu d’une réponse de référence. Les identifiants concernés sont :

```text
test_python_020
test_python_022
test_python_024
test_python_032
test_scikit_learn_003
```

Cette erreur est dans le jeu final créé lors de la reformulation de certaines questions, pas dans la récupération Kaggle. Ces cinq lignes représentent 4,2 % du jeu. Pour une présentation méthodologiquement rigoureuse, elles doivent être réparées puis le run doit être refait une fois. En attendant, une analyse de sensibilité sur les 115 lignes correctement formées donne les résultats suivants :

| Métrique | Moyenne sur 115 lignes correctement formées |
|---|---:|
| Faithfulness | 0,888 |
| Answer Relevancy | 0,916 |
| Context Precision | 0,847 |
| Context Recall | 0,917 |
| Factual Correctness | 0,698 |

Les conclusions principales ne changent pas, mais ces valeurs sur 115 lignes sont les plus prudentes à citer provisoirement. Aucun score ne doit être présenté comme définitif avant la correction des cinq lignes.

## 7. Réponse absurde à la question `pd.head()`

La question `pd.head()` concerne **pandas**, qui n’est pas l’un des trois domaines documentaires explicitement indexés par ce projet (Python, Scikit-learn et LangChain). La réponse contenant une longue liste de noms est donc un exemple de question hors périmètre à laquelle le système n’aurait pas dû tenter de répondre. Une future amélioration recommandée est un mécanisme de refus : lorsque les scores de récupération sont trop faibles ou que la question ne correspond pas à un domaine indexé, l’interface devrait répondre qu’elle ne dispose pas d’une source fiable. Ce point est séparé de Ragas et ne modifie pas les résultats déjà calculés.

## 8. Comparaison avec la baseline : règle de rigueur

La baseline historique utilisait 20 questions et un évaluateur « Ragas-inspired » personnalisé. Elle avait produit Faithfulness = 0,789, Answer Relevancy = 0,727, Context Precision = 0,562 et Context Recall = 0,625. Le système final utilise 120 questions, des chunks de référence annotés, Ragas 0.4.3 officiel et un juge Mistral externe.

Par conséquent, il est incorrect d’écrire qu’un score final de 0,888 « améliore exactement » une baseline de 0,789 : les questions, le juge et l’implémentation métrique diffèrent. La formulation académique correcte est la suivante :

> La baseline a servi de diagnostic initial sur 20 questions avec un évaluateur personnalisé. Le système final est évalué sur un protocole plus robuste de 120 questions équilibrées, avec références et identifiants de contexte, en utilisant Ragas 0.4.3. Les résultats finaux décrivent donc la qualité du système amélioré dans un protocole plus exigeant, mais ne constituent pas une comparaison numérique stricte avant/après.

## 9. Recommandations ordonnées

La prochaine correction utile n’est pas de modifier le modèle ni de relancer toute la préparation. Il faut d’abord réparer les cinq lignes mal formées du jeu de test. Ensuite, un unique run Ragas permettra de geler le résultat final sur 120 questions. Après ce gel, la priorité d’amélioration technique doit être LangChain : enrichissement ciblé de ses sources, recherche par URL ou section, et mécanisme de refus quand aucun contexte fiable n’est trouvé.

Il est recommandé de conserver l’archive originale, le manifeste, `samples.jsonl`, `summary.json`, les graphiques produits par Kaggle et ce rapport. Ils constituent une trace de reproductibilité utile pour le rapport de stage.

## Références

[1] [Ragas — Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)

[2] [Ragas — Response Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevancy/)

[3] [Ragas — Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)

[4] [Ragas — Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)

[5] [Ragas — Factual Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)
