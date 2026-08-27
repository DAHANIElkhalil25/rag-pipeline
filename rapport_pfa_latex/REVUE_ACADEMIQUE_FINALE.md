# Évaluation académique finale — Rapport PFA RAG

## Position de l’évaluateur

Cette revue est une appréciation technique et rédactionnelle indépendante, fondée sur le code et les artefacts disponibles dans le dépôt, les résultats Ragas archivés, les sources bibliographiques intégrées et le PDF compilé. Elle ne constitue pas une note officielle de l’INSEA et ne remplace pas la validation par l’encadrant ou le jury.

## Grille de lecture

| Critère évalué | Appréciation | Justification |
|---|---|---|
| Adéquation au sujet de PFA | Très satisfaisant | Le rapport présente un projet appliqué complet : besoin, conception, réalisation, interface et évaluation. |
| Problématique et objectifs | Très satisfaisant | Le périmètre Python, Scikit-learn et LangChain est explicite ; la valeur ajoutée de la traçabilité documentaire est bien formulée. |
| Fondements théoriques | Très satisfaisant | Le chapitre distingue désormais LLM, RAG, chunks, embeddings, BM25, recherche hybride, cross-encoder, hallucination et évaluation. Les équations restent simples et utiles. |
| Justification des choix techniques | Satisfaisant à très satisfaisant | E5, FAISS, BM25, BGE, Mistral, Ragas et Gradio sont présentés avec leur rôle et des références. Un benchmark d’ablation plus complet serait une amélioration future, non un défaut bloquant de PFA. |
| Rigueur de l’évaluation | Très satisfaisant avec réserve | Les résultats de 120 questions sont détaillés ; la couverture, les six évaluations partielles et la sensibilité à 115 lignes sont déclarées. |
| Usage de la baseline | Très satisfaisant | L’artefact, le commit, le rôle diagnostique, les mesures et la non-comparabilité sont explicités. Le rapport ne revendique pas une amélioration numérique abusive. |
| Traçabilité et reproductibilité | Très satisfaisant | Le rapport décrit le dépôt, le notebook, les manifests, les IDs de chunks, les sorties par question et l’archive de résultats. |
| Analyse critique | Très satisfaisant | Le point faible LangChain, les cinq lignes mal formées, le juge LLM externe et la validation à relecteur unique sont discutés sans les masquer. |
| Bibliographie | Très satisfaisant | Les travaux fondateurs et la documentation officielle sont combinés avec les artefacts internes indispensables à la baseline et à l’organisme. |
| Qualité rédactionnelle et mise en page | Très satisfaisant | La structure est cohérente, les tableaux sont lisibles, les citations numériques sont homogènes et le PDF A4 de 45 pages compile sans erreur ni débordement signalé. |

## Forces déterminantes

Le rapport ne se limite pas à décrire des bibliothèques. Il explique la chaîne causale entre la qualité du corpus, la récupération, le contexte transmis au LLM, la génération et les métriques observées. Cette approche est particulièrement importante pour un PFA sur les systèmes RAG, car elle démontre une compréhension méthodologique au-delà de l’assemblage technique.

La présentation de la baseline est désormais conforme à une démarche expérimentale rigoureuse. Elle n’est ni supprimée ni utilisée comme une preuve statistique incorrecte : son statut de diagnostic initial est explicite. La distinction entre le générateur local Mistral-7B-Instruct et le juge externe Mistral est également une bonne pratique qui évite de confondre deux rôles différents.

L’analyse des résultats est prudente. Les valeurs globales sont interprétées sans sur-promesse : la fidélité et la pertinence sont encourageantes, mais l’exactitude factuelle et la récupération LangChain restent des axes de travail. Le rapport identifie aussi le défaut du jeu de test, au lieu de présenter des scores provisoires comme définitifs.

## Réserves à connaître avant dépôt

| Priorité | Point à vérifier ou à corriger | Conséquence si non traité |
|---|---|---|
| Indispensable | Remplacer le placeholder de l’encadrant organisme par son nom et sa fonction exacts. | La page de garde reste incomplète administrativement. |
| Indispensable | Vérifier les deux noms des jurés INSEA et les conserver uniquement s’ils sont confirmés. | Le jury pourrait ne pas correspondre à la composition officielle. |
| Indispensable | Corriger les cinq lignes du jeu final et relancer une seule évaluation Ragas avant d’annoncer les scores comme définitifs. | Les scores restent correctement qualifiés de provisoires dans le rapport actuel. |
| Recommandé | Faire relire le résumé, l’abstract et le chapitre organisme par l’encadrant. | Cela garantit l’exactitude des formulations institutionnelles et la terminologie préférée. |
| Recommandé | Ajouter les dates réelles de stage et de soutenance si l’établissement les demande. | Il s’agit d’une complétude de forme, non d’un problème scientifique. |

## Verdict professionnel

Dans son état révisé, le document est **un rapport PFA professionnel, cohérent et défendable devant un jury**, sous réserve de compléter les informations administratives et de conserver la qualification « provisoire » pour les résultats Ragas à 120 lignes. La qualité la plus forte du travail est sa transparence : les résultats sont exploitables, mais leurs limites sont précisées avec des conséquences méthodologiques claires.

L’amélioration la plus utile après le stage ne consiste pas à multiplier les exécutions Kaggle. Elle consiste à corriger le jeu de test, exécuter une unique fois le run final, puis remplacer dans le rapport la mention « provisoire » par les chiffres reproduits. Cette démarche respecte la simplicité demandée tout en renforçant réellement la valeur académique du projet.

## Contrôle visuel du PDF livré

Le PDF final de 45 pages a été contrôlé après compilation. Le tableau des résultats Ragas est lisible et la coupure volontaire de « Factual Correctness » évite un débordement horizontal. La bibliographie enrichie est lisible, numérotée de manière continue et associe correctement articles scientifiques, documentations officielles et artefacts internes du projet. La compilation finale ne signale ni erreur, ni citation non résolue, ni débordement de ligne.
