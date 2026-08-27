# Évaluation académique experte — version initiale du rapport PFA

## Cadre de relecture

Cette évaluation a été conduite à partir du fichier `rapport_PFA_RAG.tex`, des résultats archivés de l’exécution Ragas finale, de la baseline `baseline_metrics_v1.json`, du code de la baseline historique et des deux visuels transmis sur 3D Smart Factory. Elle ne remplace pas la validation administrative ou pédagogique de l’encadrant INSEA, mais elle vérifie la cohérence scientifique, méthodologique, documentaire et rédactionnelle du rapport.

## Appréciation synthétique avant révision

| Critère | Appréciation initiale | Évaluation | Amélioration prévue |
|---|---|---|---|
| Problématique et objectifs | Bon | Le besoin et le périmètre sont clairs. | Séparer plus nettement contexte, besoin et question de recherche. |
| Fondements théoriques | Moyen | Les notions clés sont présentes, mais trop condensées et peu formalisées. | Créer un chapitre théorique plus structuré : LLM, RAG, chunking, dense, BM25, hybridation, reranking, hallucination et évaluation. |
| Organisme d’accueil | À renforcer | Le texte est prudent, mais l’information était partiellement générique et les captures figuraient dans le PDF. | Retirer les captures ; utiliser seulement le contenu vérifié : Mohammédia, Lotissement La Gare, accompagnement intégré, Industrie 4.0, projets technologiques et chaîne de valeur. |
| Choix techniques | Bon | Les composants réels du pipeline sont identifiés et cohérents. | Relier chaque choix à une référence et aux contraintes françaises/anglaises du corpus. |
| Baseline | Insuffisant | Son rôle est évoqué, mais son identité, ses paramètres, ses mesures et sa non-comparabilité sont insuffisamment justifiés. | Ajouter une section dédiée fondée sur l’artefact `baseline_metrics_v1.json` et citer cet artefact dans la bibliographie. |
| Évaluation Ragas | Bon | Les scores, la couverture et les limites sont rapportés de manière honnête. | Séparer clairement protocole, résultats, interprétation et analyse de sensibilité. |
| Bibliographie | À renforcer | Les références essentielles existent, mais l’ensemble est court pour un rapport académique complet. | Ajouter des articles fondateurs et des sources officielles : DPR, revues RAG, revue d’évaluation, Transformers, Gradio, quantification, outils et corpus. |
| Transparence des limites | Très bon | Les cinq lignes mal formées, les six évaluations partielles et le juge LLM sont explicitement signalés. | Maintenir ces réserves dans le résumé, les résultats et la conclusion. |
| Rédaction et mise en page | Bon | Le français, la hiérarchie et la mise en page sont cohérents ; l’écriture doit devenir plus analytique. | Réduire les formulations répétitives, ajouter des transitions, des définitions, des tableaux synthétiques et des renvois croisés. |

## Verdict professionnel

La version initiale constitue une bonne base de projet appliqué : elle est compilable, traçable et honnête sur les résultats. Pour atteindre le niveau attendu d’un rapport PFA professionnel, elle doit cependant approfondir sa démonstration scientifique. La révision doit montrer non seulement **ce qui a été développé**, mais aussi **pourquoi chaque choix a été fait**, **ce que les résultats permettent réellement de conclure**, et **ce qui reste limité**.

## Principes de correction appliqués

1. Aucun résultat ne sera présenté comme définitif tant que les cinq lignes mal formées ne sont pas corrigées et que le run final n’est pas reproduit.
2. La baseline restera un diagnostic initial, documenté et archivé ; elle ne sera pas utilisée pour revendiquer une amélioration numérique directe.
3. Toute affirmation technique significative sera reliée à une source académique ou à une documentation officielle.
4. Les informations relatives à 3D Smart Factory seront limitées aux éléments explicitement présents dans les supports remis ; aucun chiffre, statut juridique, effectif ou responsable non fourni ne sera ajouté.
5. Les visuels de 3D Smart Factory ne seront pas reproduits dans le PDF final, conformément à l’instruction de l’étudiant.

## Contrôle visuel de la version révisée — constats à intégrer

La page de garde est lisible, identifie clairement le document comme un **PFA**, et comporte uniquement les deux membres du jury INSEA demandés. La hiérarchie institutionnelle, la police Times et l’espacement général sont cohérents avec le rapport de référence. Deux corrections de finition sont toutefois nécessaires : le titre se coupe de façon peu élégante dans le mot « interroga-tion » et la mention « Responsable de stage – 3D Smart Factory » doit demeurer un placeholder visible tant que le nom n’est pas confirmé.

Une page intérieure contrôlée confirme la lisibilité du corps du texte, des tableaux et de l’en-tête de chapitre. Le contenu demeure aéré et adapté au format A4. Pour un rapport français entièrement cohérent, le libellé automatique « Table » doit être remplacé par « Tableau » dans les légendes. Ces ajustements sont des améliorations de qualité éditoriale ; aucun problème de contenu ou de débordement visible n’a été constaté sur les pages contrôlées.

## Contrôle visuel final de la version enrichie

Les corrections de finition ont été vérifiées. Le titre de couverture est désormais réparti sur trois lignes sans césure inesthétique. La nature PFA, l’année universitaire et le jury à deux membres INSEA sont visibles. Le placeholder de l’encadrant de l’organisme est clairement signalé, conformément au principe de non-invention.

La page contrôlée du chapitre théorique présente une hiérarchie claire, un texte aéré, des citations numériques et une formule de similarité lisible. L’en-tête, le pied de page et la composition restent cohérents avec le style retenu. Aucune capture d’écran de 3D Smart Factory n’apparaît dans le document final ; les informations exploitables ont été reformulées dans le texte et associées à une source interne explicitement identifiée.

Les pages de résultats Ragas et de positionnement de la baseline ont également été contrôlées visuellement. La présentation des résultats est lisible, les tableaux sont correctement légendés en français et la limitation sur les cinq lignes mal formées est explicite. La section baseline explique clairement son statut de diagnostic historique et interdit toute comparaison numérique abusive. Le seul défaut visuel identifié est la césure du terme « Factual Correctness » dans le tableau global ; il est corrigé dans la source par un retour à la ligne volontaire.
