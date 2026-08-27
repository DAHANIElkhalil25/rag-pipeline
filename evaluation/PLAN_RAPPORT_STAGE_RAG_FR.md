# Plan de rapport de stage — Assistant RAG pour documentation technique

## Positionnement du rapport

Ce plan convient à un rapport de stage académique et professionnel consacré à la conception, l’implémentation et l’évaluation d’un assistant de questions-réponses fondé sur la génération augmentée par récupération (*Retrieval-Augmented Generation*, RAG). Le système exploite une documentation technique en trois domaines : **Python**, **Scikit-learn** et **LangChain**.

Le fil conducteur du rapport doit rester simple : un utilisateur pose une question technique ; le système recherche d’abord des passages fiables dans la documentation ; puis un modèle de langage produit une réponse à partir de ces passages. Le rapport ne doit pas seulement présenter un résultat : il doit démontrer une démarche d’ingénierie, d’évaluation et d’amélioration progressive.

> **Fil rouge recommandé :** « Comment concevoir et évaluer un assistant RAG capable de répondre de manière pertinente, fondée sur des sources et reproductible à partir d’une documentation technique hétérogène ? »

## Éléments avant le corps du rapport

| Élément | Contenu attendu |
|---|---|
| Page de garde | Établissement, filière, titre du projet, entreprise/organisme, encadrants, auteur, période du stage et année universitaire. |
| Remerciements | Courts, professionnels et personnalisés. |
| Résumé en français | Problème, méthode RAG, technologies, résultats principaux et limites ; environ une demi-page. |
| Abstract en anglais | Traduction fidèle du résumé français. |
| Mots-clés | RAG, LLM, recherche sémantique, FAISS, reranking, Ragas, documentation technique. |
| Listes | Table des matières, liste des figures, liste des tableaux, liste des abréviations. |

## Introduction générale

L’introduction présente l’essor des modèles de langage, leur intérêt pour l’accès à l’information technique et leur limite principale : ils peuvent générer une réponse plausible sans preuve documentaire. Elle introduit le RAG comme une réponse à ce risque, car le modèle est alimenté par des passages récupérés dans un corpus contrôlé.

Cette partie doit ensuite présenter le contexte du stage, la problématique, les objectifs, la méthode de travail et la structure du rapport. Elle doit annoncer que le projet ne cherche pas à construire un assistant généraliste, mais un assistant spécialisé dans un corpus documentaire identifié.

| Sous-section | Question à traiter |
|---|---|
| Contexte | Pourquoi un assistant documentaire est-il utile dans l’organisme ou pour les développeurs ? |
| Problématique | Comment produire des réponses techniques pertinentes et appuyées par des sources ? |
| Objectifs | Concevoir le pipeline, créer une interface, évaluer le système et analyser ses limites. |
| Contribution | Pipeline RAG reproductible, récupération hybride, reranking, jeu de test structuré et évaluation Ragas. |

# Partie I — Contexte du stage et analyse du besoin

## Chapitre 1 — Présentation de l’organisme d’accueil

Présentez l’entreprise ou l’organisme, son activité, son organisation, son service d’accueil et la place de votre stage. Ne transformez pas ce chapitre en historique très long : l’objectif est de montrer le lien entre l’environnement professionnel et le besoin traité.

## Chapitre 2 — Étude du besoin et cahier des charges

Expliquez le besoin des utilisateurs : accéder rapidement à une information technique fiable sans parcourir manuellement de longues pages de documentation. Définissez les utilisateurs visés, les trois domaines documentaires, les questions couvertes et les limites du périmètre.

| Catégorie | Contenu à écrire |
|---|---|
| Exigences fonctionnelles | Poser une question, récupérer des passages, générer une réponse, afficher les sources et utiliser une interface web. |
| Exigences non fonctionnelles | Reproductibilité, rapidité raisonnable, traçabilité des chunks, exécution Kaggle et confidentialité des clés API. |
| Hors périmètre | Documentation pandas et autres domaines non indexés ; réponses sans source suffisante. |
| Critères de succès | Réponses pertinentes, contexte utile, fidélité au contexte et évaluation quantitative sauvegardée. |

# Partie II — Cadre théorique et étude de l’existant

## Chapitre 3 — Concepts fondamentaux

Présentez progressivement les notions nécessaires : modèle de langage (*LLM*), embeddings, similarité vectorielle, base vectorielle, découpage en chunks, recherche lexicale BM25, recherche dense, recherche hybride, reranking et hallucination. Pour chaque notion, donnez une définition courte puis reliez-la à votre système.

Un schéma pédagogique doit représenter le principe RAG : **question utilisateur → récupération de documents → sélection/reranking → contexte → modèle génératif → réponse avec sources**.

## Chapitre 4 — Approches d’évaluation des systèmes RAG

Expliquez pourquoi la simple impression subjective d’une réponse ne suffit pas. Présentez les mesures utilisées : *Faithfulness*, *Answer Relevancy*, *Context Precision*, *Context Recall* et *Factual Correctness*. Précisez que certaines évaluations reposent sur un modèle juge ; dans ce projet, le juge final est Mistral.

| Mesure | Ce qu’elle vérifie dans ce projet |
|---|---|
| Faithfulness | La réponse est-elle soutenue par les passages remis au générateur ? |
| Answer Relevancy | La réponse traite-t-elle réellement la question de l’utilisateur ? |
| Context Precision | Les passages récupérés sont-ils utiles pour répondre ? |
| Context Recall | Les passages récupérés couvrent-ils les informations nécessaires ? |
| Factual Correctness | La réponse est-elle cohérente avec la réponse de référence ? |

# Partie III — Conception et réalisation du système RAG

## Chapitre 5 — Architecture générale et choix techniques

Présentez l’architecture sous forme d’un diagramme clair. Justifiez chaque choix important : Python pour l’implémentation, FAISS pour la recherche vectorielle, `intfloat/multilingual-e5-base` pour relier questions françaises et documentation majoritairement anglaise, BM25 pour les termes exacts, et `BAAI/bge-reranker-v2-m3` pour réordonner les candidats.

| Étape du pipeline | Mise en œuvre du projet | Rôle |
|---|---|---|
| Collecte | Documentation officielle Python, Scikit-learn et LangChain | Constituer un corpus contrôlé. |
| Nettoyage | Suppression de bruit, dédoublonnage et métadonnées de provenance | Améliorer la qualité du corpus. |
| Chunking | Découpage avec `chunk_id`, URL, document et empreinte | Conserver une preuve traçable. |
| Indexation | FAISS + embeddings E5 multilingues | Retrouver les passages sémantiquement proches. |
| Recherche | Hybride BM25/dense, top 20 candidats | Combiner mots-clés et sens. |
| Reranking | BGE reranker, top 5 final | Retenir les passages les plus utiles. |
| Génération | Mistral-7B-Instruct | Rédiger une réponse à partir du contexte. |
| Interface | Gradio | Permettre les questions utilisateur. |

## Chapitre 6 — Implémentation et exécution dans Kaggle

Décrivez les étapes de réalisation sans recopier tout le code : organisation des scripts, gestion de la configuration, chargement des modèles, GPU Kaggle, secrets API et sauvegarde des résultats. Insérez seulement de courts extraits de code essentiels : format d’un chunk, appel de récupération ou sauvegarde d’un résultat.

Expliquez aussi l’interface : champ de question, réponse générée, sources récupérées et comportement attendu lorsque le système ne possède pas de source suffisamment fiable. La question `pd.head()` peut être présentée comme un exemple de **question hors périmètre** : pandas n’est pas dans le corpus et le système devrait le signaler au lieu de répondre sans source.

# Partie IV — Protocole expérimental, résultats et discussion

## Chapitre 7 — Constitution du jeu de test et protocole d’évaluation

Présentez d’abord la baseline historique : 20 questions et évaluateur personnalisé inspiré de Ragas. Elle sert de diagnostic initial, mais pas de comparaison numérique stricte avec l’étude finale.

Présentez ensuite le protocole final : 120 questions françaises, 40 par domaine, réponses de référence, URLs de documentation, identifiants de chunks et validation experte assistée par IA. Indiquez explicitement que cette validation n’est pas une annotation indépendante par plusieurs évaluateurs humains.

| Niveau d’évaluation | Taille | Outil | Rôle dans le rapport |
|---|---:|---|---|
| Baseline initiale | 20 questions | Évaluateur personnalisé | Diagnostic et identification des premiers problèmes. |
| Évaluation finale | 120 questions | Ragas 0.4.3 + juge Mistral | Résultat principal, plus équilibré et plus traçable. |

## Chapitre 8 — Présentation et interprétation des résultats

Présentez les résultats globaux, puis les résultats par domaine. Utilisez les graphiques générés par Kaggle et les tableaux issus de `summary.json`. Insistez davantage sur l’interprétation que sur les nombres seuls.

| Résultat Ragas actuellement observé | Valeur globale | Message à retenir |
|---|---:|---|
| Faithfulness | 0,888 | Les réponses sont généralement ancrées dans le contexte récupéré. |
| Answer Relevancy | 0,913 | Les réponses sont le plus souvent liées à la question. |
| Context Precision | 0,819 | Les contextes sont souvent utiles, avec une faiblesse plus visible pour LangChain. |
| Context Recall | 0,879 | Les informations nécessaires sont fréquemment présentes dans le top 5. |
| Factual Correctness | 0,693 | L’exactitude par rapport à la référence est l’axe d’amélioration principal. |

La discussion doit mettre en évidence trois constats : Python est globalement solide ; Scikit-learn obtient la meilleure couverture documentaire ; LangChain est plus difficile, avec un Hit@5 de 0,650 contre 0,925 pour Scikit-learn. Illustrez ensuite deux ou trois cas réels : une bonne réponse sourcée, une réponse faible et une question hors périmètre.

## Chapitre 9 — Limites, difficultés et améliorations proposées

Cette partie est essentielle pour donner un caractère professionnel au rapport. Présentez les difficultés rencontrées sans les cacher : compatibilités Ragas, client asynchrone, quota OpenAI, passage au juge Mistral, réponses incomplètes du juge, cinq lignes mal formées dans le dataset final et limites de LangChain.

| Limite observée | Conséquence | Amélioration proposée |
|---|---|---|
| Cinq lignes de test mal formées | Les scores à 120 lignes restent provisoires. | Corriger les cinq questions/références, puis refaire une unique exécution finale. |
| LangChain moins bien récupéré | 14 références LangChain absentes du top 5. | Enrichir les sources, filtrer par page/section et améliorer le reranking ciblé. |
| Questions hors corpus | Risque de réponse inventée. | Ajouter un seuil de confiance et un message de refus avec explication du périmètre. |
| Évaluation par un seul juge | Risque de variabilité de jugement. | Répéter une partie de l’étude avec un second juge ou une revue humaine. |
| Coût/quota API | Le juge externe peut être indisponible. | Prévoir un fournisseur alternatif et archiver les artefacts de chaque run. |

# Conclusion générale et perspectives

La conclusion répond directement à la problématique. Résumez la contribution : assistant RAG spécialisé, architecture hybride avec reranking, interface utilisable, protocole d’évaluation à 120 questions et résultats montrant une bonne pertinence tout en révélant des faiblesses précises.

Les perspectives doivent être réalistes : correction du dataset, mécanisme de refus hors périmètre, amélioration du corpus LangChain, citations de sources dans l’interface, tests utilisateurs et comparaison avec un second juge ou des annotations humaines.

## Bibliographie

Adoptez le style demandé par votre établissement (APA, IEEE, ISO 690 ou autre) et restez cohérent. Les références doivent inclure la documentation officielle Python, Scikit-learn, LangChain, FAISS, SentenceTransformers, Ragas, Mistral, ainsi que les articles scientifiques utilisés pour justifier le RAG et l’évaluation.

## Annexes recommandées

| Annexe | Contenu |
|---|---|
| A | Diagramme complet de l’architecture RAG. |
| B | Exemple d’un document nettoyé, d’un chunk et de ses métadonnées. |
| C | Schéma du jeu de test de 120 questions et exemple d’enregistrement JSONL. |
| D | Paramètres de récupération : E5, BM25, alpha, top 20, top 5 et reranker. |
| E | Tableaux Ragas complets par domaine et graphiques Kaggle. |
| F | Exemples de bonnes réponses, d’erreurs et de questions hors périmètre. |
| G | Liens vers GitHub, notebook Kaggle et guide d’exécution ; aucune clé API ne doit apparaître. |

## Règles de rédaction à respecter

Rédigez au passé pour les actions effectuées (« nous avons collecté », « le système a été évalué »), restez précis sur les données réellement obtenues et évitez les formulations absolues telles que « le système est parfait ». Distinguez toujours les résultats observés, les interprétations et les améliorations futures.

Ne comparez pas numériquement la baseline de 20 questions et l’évaluation Ragas finale de 120 questions comme s’il s’agissait du même protocole. La formulation correcte est que la baseline a servi de diagnostic initial, tandis que le protocole final fournit une évaluation plus robuste, plus large et mieux documentée.
