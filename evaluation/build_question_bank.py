"""Build the 120-item final-test draft from primary documentation topics.

The file deliberately marks every record as source-grounded draft material.
Human review and Kaggle-derived reference chunk IDs are still required before
``convert_annotations.py`` will create the frozen final test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_schema import write_jsonl


PY_GLOSSARY = "https://docs.python.org/3/glossary.html"
PY_CONTEXT = "https://docs.python.org/3/library/contextlib.html"
PY_TUTORIAL = "https://docs.python.org/3/tutorial/"
SK_PIPE = "https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html"
SK_PITFALLS = "https://scikit-learn.org/stable/common_pitfalls.html"
SK_PREPROCESS = "https://scikit-learn.org/stable/modules/preprocessing.html"
SK_MODEL_SELECTION = "https://scikit-learn.org/stable/model_selection.html"
LC_OVERVIEW = "https://docs.langchain.com/oss/python/langchain/overview"
LC_RETRIEVERS = "https://docs.langchain.com/oss/python/integrations/retrievers"
LC_DOCS = "https://docs.langchain.com/oss/python/langchain/retrieval"


PYTHON_ITEMS = [
    ("Qu'est-ce qu'un décorateur Python ?", "Un décorateur est une fonction qui renvoie une autre fonction et transforme habituellement une fonction avec la syntaxe @.", PY_GLOSSARY),
    ("Quelle différence y a-t-il entre un itérable et un itérateur ?", "Un itérable peut fournir un itérateur ; un itérateur fournit les éléments successifs et signale la fin par StopIteration.", PY_GLOSSARY),
    ("Quel est le rôle de yield dans une fonction ?", "yield produit une valeur et suspend l'exécution ; l'appel de la fonction crée alors un générateur qui peut reprendre ensuite.", PY_GLOSSARY),
    ("Que signifie EAFP en Python ?", "EAFP signifie préférer essayer l'opération puis gérer l'exception plutôt que vérifier toutes les conditions avant l'opération.", PY_GLOSSARY),
    ("À quoi sert le mot-clé with ?", "with exécute un bloc sous le contrôle d'un gestionnaire de contexte afin d'assurer l'entrée et le nettoyage à la sortie.", PY_CONTEXT),
    ("Que doit fournir un gestionnaire de contexte synchrone ?", "Un gestionnaire de contexte synchronisé met en œuvre le protocole avec __enter__ et __exit__.", PY_CONTEXT),
    ("Quel est le rôle d'un bloc finally ?", "finally exécute son bloc lors de la sortie du try, qu'une exception ait été levée ou non, ce qui le rend adapté au nettoyage.", PY_TUTORIAL + "errors.html"),
    ("Quelle est la différence entre raise et except ?", "raise déclenche ou relance une exception ; except intercepte une exception correspondant à un type dans un bloc try.", PY_TUTORIAL + "errors.html"),
    ("Pourquoi créer une exception personnalisée ?", "Une exception personnalisée exprime une erreur propre au domaine et permet au code appelant de la gérer de façon précise.", PY_TUTORIAL + "errors.html"),
    ("Quel est le rôle de __init__ dans une classe ?", "__init__ est une méthode spéciale appelée après la création d'une instance pour initialiser son état.", PY_TUTORIAL + "classes.html"),
    ("Qu'appelle-t-on héritage de classe ?", "L'héritage permet à une classe dérivée de réutiliser et spécialiser les attributs et méthodes d'une classe de base.", PY_TUTORIAL + "classes.html"),
    ("À quoi sert super() dans une méthode ?", "super() donne accès à l'implémentation de la superclasse selon l'ordre de résolution des méthodes.", PY_TUTORIAL + "classes.html"),
    ("Quelle différence y a-t-il entre == et is ?", "== compare l'égalité de valeur selon l'objet ; is vérifie si deux références désignent le même objet.", PY_GLOSSARY),
    ("Pourquoi une liste mutable comme valeur par défaut est-elle risquée ?", "La valeur par défaut est créée une seule fois ; une liste mutable peut donc être partagée entre plusieurs appels.", PY_TUTORIAL + "controlflow.html"),
    ("Quelle différence entre une copie superficielle et une copie profonde ?", "Une copie superficielle réutilise les objets imbriqués ; une copie profonde reconstruit aussi les objets contenus.", "https://docs.python.org/3/library/copy.html"),
    ("À quoi sert un module Python ?", "Un module est un fichier contenant des définitions et instructions Python qui peut être importé et réutilisé.", PY_TUTORIAL + "modules.html"),
    ("Qu'est-ce qu'un package Python ?", "Un package organise des modules dans un espace de noms hiérarchique, généralement représenté par un répertoire.", PY_TUTORIAL + "modules.html"),
    ("Quelle différence entre import module et from module import name ?", "import module conserve le nom du module ; from module import name introduit directement le nom importé dans l'espace courant.", PY_TUTORIAL + "modules.html"),
    ("Que fait async def ?", "async def définit une fonction coroutine dont l'appel produit un objet coroutine à exécuter dans un environnement asynchrone.", PY_GLOSSARY),
    ("Quel est le rôle de await ?", "await suspend une coroutine jusqu'à l'achèvement d'un objet awaitable puis reprend avec son résultat.", PY_GLOSSARY),
    ("Qu'est-ce qu'un générateur asynchrone ?", "Un générateur asynchrone est défini avec async def et yield ; il produit des valeurs consommables avec async for.", PY_GLOSSARY),
    ("Pourquoi utiliser pathlib plutôt que concaténer des chaînes de chemins ?", "pathlib fournit des objets Path et des opérations de chemin portables plutôt que de manipuler manuellement des séparateurs de chaînes.", "https://docs.python.org/3/library/pathlib.html"),
    ("Quelle différence entre open en mode r, w et a ?", "r lit un fichier existant, w écrit en tronquant ou créant le fichier, et a écrit à la fin en créant le fichier si nécessaire.", PY_TUTORIAL + "inputoutput.html"),
    ("Pourquoi préciser encoding lors de l'ouverture d'un fichier texte ?", "encoding définit comment convertir les octets et les chaînes ; le préciser évite de dépendre d'un encodage par défaut de plateforme.", PY_TUTORIAL + "inputoutput.html"),
    ("Quel est l'intérêt d'une compréhension de liste ?", "Une compréhension de liste construit une liste de façon concise à partir d'un itérable et peut inclure une condition.", PY_TUTORIAL + "datastructures.html"),
    ("À quoi sert enumerate dans une boucle ?", "enumerate fournit simultanément l'indice et la valeur de chaque élément d'un itérable.", PY_TUTORIAL + "datastructures.html"),
    ("À quoi sert zip ?", "zip agrège des itérables et produit des tuples constitués des éléments de même position.", PY_TUTORIAL + "datastructures.html"),
    ("Quelle différence entre *args et **kwargs ?", "*args collecte des arguments positionnels supplémentaires ; **kwargs collecte des arguments nommés supplémentaires dans un dictionnaire.", PY_TUTORIAL + "controlflow.html"),
    ("Quel est le rôle des annotations de type ?", "Les annotations associent par convention des informations de type à des variables, paramètres et retours ; elles n'imposent pas elles-mêmes un contrôle à l'exécution.", PY_GLOSSARY),
    ("À quoi sert @property ?", "@property permet d'exposer une méthode comme un attribut et de contrôler son accès, par exemple pour calculer ou valider une valeur.", PY_TUTORIAL + "classes.html"),
    ("Quand utiliser dataclasses.dataclass ?", "@dataclass génère notamment des méthodes comme __init__ et __repr__ pour des classes principalement destinées à stocker des données.", "https://docs.python.org/3/library/dataclasses.html"),
    ("À quoi sert contextlib.suppress ?", "contextlib.suppress ignore uniquement les exceptions spécifiées dans un bloc with et doit être réservé aux erreurs attendues.", PY_CONTEXT),
    ("Quel problème résout contextlib.ExitStack ?", "ExitStack permet de composer dynamiquement plusieurs gestionnaires de contexte et callbacks de nettoyage dans une seule portée.", PY_CONTEXT),
    ("Quelle différence entre une fonction et un callable ?", "Une fonction est callable ; une instance qui définit __call__ est aussi callable sans être nécessairement une fonction.", PY_GLOSSARY),
    ("Pourquoi utiliser logging plutôt que print dans une application ?", "logging permet de structurer les niveaux, destinations et formats des messages sans mélanger diagnostic et sortie fonctionnelle.", "https://docs.python.org/3/library/logging.html"),
    ("Quel est le rôle de json.dumps et json.loads ?", "json.dumps sérialise un objet Python en texte JSON ; json.loads désérialise un texte JSON en objet Python.", "https://docs.python.org/3/library/json.html"),
    ("Pourquoi un générateur est-il utile pour un grand flux de données ?", "Un générateur produit les éléments à la demande et évite de matérialiser toute la collection en mémoire.", PY_GLOSSARY),
    ("Que signifie la portée d'une variable ?", "La portée définit les régions du programme où un nom peut être résolu, notamment locale, englobante, globale ou intégrée.", PY_GLOSSARY),
    ("Quel est le risque d'utiliser une exception trop générale comme Exception ?", "Intercepter une exception très générale peut masquer des erreurs inattendues ; il est préférable d'intercepter le type d'erreur attendu.", PY_TUTORIAL + "errors.html"),
    ("Pourquoi un contexte with est-il préférable pour un fichier ?", "Le gestionnaire de contexte garantit que le fichier est fermé à la sortie du bloc, y compris lorsqu'une erreur survient.", PY_CONTEXT),
]

SKLEARN_ITEMS = [
    ("Qu'est-ce qu'un Pipeline scikit-learn ?", "Pipeline chaîne des transformateurs puis éventuellement un prédicteur final afin de les ajuster et les valider ensemble.", SK_PIPE),
    ("Quelle interface doivent fournir les étapes intermédiaires d'un Pipeline ?", "Les étapes intermédiaires doivent implémenter fit et transform ; l'estimateur final doit au minimum implémenter fit.", SK_PIPE),
    ("Comment régler C d'un SVC dans un Pipeline nommé svc ?", "Utilisez le nom de l'étape puis deux underscores, par exemple svc__C, avec set_params ou une recherche d'hyperparamètres.", SK_PIPE),
    ("Pourquoi utiliser un Pipeline pendant la validation croisée ?", "Le Pipeline applique les transformations dans chaque pli avec les données d'entraînement correspondantes et réduit le risque de fuite de données.", SK_PITFALLS),
    ("Qu'est-ce qu'une fuite de données ?", "Une fuite survient lorsqu'une information indisponible au moment de prédire influence la construction du modèle et rend les scores trop optimistes.", SK_PITFALLS),
    ("Quelle règle suivre pour fit sur les données de test ?", "Ne jamais appeler fit ou fit_transform sur les données de test ; les transformations sont apprises sur l'entraînement puis appliquées au test avec transform.", SK_PITFALLS),
    ("Quel est le rôle de StandardScaler ?", "StandardScaler centre les caractéristiques puis les met à l'échelle selon leur écart-type appris sur les données d'entraînement.", SK_PREPROCESS),
    ("Pourquoi la standardisation peut-elle aider un modèle linéaire ou un SVM ?", "Des variables sur des échelles très différentes peuvent dominer la fonction objectif ; la standardisation rend leurs échelles comparables.", SK_PREPROCESS),
    ("Quel scaler privilégier en présence de nombreuses valeurs aberrantes ?", "RobustScaler peut être préférable car il utilise des estimations plus robustes du centre et de l'étendue.", SK_PREPROCESS),
    ("Pourquoi StandardScaler avec centrage est-il problématique sur une matrice sparse ?", "Le centrage détruit la sparsité et peut provoquer une allocation mémoire excessive ; utilisez with_mean=False ou MaxAbsScaler.", SK_PREPROCESS),
    ("Quelle différence entre MinMaxScaler et StandardScaler ?", "MinMaxScaler projette les caractéristiques dans une plage donnée, alors que StandardScaler centre et met à l'échelle par l'écart-type.", SK_PREPROCESS),
    ("À quoi sert OneHotEncoder ?", "OneHotEncoder transforme chaque catégorie en variables binaires, évitant d'interpréter arbitrairement les catégories comme ordonnées.", SK_PREPROCESS),
    ("Pourquoi OrdinalEncoder peut-il être inadapté à une catégorie nominale ?", "Il attribue des nombres aux catégories ; un estimateur peut interpréter ces nombres comme une relation d'ordre non voulue.", SK_PREPROCESS),
    ("Quand utiliser cross_val_score ?", "cross_val_score évalue un estimateur selon une stratégie de validation croisée et renvoie un score par pli.", SK_MODEL_SELECTION),
    ("Que mesure la validation croisée ?", "Elle estime la performance de généralisation en entraînant et évaluant l'estimateur sur plusieurs partitions des données.", SK_MODEL_SELECTION),
    ("Quelle différence entre GridSearchCV et RandomizedSearchCV ?", "GridSearchCV explore exhaustivement une grille définie ; RandomizedSearchCV échantillonne un nombre fixé de configurations selon des distributions.", SK_MODEL_SELECTION),
    ("Pourquoi séparer entraînement et test avant le prétraitement ?", "Sinon le prétraitement apprend des informations des données de test, ce qui crée une fuite et biaise l'évaluation.", SK_PITFALLS),
    ("Quel est le rôle de transform après fit pour un scaler ?", "transform applique aux nouvelles données les paramètres appris par fit sans les réestimer.", SK_PREPROCESS),
    ("Que fait fit_transform ?", "fit_transform apprend les paramètres du transformateur sur les données fournies puis applique immédiatement la transformation.", SK_PIPE),
    ("Pourquoi le dernier estimateur d'un Pipeline n'a-t-il pas besoin de transform ?", "Le dernier élément peut être un prédicteur ; il reçoit les données transformées par les étapes précédentes et doit pouvoir être ajusté.", SK_PIPE),
    ("À quoi sert train_test_split ?", "train_test_split divise des tableaux ou matrices en sous-ensembles d'entraînement et de test.", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"),
    ("Quel est le rôle de random_state ?", "random_state contrôle l'aléa des composants qui le supportent et peut aider à reproduire une expérience lorsqu'il est fixé de manière cohérente.", SK_PITFALLS),
    ("Pourquoi stratifier une séparation de classification ?", "La stratification aide à conserver la proportion des classes entre les sous-ensembles lorsqu'elle est appropriée.", "https://scikit-learn.org/stable/modules/cross_validation.html"),
    ("Qu'est-ce qu'une métrique de scoring dans scikit-learn ?", "Une métrique ou fonction de scoring quantifie la qualité des prédictions selon la tâche et l'objectif choisis.", SK_MODEL_SELECTION),
    ("Pourquoi choisir la métrique avant de régler les hyperparamètres ?", "La recherche optimise le score fourni ; la métrique doit donc représenter l'objectif réel de l'application.", SK_MODEL_SELECTION),
    ("Quel risque y a-t-il à sélectionner des variables sur toutes les données avant une séparation ?", "La sélection voit les données de test et crée une fuite, ce qui peut gonfler artificiellement les performances mesurées.", SK_PITFALLS),
    ("Quel est l'intérêt de SimpleImputer ?", "SimpleImputer remplace les valeurs manquantes selon une stratégie simple, par exemple moyenne, médiane ou constante.", "https://scikit-learn.org/stable/modules/impute.html"),
    ("Pourquoi placer SimpleImputer dans un Pipeline ?", "L'imputation est alors ajustée uniquement sur chaque partie d'entraînement et appliquée de façon cohérente aux données de validation ou test.", SK_PITFALLS),
    ("Quelle différence entre predict et predict_proba ?", "predict retourne les prédictions de classe ou de cible ; predict_proba retourne des probabilités de classe lorsque l'estimateur le prend en charge.", SK_PIPE),
    ("Que fait get_params(deep=True) sur un Pipeline ?", "Il retourne les paramètres du Pipeline et, avec deep=True, ceux des estimateurs contenus.", SK_PIPE),
    ("Comment supprimer une étape optionnelle d'un Pipeline ?", "Une étape peut être remplacée par 'passthrough' ou None selon l'API du Pipeline.", SK_PIPE),
    ("Pourquoi réutiliser le même Pipeline en production ?", "Il garantit que les transformations appliquées pendant l'entraînement sont également appliquées avant la prédiction en production.", SK_PITFALLS),
    ("Qu'est-ce que l'optimisation d'hyperparamètres ?", "C'est la recherche de valeurs de paramètres de modèle choisies avant l'apprentissage afin d'optimiser un score de validation.", SK_MODEL_SELECTION),
    ("Pourquoi un score très élevé peut-il être suspect ?", "Un score inhabituellement élevé peut provenir d'une fuite de données, d'un protocole de test incorrect ou d'une cible involontairement présente dans les variables.", SK_PITFALLS),
    ("Que signifie normaliser un échantillon avec Normalizer ?", "Normalizer met chaque échantillon individuellement à une norme donnée, souvent L2, contrairement à StandardScaler qui apprend des statistiques par caractéristique.", SK_PREPROCESS),
    ("Pourquoi ne pas comparer des modèles avec des plis différents ?", "Des partitions différentes introduisent une variation de données ; pour comparer directement, il faut un protocole de validation commun et documenté.", SK_PITFALLS),
    ("Quel est l'avantage de la validation croisée pour un petit jeu de données ?", "Elle utilise plusieurs partitions afin d'estimer la performance sans réserver un unique petit sous-ensemble pour le test de développement.", SK_MODEL_SELECTION),
    ("Comment éviter une transformation incohérente entre train et test ?", "Apprenez la transformation avec fit sur l'entraînement et appliquez transform au test, idéalement à travers un Pipeline.", SK_PITFALLS),
    ("Pourquoi les paramètres de Pipeline utilisent-ils __ ?", "Le séparateur __ permet de cibler un paramètre d'une étape interne par son nom, par exemple scaler__with_mean.", SK_PIPE),
    ("Quel est le rôle de score dans un estimateur ?", "score fournit une mesure de qualité par défaut de l'estimateur, mais le protocole doit préciser si cette mesure est adaptée au problème.", SK_PIPE),
]

LANGCHAIN_ITEMS = [
    ("Quel est le rôle principal de LangChain ?", "LangChain fournit des primitives pour composer des applications et agents autour de modèles, outils, prompts et middleware.", LC_OVERVIEW),
    ("Comment LangChain décrit-il un agent ?", "Un agent combine un modèle et un harness ; le harness inclut notamment le prompt, les outils et le middleware qui encadrent la boucle du modèle.", LC_OVERVIEW),
    ("À quoi sert create_agent ?", "create_agent crée un harness d'agent configurable en associant un modèle, des outils et éventuellement un prompt système.", LC_OVERVIEW),
    ("Qu'est-ce qu'un retriever dans LangChain ?", "Un retriever est une interface qui reçoit une requête non structurée et renvoie une liste de Documents pertinents.", LC_RETRIEVERS),
    ("Un retriever doit-il stocker des documents ?", "Non. Un retriever n'a pas besoin de stocker les documents ; il doit seulement pouvoir les récupérer.", LC_RETRIEVERS),
    ("Quelle relation existe entre vector store et retriever ?", "Un vector store peut être converti en retriever, mais le concept de retriever est plus général que celui de vector store.", LC_RETRIEVERS),
    ("Quel type d'entrée accepte un retriever ?", "Un retriever accepte une requête sous forme de chaîne et renvoie des objets Document.", LC_RETRIEVERS),
    ("Pourquoi séparer le modèle et les outils dans un agent ?", "Le modèle raisonne et produit des décisions, tandis que les outils donnent accès à des capacités ou données externes contrôlées.", LC_OVERVIEW),
    ("Quel est le rôle d'un prompt système dans un agent ?", "Le prompt système définit des instructions et contraintes de comportement qui encadrent les réponses du modèle dans le harness.", LC_OVERVIEW),
    ("Qu'appelle-t-on middleware dans LangChain ?", "Le middleware ajoute ou modifie des comportements autour de l'agent, par exemple des garde-fous, politiques d'outils ou logiques de routage.", LC_OVERVIEW),
    ("Quelle différence entre LangChain et LangGraph selon la documentation ?", "LangChain sert à composer un harness d'agent configurable ; LangGraph cible des besoins d'orchestration plus avancés mêlant flux déterministes et agentiques.", LC_OVERVIEW),
    ("Quel est le rôle de LangSmith dans l'écosystème présenté ?", "LangSmith sert à tracer, déboguer et évaluer les comportements d'agents à partir de leurs exécutions.", LC_OVERVIEW),
    ("Pourquoi indexer des documents avant un RAG ?", "L'index rend possible la récupération de passages pertinents qui seront fournis comme contexte au modèle génératif.", LC_DOCS),
    ("Quelle est la première responsabilité d'un retriever dans un RAG ?", "Il doit retourner des documents pertinents pour la requête afin que le système puisse s'appuyer sur des preuves récupérées.", LC_RETRIEVERS),
    ("Pourquoi un vector store n'est-il pas équivalent à un agent ?", "Un vector store organise et recherche des représentations de documents ; un agent orchestre un modèle, des outils et une boucle de décision.", LC_OVERVIEW),
    ("Que contient typiquement un Document récupéré ?", "Un Document représente un contenu récupéré avec son texte et des métadonnées utiles à la provenance et au traitement.", LC_RETRIEVERS),
    ("Pourquoi les métadonnées de document sont-elles utiles dans un RAG ?", "Elles permettent de tracer la source, filtrer ou citer des passages, et analyser les résultats de récupération.", LC_RETRIEVERS),
    ("Quel est le risque d'envoyer tout le corpus directement au modèle ?", "Le contexte peut dépasser les limites, augmenter le coût et distraire le modèle ; la récupération vise à fournir un sous-ensemble pertinent.", LC_DOCS),
    ("Quel est le but d'une chaîne de récupération puis génération ?", "Elle récupère d'abord des documents puis utilise leur contenu comme contexte pour produire une réponse davantage ancrée dans les sources.", LC_DOCS),
    ("Pourquoi la qualité du retriever influence-t-elle la réponse d'un RAG ?", "Le générateur ne peut s'appuyer que sur le contexte fourni ; des documents absents ou non pertinents limitent la complétude et la fidélité de la réponse.", LC_RETRIEVERS),
    ("Quel rôle jouent les outils dans create_agent ?", "Les outils sont des fonctions ou capacités que l'agent peut appeler pour accomplir des actions ou accéder à des informations.", LC_OVERVIEW),
    ("Pourquoi ne faut-il pas donner tous les outils possibles à un agent sans contrôle ?", "Le harness doit composer seulement les capacités nécessaires et appliquer des politiques afin de limiter le comportement et la surface d'erreur.", LC_OVERVIEW),
    ("Qu'est-ce que la portabilité de l'interface de modèle annoncée par LangChain ?", "Une interface standard permet de changer de fournisseur de modèle avec moins de changements dans l'application.", LC_OVERVIEW),
    ("Quel bénéfice apporte le traçage des exécutions d'agent ?", "Les traces permettent d'inspecter les appels, états et latences pour comprendre les échecs et améliorer le comportement.", LC_OVERVIEW),
    ("Pourquoi différencier récupération et stockage vectoriel dans une architecture ?", "Cette séparation permet de changer la source ou la stratégie de récupération sans confondre l'interface de recherche avec le mécanisme de stockage.", LC_RETRIEVERS),
    ("Quel est le rôle d'une requête non structurée pour un retriever ?", "La requête en langage naturel est l'entrée utilisée par le retriever pour sélectionner les Documents à retourner.", LC_RETRIEVERS),
    ("Comment un retriever personnalisé peut-il s'intégrer à LangChain ?", "La documentation indique que des retrievers personnalisés peuvent être implémentés en sous-classant BaseRetriever.", LC_RETRIEVERS),
    ("Pourquoi une application RAG doit-elle conserver les sources récupérées ?", "Conserver les sources rend la réponse vérifiable, facilite le débogage de la récupération et permet d'afficher des citations à l'utilisateur.", LC_RETRIEVERS),
    ("Qu'est-ce que l'orchestration agentique par rapport à une réponse unique ?", "L'orchestration gère la séquence d'actions, appels d'outils et états autour du modèle plutôt que de demander une seule complétion isolée.", LC_OVERVIEW),
    ("Quel est le rôle du modèle dans un agent LangChain ?", "Le modèle produit ou guide les décisions dans la boucle, tandis que le harness organise les outils, instructions et middleware autour de lui.", LC_OVERVIEW),
    ("Pourquoi utiliser des composants réutilisables dans LangChain ?", "La composition de primitives rend le système plus configurable et permet d'adapter seulement les composants nécessaires au cas d'usage.", LC_OVERVIEW),
    ("Quelle différence entre un outil et un retriever ?", "Un retriever retourne des Documents pour une requête ; un outil est une capacité générale que l'agent peut appeler, éventuellement pour rechercher ou agir.", LC_RETRIEVERS),
    ("Pourquoi tester un retriever séparément du générateur ?", "Cela permet de distinguer les erreurs de sélection de contexte des erreurs de formulation du modèle génératif.", LC_RETRIEVERS),
    ("Comment un RAG peut-il réduire les réponses non sourcées ?", "En fournissant au modèle des passages récupérés et en exigeant que la réponse s'appuie sur ces passages avec des citations ou refus d'absence de preuve.", LC_DOCS),
    ("Quel est l'intérêt d'un système de traces pour une évaluation RAG ?", "Les traces relient question, récupération, appels de modèle et réponse, ce qui aide à localiser le composant responsable d'un résultat faible.", LC_OVERVIEW),
    ("Pourquoi un retriever est-il plus général qu'un vector store ?", "Il décrit seulement le contrat de retour de Documents ; l'implémentation peut venir d'un vector store ou d'une autre source d'information.", LC_RETRIEVERS),
    ("Quel problème peut résoudre la compression automatique de contexte mentionnée pour Deep Agents ?", "Elle aide à gérer un contexte long en résumant ou réduisant l'information conservée dans l'exécution de l'agent.", LC_OVERVIEW),
    ("Pourquoi définir un cas d'usage avant de composer un agent ?", "La documentation recommande de composer exactement le harness requis afin d'éviter des outils, politiques et complexités inutiles.", LC_OVERVIEW),
    ("Quel est le rôle des garde-fous dans un harness d'agent ?", "Les garde-fous sont des comportements ajoutés autour du modèle pour contrôler les actions et réduire les réponses ou usages non souhaités.", LC_OVERVIEW),
    ("Pourquoi la récupération de documents est-elle utile pour la documentation technique ?", "Elle permet de sélectionner les passages pertinents d'un corpus versionné plutôt que de dépendre uniquement de la connaissance interne du modèle.", LC_DOCS),
]


def build_records(domain: str, items: list[tuple[str, str, str]]) -> list[dict]:
    assert len(items) == 40, f"{domain}: expected 40 records, found {len(items)}"
    question_types = ["factual"] * 8 + ["conceptual"] * 8 + ["procedural"] * 8 + ["troubleshooting"] * 8 + ["multi_step"] * 8
    difficulties = ["easy"] * 12 + ["medium"] * 20 + ["hard"] * 8
    records = []
    for index, ((question, reference, url), question_type, difficulty) in enumerate(zip(items, question_types, difficulties), start=1):
        records.append({
            "question_id": f"test_{domain}_{index:03d}",
            "split": "test",
            "domain": domain,
            "question_type": question_type,
            "difficulty": difficulty,
            "language": "fr",
            "user_input": question,
            "reference": reference,
            "reference_context_ids": [],
            "reference_source_urls": [url],
            "source_versions": {"python": "3.13"} if domain == "python" else {"scikit-learn": "1.5"} if domain == "scikit_learn" else {"langchain": "latest"},
            "annotation": {
                "review_status": "source_grounded_draft",
                "annotator": "Manus AI",
                "reviewer": "",
                "notes": "Review against the exact Kaggle corpus revision and add reference_context_ids before final conversion.",
            },
        })
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "datasets"
    target = root / "test_dataset_v1_source_grounded_draft.jsonl"
    records = build_records("python", PYTHON_ITEMS) + build_records("scikit_learn", SKLEARN_ITEMS) + build_records("langchain", LANGCHAIN_ITEMS)
    write_jsonl(target, records)
    print(f"Wrote {len(records)} source-grounded draft records to {target}")
