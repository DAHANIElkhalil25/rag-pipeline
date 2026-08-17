"""
Étape 6 : Évaluation du système RAG selon la méthodologie RAGAS.

Ce module implémente une version simplifiée de RAGAS (Retrieval Augmented Generation Assessment),
un framework d'évaluation sans référence (ou avec référence) pour les systèmes RAG.
Référence : Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). 
RAGAS: Automated Evaluation of Retrieval Augmented Generation. 

Les métriques calculées sont :
- Faithfulness : L'exactitude de la réponse générée par rapport au contexte récupéré.
- Answer Relevancy : La pertinence de la réponse générée par rapport à la question.
- Context Precision : La pertinence des documents récupérés vis-à-vis de la question.
- Context Recall : La couverture des informations nécessaires par le contexte récupéré (mesuré via la réponse de référence).
"""

import os
import sys
import time
import json
import csv
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from numpy.linalg import norm

# Fix Windows UnicodeEncodeError : forcer l'encodage UTF-8 sur stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python < 3.7 : pas disponible, on continue

from config import (
    PROCESSED_DIR,
    VECTORSTORE_DIR,
    BENCHMARK_DIR,
    EVALUATION_DIR,
    LLM_CONFIG,
    logger
)
from etape5_generation import load_index, load_search_config, auto_detect_client, RAGPipeline

GROUND_TRUTH_QA: List[Dict[str, Any]] = [
    # Python (7 questions)
    {
        "question": "Qu'est-ce qu'une list comprehension en Python ?",
        "reference_answer": "Une list comprehension est une syntaxe concise pour créer des listes à partir d'itérables existants, en appliquant éventuellement des filtres et des transformations sur une seule ligne.",
        "expected_keywords": ["concise", "créer", "itérable"],
        "source_filter": "Python"
    },
    {
        "question": "Comment gérer les exceptions en Python ?",
        "reference_answer": "Les exceptions sont gérées avec les blocs try et except. On met le code susceptible de lever une erreur dans le bloc try, et on la gère dans le bloc except. On peut utiliser finally pour exécuter du code à la fin quoiqu'il arrive.",
        "expected_keywords": ["try", "except", "finally"],
        "source_filter": "Python"
    },
    {
        "question": "Qu'est-ce qu'un décorateur en Python ?",
        "reference_answer": "Un décorateur est une fonction qui prend une autre fonction en argument et étend son comportement sans la modifier de façon permanente, souvent avec la syntaxe @decorateur.",
        "expected_keywords": ["fonction", "comportement", "@"],
        "source_filter": "Python"
    },
    {
        "question": "Quelle est la différence entre un tuple et une liste en Python ?",
        "reference_answer": "La principale différence est qu'une liste est mutable (on peut la modifier) alors qu'un tuple est immuable (non modifiable après sa création).",
        "expected_keywords": ["mutable", "immuable"],
        "source_filter": "Python"
    },
    {
        "question": "Que fait le mot clé yield en Python ?",
        "reference_answer": "Le mot clé yield transforme une fonction en générateur, permettant de renvoyer une valeur tout en conservant l'état local pour reprendre l'exécution par la suite.",
        "expected_keywords": ["générateur", "état", "renvoyer"],
        "source_filter": "Python"
    },
    {
        "question": "Qu'est-ce que le Global Interpreter Lock (GIL) ?",
        "reference_answer": "Le GIL est un mécanisme dans CPython qui empêche l'exécution simultanée de plusieurs threads de bytecodes Python, ce qui limite le parallélisme multithread, mais pas le multiprocessing.",
        "expected_keywords": ["multithread", "simultanée", "CPython"],
        "source_filter": "Python"
    },
    {
        "question": "Comment fonctionne la gestion de la mémoire et le ramasse-miettes en Python ?",
        "reference_answer": "Python gère la mémoire automatiquement principalement via le comptage de références et un garbage collector cyclique pour détecter et nettoyer les cycles de références.",
        "expected_keywords": ["comptage de références", "garbage collector", "cycles"],
        "source_filter": "Python"
    },
    # Sklearn (6 questions)
    {
        "question": "Qu'est-ce que GridSearchCV dans Scikit-Learn ?",
        "reference_answer": "GridSearchCV est une fonction de Scikit-Learn qui permet d'effectuer une recherche exhaustive sur une grille de paramètres spécifiée pour un estimateur, tout en utilisant la validation croisée pour évaluer chaque combinaison.",
        "expected_keywords": ["recherche", "paramètres", "validation croisée"],
        "source_filter": "Sklearn"
    },
    {
        "question": "Quelle est la différence entre fit() et fit_transform() ?",
        "reference_answer": "fit() calcule les paramètres nécessaires à la transformation (comme la moyenne), tandis que fit_transform() calcule ces paramètres puis applique immédiatement la transformation sur les mêmes données.",
        "expected_keywords": ["calcule", "applique", "données"],
        "source_filter": "Sklearn"
    },
    {
        "question": "A quoi sert StandardScaler dans Scikit-Learn ?",
        "reference_answer": "StandardScaler standardise les caractéristiques en supprimant la moyenne et en mettant à l'échelle la variance unitaire (centrage et réduction).",
        "expected_keywords": ["moyenne", "variance", "centrage"],
        "source_filter": "Sklearn"
    },
    {
        "question": "Comment gérer les valeurs manquantes avec Scikit-Learn ?",
        "reference_answer": "Les valeurs manquantes peuvent être gérées avec la classe SimpleImputer, qui permet de remplacer les valeurs absentes par la moyenne, la médiane, le mode ou une constante, ou avec KNNImputer.",
        "expected_keywords": ["SimpleImputer", "remplacer", "moyenne"],
        "source_filter": "Sklearn"
    },
    {
        "question": "Qu'est-ce qu'un pipeline dans Scikit-Learn ?",
        "reference_answer": "Un pipeline permet d'enchaîner séquentiellement plusieurs étapes de traitement (comme le pré-traitement) avec un estimateur final, facilitant l'application des mêmes transformations lors de l'entraînement et des tests.",
        "expected_keywords": ["enchaîner", "traitement", "estimateur"],
        "source_filter": "Sklearn"
    },
    {
        "question": "Qu'est-ce que la validation croisée K-Fold ?",
        "reference_answer": "K-Fold divise l'ensemble de données en K sous-ensembles ou 'plis'. Le modèle est entraîné sur K-1 plis et évalué sur le pli restant, et ce processus est répété K fois.",
        "expected_keywords": ["divise", "entraîné", "évalué"],
        "source_filter": "Sklearn"
    },
    # LangChain (7 questions)
    {
        "question": "Qu'est-ce que LangChain ?",
        "reference_answer": "LangChain est un framework conçu pour simplifier la création d'applications basées sur les grands modèles de langage (LLMs) en fournissant des composants modulaires comme les chaînes, les agents et les outils.",
        "expected_keywords": ["framework", "LLMs", "modulaires"],
        "source_filter": "LangChain"
    },
    {
        "question": "Quel est le rôle d'un 'Agent' dans LangChain ?",
        "reference_answer": "Un Agent dans LangChain utilise un LLM pour décider quelles actions entreprendre de manière itérative, en utilisant différents outils disponibles pour accomplir une tâche complexe.",
        "expected_keywords": ["décider", "actions", "outils"],
        "source_filter": "LangChain"
    },
    {
        "question": "A quoi sert la mémoire (Memory) dans LangChain ?",
        "reference_answer": "La mémoire permet à une chaîne ou à un agent de se souvenir des interactions passées, maintenant ainsi un contexte conversationnel pour des échanges cohérents sur plusieurs tours.",
        "expected_keywords": ["souvenir", "interactions", "contexte"],
        "source_filter": "LangChain"
    },
    {
        "question": "Qu'est-ce qu'un Document Loader dans LangChain ?",
        "reference_answer": "Un Document Loader est un composant qui charge des données provenant de diverses sources (fichiers PDF, pages web, bases de données) sous forme de documents standards exploitables par le framework.",
        "expected_keywords": ["charge", "données", "sources"],
        "source_filter": "LangChain"
    },
    {
        "question": "A quoi servent les Text Splitters dans LangChain ?",
        "reference_answer": "Les Text Splitters découpent de longs documents en morceaux plus petits (chunks), ce qui est essentiel pour respecter la limite de contexte des LLMs lors de l'indexation et de la recherche vectorielle.",
        "expected_keywords": ["découpent", "morceaux", "contexte"],
        "source_filter": "LangChain"
    },
    {
        "question": "Qu'est-ce qu'une RetrievalQA chain dans LangChain ?",
        "reference_answer": "C'est une chaîne spécifique qui combine un retriever (comme une base de données vectorielle) et un LLM pour répondre à des questions en utilisant les documents pertinents récupérés comme contexte.",
        "expected_keywords": ["chaîne", "retriever", "documents"],
        "source_filter": "LangChain"
    },
    {
        "question": "Qu'est-ce que le LCEL (LangChain Expression Language) ?",
        "reference_answer": "Le LCEL est un langage déclaratif qui simplifie la composition de chaînes dans LangChain en permettant de lier facilement les composants avec le symbole pipe (|).",
        "expected_keywords": ["déclaratif", "composition", "pipe"],
        "source_filter": "LangChain"
    }
]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

def parse_json_from_llm(text: str) -> Any:
    """
    Parse la réponse LLM en JSON avec plusieurs stratégies de fallback.

    Mistral 7B ne renvoie pas toujours du JSON propre — il peut inclure
    du texte explicatif avant/après, utiliser des guillemets simples,
    ou omettre les accolades. Cette fonction gère tous ces cas.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM output")

    text = text.strip()

    # Stratégie 1 : Extraire un bloc ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Stratégie 2 : Trouver le premier { ... } ou [ ... ] dans le texte
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                # Essayer en remplaçant les guillemets simples par des doubles
                fixed = text[start:end + 1].replace("'", '"')
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

    # Stratégie 3 : Parsing direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Cannot parse JSON from LLM output: {text[:200]}")


def parse_boolean_from_llm(text: str) -> bool:
    """
    Extrait un booléen de la réponse LLM, même si ce n'est pas du JSON.

    Cherche dans l'ordre :
    1. JSON avec clé 'supported' ou 'relevant'
    2. Mots-clés : true/false, oui/non, yes/no
    """
    if not text:
        return False

    # Essayer le JSON d'abord
    try:
        result = parse_json_from_llm(text)
        if isinstance(result, dict):
            for key in ['supported', 'relevant', 'verdict']:
                if key in result:
                    return bool(result[key])
        return bool(result)
    except (ValueError, TypeError):
        pass

    # Fallback : chercher des mots-clés dans le texte
    text_lower = text.lower().strip()
    positive = ['true', '"supported": true', '"relevant": true',
                'oui', 'yes', 'est soutenu', 'est pertinent',
                'supporte', 'confirme', 'contient']
    negative = ['false', '"supported": false', '"relevant": false',
                'non', 'no', "n'est pas soutenu", "n'est pas pertinent",
                'ne supporte pas', 'ne contient pas']

    # Compter les indicateurs positifs vs négatifs
    pos_score = sum(1 for p in positive if p in text_lower)
    neg_score = sum(1 for n in negative if n in text_lower)

    return pos_score > neg_score


def parse_list_from_llm(text: str, key: str) -> List[str]:
    """
    Extrait une liste de strings de la réponse LLM.

    Args:
        text: Sortie brute du LLM.
        key: Clé JSON attendue (ex: 'claims', 'questions').

    Returns:
        Liste de strings extraites, ou liste vide.
    """
    if not text:
        return []

    # Essayer le JSON
    try:
        result = parse_json_from_llm(text)
        if isinstance(result, dict) and key in result:
            items = result[key]
            if isinstance(items, list):
                return [str(item) for item in items if item]
        if isinstance(result, list):
            return [str(item) for item in result if item]
    except (ValueError, TypeError):
        pass

    # Fallback : chercher des listes numérotées ou à puces dans le texte
    lines = text.strip().split('\n')
    items = []
    for line in lines:
        line = line.strip()
        # Patterns : "1. ...", "- ...", "• ...", "* ..."
        match = re.match(r'^(?:\d+[\.\)]\s*|[-•*]\s+|"\s*)(.*?)(?:"\s*,?\s*)?$', line)
        if match and len(match.group(1).strip()) > 10:
            items.append(match.group(1).strip().strip('"').strip("'"))

    return items if items else []

class RAGASEvaluator:
    """
    Évaluateur RAGAS pour le système RAG.

    Implémente les 4 métriques de Es et al. (2024) :
    - Faithfulness, Answer Relevancy, Context Precision, Context Recall.
    """

    def __init__(self, pipeline: RAGPipeline, llm_client: Any,
                 embedding_model: Any = None):
        """
        Initialise l'évaluateur.
        
        Args:
            pipeline: L'objet RAGPipeline configuré.
            llm_client: Le client LLM (HuggingFaceClient ou OllamaClient).
            embedding_model: Le modèle d'embedding pour Answer Relevancy.
        """
        self.pipeline = pipeline
        self.llm = llm_client
        self.embedding_model = embedding_model or pipeline.embedding_model

    def _call_llm(self, prompt: str) -> str:
        """
        Appel générique au LLM via l'interface LLMClient.generate().
        Compatible avec HuggingFaceClient et OllamaClient.
        """
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Erreur lors de l'appel LLM pour l'évaluation : {e}")
            raise e

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Évalue l'exactitude de la réponse par rapport au contexte (Faithfulness).
        Score = claims soutenues / total claims.
        """
        if not contexts or not answer:
            return 0.0

        prompt_extract = (
            "Liste les affirmations factuelles contenues dans cette réponse.\n"
            "Donne une liste numérotée (1., 2., 3., etc.).\n\n"
            f"Réponse : {answer[:1000]}\n\n"
            "Affirmations :\n"
        )

        try:
            extract_res = self._call_llm(prompt_extract)
            claims = parse_list_from_llm(extract_res, "claims")
        except Exception:
            claims = []

        if not claims:
            return 0.5

        contexts_str = "\n".join(c[:500] for c in contexts[:3])
        supported_count = 0

        for claim in claims[:5]:  # Limiter à 5 claims pour la vitesse
            prompt_verify = (
                f"Le contexte suivant soutient-il cette affirmation ?\n\n"
                f"Contexte : {contexts_str[:1500]}\n\n"
                f"Affirmation : {claim}\n\n"
                "Réponds par 'true' ou 'false' uniquement.\n"
            )
            try:
                verify_res = self._call_llm(prompt_verify)
                if parse_boolean_from_llm(verify_res):
                    supported_count += 1
            except Exception:
                supported_count += 0.5  # Incertain = 0.5

        return supported_count / len(claims) if claims else 0.5

    def evaluate_answer_relevancy(self, question: str, answer: str, embedding_model: Any) -> float:
        """
        Évalue la pertinence de la réponse par rapport à la question.
        Génère des questions inverses et mesure la similarité cosinus.
        """
        if not answer or not question:
            return 0.0

        prompt = (
            "Génère 3 questions auxquelles cette réponse pourrait répondre.\n"
            "Donne une liste numérotée (1., 2., 3.).\n\n"
            f"Réponse : {answer[:1000]}\n\n"
            "Questions :\n"
        )

        try:
            res = self._call_llm(prompt)
            gen_questions = parse_list_from_llm(res, "questions")
            if not gen_questions:
                return 0.5
        except Exception:
            return 0.5

        try:
            q_emb = embedding_model.encode(question)
            gen_embs = [embedding_model.encode(q) for q in gen_questions[:3]]
            similarities = [cosine_similarity(q_emb, ge) for ge in gen_embs]
            return max(0.0, min(1.0, float(np.mean(similarities))))
        except Exception as e:
            logger.error(f"Erreur d'embedding Answer Relevancy : {e}")
            return 0.5

    def evaluate_context_precision(self, question: str, contexts: List[str], reference_answer: str) -> float:
        """
        Évalue la précision du contexte vis-à-vis de la question (Context Precision).
        Score = passages pertinents / total passages.
        """
        if not contexts:
            return 0.0

        relevant_count = 0
        for ctx in contexts[:5]:  # Limiter à 5 passages
            prompt = (
                f"Ce passage est-il utile pour répondre à cette question ?\n\n"
                f"Question : {question}\n\n"
                f"Passage : {ctx[:800]}\n\n"
                "Réponds par 'true' ou 'false' uniquement.\n"
            )
            try:
                res = self._call_llm(prompt)
                if parse_boolean_from_llm(res):
                    relevant_count += 1
            except Exception:
                relevant_count += 0.5  # Incertain

        return relevant_count / len(contexts) if contexts else 0.5

    def evaluate_context_recall(self, reference_answer: str, contexts: List[str]) -> float:
        """
        Évalue la couverture du contexte par rapport à la réponse de référence.
        Score = claims de la référence couvertes / total claims.
        """
        if not contexts or not reference_answer:
            return 0.0

        prompt_extract = (
            "Liste les informations clés contenues dans cette réponse de référence.\n"
            "Donne une liste numérotée (1., 2., 3.).\n\n"
            f"Réponse : {reference_answer}\n\n"
            "Informations clés :\n"
        )

        try:
            extract_res = self._call_llm(prompt_extract)
            claims = parse_list_from_llm(extract_res, "claims")
        except Exception:
            claims = []

        if not claims:
            return 0.5

        contexts_str = "\n".join(c[:500] for c in contexts[:3])
        supported_count = 0

        for claim in claims[:5]:  # Limiter à 5
            prompt_verify = (
                f"Le contexte suivant contient-il cette information ?\n\n"
                f"Contexte : {contexts_str[:1500]}\n\n"
                f"Information : {claim}\n\n"
                "Réponds par 'true' ou 'false' uniquement.\n"
            )
            try:
                verify_res = self._call_llm(prompt_verify)
                if parse_boolean_from_llm(verify_res):
                    supported_count += 1
            except Exception:
                supported_count += 0.5

        return supported_count / len(claims) if claims else 0.5

    def evaluate_single(self, question: str, reference_answer: str) -> Dict[str, Any]:
        """
        Exécute la pipeline RAG et calcule toutes les métriques RAGAS pour une paire QA.

        Correction bug critique (v1.1) :
        - L'ancienne implémentation appelait self.pipeline.generate_answer() qui n'existe pas.
          La méthode correcte est pipeline.answer() mais elle ne retourne pas le texte brut des chunks.
        - On utilise maintenant pipeline.retrieve() pour récupérer les chunks avec leur texte (chunk_text),
          puis pipeline.build_prompt() + pipeline.generate() pour générer la réponse.
          Cela garantit d'avoir à la fois la réponse générée ET le texte des contextes pour les métriques.
        """
        logger.info(f"Evaluation de la question : '{question}'")
        answer = ""
        contexts = []  # Textes bruts des chunks recupérés
        try:
            # 1. Retrieval : récupérer les k chunks pertinents avec leur texte
            k = LLM_CONFIG.get("top_k_retrieval", 5)
            retrieved_chunks = self.pipeline.retrieve(question, k=k)

            # Extraire le texte brut de chaque chunk pour les métriques de contexte
            # La clé correcte dans les chunks est 'chunk_text' (pas 'content')
            contexts = [chunk.get("chunk_text", "") for chunk in retrieved_chunks
                        if chunk.get("chunk_text", "").strip()]

            # 2. Génération : construire le prompt et générer la réponse
            if retrieved_chunks:
                prompt = self.pipeline.build_prompt(question, retrieved_chunks)
                answer = self.pipeline.generate(prompt)
            else:
                answer = "Aucun document pertinent trouve pour répondre à cette question."

        except Exception as e:
            logger.error(f"Erreur de la pipeline pour la question '{question}' : {e}")
            answer = ""
            contexts = []

        metrics = {
            "faithfulness": self.evaluate_faithfulness(answer, contexts),
            "answer_relevancy": self.evaluate_answer_relevancy(question, answer, self.embedding_model),
            "context_precision": self.evaluate_context_precision(question, contexts, reference_answer),
            "context_recall": self.evaluate_context_recall(reference_answer, contexts)
        }

        return {
            "question": question,
            "generated_answer": answer,
            "reference_answer": reference_answer,
            "contexts": contexts,
            "metrics": metrics
        }

    def evaluate_all(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Évalue toutes les paires Question/Réponse.
        """
        results = []
        for i, qa in enumerate(qa_pairs):
            logger.info(f"--- Évaluation Q {i+1}/{len(qa_pairs)} ---")
            res = self.evaluate_single(qa["question"], qa["reference_answer"])
            res["source_filter"] = qa.get("source_filter", "Autre")
            results.append(res)
            
        # Calcul des moyennes
        mean_metrics = {
            "faithfulness": float(np.mean([r["metrics"]["faithfulness"] for r in results])),
            "answer_relevancy": float(np.mean([r["metrics"]["answer_relevancy"] for r in results])),
            "context_precision": float(np.mean([r["metrics"]["context_precision"] for r in results])),
            "context_recall": float(np.mean([r["metrics"]["context_recall"] for r in results]))
        }
        
        return {
            "mean_metrics": mean_metrics,
            "details": results
        }


def generate_evaluation_report(results: Dict[str, Any], elapsed: float):
    """
    Génère et sauvegarde le rapport d'évaluation.
    """
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    report_json_path = EVALUATION_DIR / "ragas_report.json"
    report_csv_path = EVALUATION_DIR / "ragas_details.csv"
    
    # 1. Sauvegarde JSON
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    # 2. Sauvegarde CSV
    with open(report_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question", "Source", "Generated_Answer", "Faithfulness", 
            "Answer_Relevancy", "Context_Precision", "Context_Recall"
        ])
        for detail in results["details"]:
            writer.writerow([
                detail["question"],
                detail.get("source_filter", ""),
                detail["generated_answer"],
                f"{detail['metrics']['faithfulness']:.4f}",
                f"{detail['metrics']['answer_relevancy']:.4f}",
                f"{detail['metrics']['context_precision']:.4f}",
                f"{detail['metrics']['context_recall']:.4f}"
            ])
            
    # 3. Print résumé dans le terminal
    print("\n" + "="*60)
    print(" " * 15 + "RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("="*60)
    print(f"Temps écoulé : {elapsed:.2f} secondes")
    print("\nMoyennes globales :")
    print("-" * 30)
    for k, v in results["mean_metrics"].items():
        print(f"{k.ljust(20)} : {v:.4f}")
        
    # 4. Print breakdown par source
    print("\nMoyennes par source :")
    print("-" * 30)
    sources = set([d.get("source_filter", "") for d in results["details"]])
    for src in sources:
        if not src:
            continue
        src_details = [d for d in results["details"] if d.get("source_filter") == src]
        if src_details:
            f_mean = float(np.mean([d["metrics"]["faithfulness"] for d in src_details]))
            ar_mean = float(np.mean([d["metrics"]["answer_relevancy"] for d in src_details]))
            cp_mean = float(np.mean([d["metrics"]["context_precision"] for d in src_details]))
            cr_mean = float(np.mean([d["metrics"]["context_recall"] for d in src_details]))
            print(f"Source: {src} ({len(src_details)} questions)")
            print(f"  Faithfulness      : {f_mean:.4f}")
            print(f"  Answer Relevancy  : {ar_mean:.4f}")
            print(f"  Context Precision : {cp_mean:.4f}")
            print(f"  Context Recall    : {cr_mean:.4f}")
            print()

def main():
    """
    Point d'entrée de l'évaluation RAGAS.
    """
    print("\n" + "=" * 65)
    print("📊  ÉTAPE 6 — ÉVALUATION RAGAS DU SYSTÈME RAG")
    print("=" * 65)
    print("\n  Évaluation du système selon les métriques RAGAS")
    print("  Réf. Es et al. (2024) — RAGAS: Automated Evaluation of RAG.\n")
    
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    start_time = time.time()
    
    # Chargement de l'index et des chunks
    logger.info("Chargement de l'index et de la configuration...")
    index, chunks = load_index()
    print(f"  → {len(chunks)} chunks chargés.")
    
    # Configuration depuis le benchmark
    search_config = load_search_config()
    model_name = search_config.get("embedding_model", "all-MiniLM-L6-v2")
    
    logger.info(f"Chargement du modèle d'embedding : {model_name}...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(model_name)
    
    # Initialisation du LLM
    logger.info("Initialisation du client LLM...")
    llm_client = auto_detect_client()
    
    # Construction du pipeline RAG
    logger.info("Initialisation du pipeline RAG...")
    pipeline = RAGPipeline(
        llm_client=llm_client,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        search_config=search_config,
    )
    
    evaluator = RAGASEvaluator(pipeline, llm_client, embedding_model)
    
    print(f"  → Évaluation sur {len(GROUND_TRUTH_QA)} questions (ground truth)")
    print("  → Métriques : Faithfulness, Answer Relevancy, Context Precision, Context Recall\n")
    
    logger.info("Démarrage de l'évaluation...")
    report = evaluator.evaluate_all(GROUND_TRUTH_QA)
    
    elapsed = time.time() - start_time
    
    generate_evaluation_report(report, elapsed)
    
    print("\n" + "=" * 65)
    print("🎉  Étape 6 terminée !")
    print(f"    → Rapport sauvegardé dans {EVALUATION_DIR}")
    print(f"    ⏱️  Durée totale : {elapsed/60:.1f} minutes")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
