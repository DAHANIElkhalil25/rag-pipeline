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
import time
import json
import csv
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from numpy.linalg import norm

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
    Tente de parser une réponse LLM en JSON.
    """
    try:
        # Extraire ce qui est entre ```json et ``` s'il y a des balises
        match = re.search(r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # Sinon on essaie direct
        # Trouver la première accolade ou crochet
        start = text.find('{')
        start_list = text.find('[')
        if start == -1 and start_list != -1:
            start = start_list
        elif start != -1 and start_list != -1:
            start = min(start, start_list)
            
        end = text.rfind('}')
        end_list = text.rfind(']')
        if end == -1 and end_list != -1:
            end = end_list
        elif end != -1 and end_list != -1:
            end = max(end, end_list)

        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Impossible de parser la sortie LLM en JSON : {e}. Texte brut : {text}")
        raise ValueError("Invalid JSON output")

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
        """
        if not contexts or not answer:
            return 0.0

        prompt_extract = (
            "Extrais les affirmations clés (claims) de la réponse suivante. "
            "Renvoie un objet JSON avec une clé 'claims' contenant une liste de chaînes de caractères.\n\n"
            f"Réponse : {answer}\n\n"
            "Format attendu:\n"
            "{\"claims\": [\"affirmation 1\", \"affirmation 2\"]}\n"
        )

        try:
            extract_res = self._call_llm(prompt_extract)
            json_res = parse_json_from_llm(extract_res)
            claims = json_res.get("claims", [])
        except Exception:
            return 0.5

        if not claims:
            return 0.5

        contexts_str = "\n".join(contexts)
        supported_count = 0

        for claim in claims:
            prompt_verify = (
                "Étant donné le contexte suivant et une affirmation, vérifie si le contexte soutient l'affirmation.\n"
                "Réponds UNIQUEMENT par un objet JSON avec une clé 'supported' dont la valeur est un booléen (true ou false).\n\n"
                f"Contexte : {contexts_str}\n\n"
                f"Affirmation : {claim}\n\n"
                "Format attendu:\n"
                "{\"supported\": true}\n"
            )
            try:
                verify_res = self._call_llm(prompt_verify)
                verify_json = parse_json_from_llm(verify_res)
                if verify_json.get("supported") is True:
                    supported_count += 1
            except Exception:
                pass

        return supported_count / len(claims) if claims else 0.5

    def evaluate_answer_relevancy(self, question: str, answer: str, embedding_model: Any) -> float:
        """
        Évalue la pertinence de la réponse par rapport à la question.
        """
        if not answer or not question:
            return 0.0

        prompt = (
            "Génère 3 questions distinctes pour lesquelles la réponse fournie serait appropriée. "
            "Renvoie UNIQUEMENT un objet JSON avec une clé 'questions' contenant une liste de 3 chaînes.\n\n"
            f"Réponse : {answer}\n\n"
            "Format attendu:\n"
            "{\"questions\": [\"q1\", \"q2\", \"q3\"]}\n"
        )

        try:
            res = self._call_llm(prompt)
            json_res = parse_json_from_llm(res)
            gen_questions = json_res.get("questions", [])
            if not isinstance(gen_questions, list) or len(gen_questions) == 0:
                return 0.5
        except Exception:
            return 0.5

        try:
            # Embeddings
            q_emb = embedding_model.encode(question)
            gen_embs = [embedding_model.encode(q) for q in gen_questions]

            similarities = [cosine_similarity(q_emb, ge) for ge in gen_embs]
            return float(np.mean(similarities))
        except Exception as e:
            logger.error(f"Erreur d'embedding Answer Relevancy : {e}")
            return 0.5

    def evaluate_context_precision(self, question: str, contexts: List[str], reference_answer: str) -> float:
        """
        Évalue la précision du contexte vis-à-vis de la question (Context Precision).
        """
        if not contexts:
            return 0.0

        relevant_count = 0
        for ctx in contexts:
            prompt = (
                "Vérifie si le contexte suivant contient des informations utiles pour répondre à la question.\n"
                f"Question : {question}\n"
                f"Contexte : {ctx}\n"
                "Réponds UNIQUEMENT par un objet JSON avec une clé 'relevant' (booléen true/false).\n\n"
                "Format attendu:\n"
                "{\"relevant\": true}\n"
            )
            try:
                res = self._call_llm(prompt)
                json_res = parse_json_from_llm(res)
                if json_res.get("relevant") is True:
                    relevant_count += 1
            except Exception:
                pass
                
        return relevant_count / len(contexts) if contexts else 0.5

    def evaluate_context_recall(self, reference_answer: str, contexts: List[str]) -> float:
        """
        Évalue la couverture du contexte récupéré par rapport à la réponse de référence (Context Recall).
        """
        if not contexts or not reference_answer:
            return 0.0

        prompt_extract = (
            "Extrais les affirmations clés (claims) de la réponse de référence suivante. "
            "Renvoie un objet JSON avec une clé 'claims' contenant une liste de chaînes.\n\n"
            f"Réponse : {reference_answer}\n\n"
            "Format attendu:\n"
            "{\"claims\": [\"affirmation 1\", \"affirmation 2\"]}\n"
        )

        try:
            extract_res = self._call_llm(prompt_extract)
            json_res = parse_json_from_llm(extract_res)
            claims = json_res.get("claims", [])
        except Exception:
            return 0.5

        if not claims:
            return 0.5

        contexts_str = "\n".join(contexts)
        supported_count = 0

        for claim in claims:
            prompt_verify = (
                "Étant donné le contexte suivant et une affirmation de référence, vérifie si le contexte contient des informations supportant cette affirmation.\n"
                "Réponds UNIQUEMENT par un objet JSON avec une clé 'supported' (booléen true/false).\n\n"
                f"Contexte : {contexts_str}\n\n"
                f"Affirmation : {claim}\n\n"
                "Format attendu:\n"
                "{\"supported\": true}\n"
            )
            try:
                verify_res = self._call_llm(prompt_verify)
                verify_json = parse_json_from_llm(verify_res)
                if verify_json.get("supported") is True:
                    supported_count += 1
            except Exception:
                pass

        return supported_count / len(claims) if claims else 0.5

    def evaluate_single(self, question: str, reference_answer: str) -> Dict[str, Any]:
        """
        Exécute la pipeline RAG et calcule toutes les métriques RAGAS pour une paire QA.
        """
        logger.info(f"Évaluation de la question : '{question}'")
        try:
            # Génération de la réponse via la pipeline RAG
            rag_result = self.pipeline.generate_answer(question)
            answer = rag_result["answer"]
            contexts = [doc["content"] for doc in rag_result["source_documents"]]
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
