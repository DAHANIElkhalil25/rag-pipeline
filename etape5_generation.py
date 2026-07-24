"""
Étape 5 — Génération Augmentée par la Recherche (RAG).

Ce module implémente la dernière étape du pipeline RAG (Retrieval-Augmented Generation)
pour un système de questions-réponses sur la documentation technique. Il combine 
la recherche d'informations (FAISS, BM25) avec la génération de texte utilisant
le modèle Mistral 7B.

Références académiques :
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". 
  Advances in Neural Information Processing Systems.
- Jiang, A. Q., et al. (2023). "Mistral 7B". arXiv preprint arXiv:2310.06825.

Le pipeline supporte l'inférence sur GPU via la quantification 4-bit (bitsandbytes) 
et propose un repli sur Ollama pour les environnements CPU.
"""

import json
import faiss
import torch
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

from config import (PROCESSED_DIR, VECTORSTORE_DIR, BENCHMARK_DIR, LLM_CONFIG, logger)
from core.search import SimpleBM25

def load_index() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Charge l'index FAISS et les métadonnées des chunks depuis le répertoire de stockage.
    
    Returns:
        Tuple[faiss.Index, List[Dict[str, Any]]]: L'index FAISS chargé et la liste des chunks.
    """
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    chunks_path = VECTORSTORE_DIR / "chunks_metadata.json"
    
    logger.info(f"Chargement de l'index depuis {index_path}...")
    index = faiss.read_index(str(index_path))
    
    logger.info(f"Chargement des chunks depuis {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    return index, chunks

def load_search_config() -> Dict[str, Any]:
    """
    Lit le rapport de benchmark pour déterminer la configuration optimale.
    Récupère le modèle d'embedding, la méthode de recherche et le coefficient alpha.
    
    Returns:
        Dict[str, Any]: Configuration incluant embedding_model, search_method et alpha.
    """
    benchmark_path = BENCHMARK_DIR / "benchmark_report.json"
    
    config = {
        "embedding_model": "all-MiniLM-L6-v2",
        "search_method": "semantic",
        "alpha": 0.7,
    }
    
    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            
            recs = report.get("recommendations", {})
            
            # Modèle d'embedding
            if "embedding_model" in recs:
                config["embedding_model"] = recs["embedding_model"].get(
                    "model", config["embedding_model"]
                )
            
            # Méthode de recherche
            if "search_method" in recs:
                method = recs["search_method"].get("method", "")
                if "hybride" in method.lower() or "hybrid" in method.lower():
                    config["search_method"] = "hybrid"
                    # Extraire alpha depuis le nom (ex: "Hybride α=0.7")
                    if "0.7" in method:
                        config["alpha"] = 0.7
                    elif "0.5" in method:
                        config["alpha"] = 0.5
                elif "bm25" in method.lower():
                    config["search_method"] = "bm25"
                else:
                    config["search_method"] = "semantic"
            
            logger.info(f"Configuration chargée depuis le benchmark : {config}")
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture du rapport de benchmark : {e}")
    else:
        logger.info("Aucun rapport de benchmark trouvé, utilisation de la configuration par défaut.")
        
    return config

class LLMClient(ABC):
    """
    Interface abstraite pour les clients de modèles de langage.
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Génère une réponse à partir d'un prompt donné.
        
        Args:
            prompt (str): Le texte d'entrée.
            
        Returns:
            str: Le texte généré.
        """
        pass

class HuggingFaceClient(LLMClient):
    """
    Client utilisant l'écosystème Hugging Face (transformers) avec quantification 4-bit.
    Idéal pour l'exécution sur GPU (Kaggle, Google Colab avec GPU T4).
    """
    
    def __init__(self, model_name: str = LLM_CONFIG["model_name"]):
        """
        Initialise le client Hugging Face.
        
        Args:
            model_name (str): Le nom du modèle Hugging Face.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def _load_model(self) -> None:
        """Charge le modèle et le tokenizer de manière paresseuse."""
        if self.model is None:
            logger.info(f"Chargement du modèle HF {self.model_name} en 4-bit...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            logger.info("Modèle HF chargé avec succès.")
            
    def generate(self, prompt: str) -> str:
        """
        Génère une réponse en utilisant le modèle HF.
        
        Args:
            prompt (str): Le texte de la requête structurée.
            
        Returns:
            str: La réponse du modèle.
        """
        self._load_model()
        
        # Template Instruct pour Mistral : [INST] prompt [/INST]
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_CONFIG["max_new_tokens"],
                temperature=LLM_CONFIG["temperature"],
                top_p=LLM_CONFIG["top_p"],
                repetition_penalty=LLM_CONFIG["repetition_penalty"],
                do_sample=LLM_CONFIG["temperature"] > 0
            )
            
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

class OllamaClient(LLMClient):
    """
    Client de repli utilisant une instance locale d'Ollama via HTTP.
    Utile pour les environnements purement CPU.
    """
    
    def __init__(self, url: str = LLM_CONFIG.get("ollama_url", "http://localhost:11434"), model: str = LLM_CONFIG.get("ollama_model", "mistral")):
        """
        Initialise le client Ollama.
        
        Args:
            url (str): L'URL de l'API Ollama.
            model (str): Le nom du modèle Ollama à interroger.
        """
        self.url = url
        self.model = model
        
    def generate(self, prompt: str) -> str:
        """
        Génère une réponse via l'API REST d'Ollama.
        
        Args:
            prompt (str): Le texte d'entrée.
            
        Returns:
            str: Le texte généré par l'API locale.
        """
        payload = {
            "model": self.model,
            "prompt": f"[INST] {prompt} [/INST]",
            "stream": False,
            "options": {
                "temperature": LLM_CONFIG["temperature"],
                "top_p": LLM_CONFIG["top_p"],
                "repeat_penalty": LLM_CONFIG["repetition_penalty"],
                "num_predict": LLM_CONFIG["max_new_tokens"]
            }
        }
        
        try:
            response = requests.post(f"{self.url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête vers Ollama : {e}")
            return "Erreur : Impossible de contacter le modèle local."

def auto_detect_client() -> LLMClient:
    """
    Détecte automatiquement le matériel disponible pour instancier le meilleur client LLM.
    
    Returns:
        LLMClient: HuggingFaceClient si CUDA est disponible, sinon OllamaClient.
    """
    if torch.cuda.is_available():
        logger.info("GPU CUDA détecté. Utilisation de HuggingFaceClient.")
        return HuggingFaceClient()
    else:
        logger.info("Aucun GPU CUDA détecté. Utilisation de OllamaClient en mode repli.")
        return OllamaClient()

class RAGPipeline:
    """
    Pipeline principal RAG. Il orchestre la recherche d'informations dans l'index vectoriel
    et la génération de réponses sourcées.
    """
    
    def __init__(self, 
                 llm_client: LLMClient, 
                 index: faiss.Index, 
                 chunks: List[Dict[str, Any]], 
                 embedding_model: SentenceTransformer, 
                 search_config: Dict[str, Any]):
        """
        Initialise le pipeline RAG.
        
        Args:
            llm_client (LLMClient): Le client de génération de langage.
            index (faiss.Index): Index vectoriel FAISS.
            chunks (List[Dict[str, Any]]): Base de données des fragments documentaires.
            embedding_model (SentenceTransformer): Modèle pour l'encodage des requêtes.
            search_config (Dict[str, Any]): Configuration des poids de recherche (hybride, sémantique).
        """
        self.llm_client = llm_client
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.search_config = search_config
        
        # Initialisation de BM25 si la recherche hybride est configurée
        if self.search_config.get("search_method") == "hybrid":
            logger.info("Initialisation de SimpleBM25 pour la recherche hybride...")
            texts = [c.get("chunk_text", "") for c in self.chunks]
            self.bm25 = SimpleBM25(texts)
        else:
            self.bm25 = None

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les fragments documentaires les plus pertinents pour une question.
        
        Args:
            question (str): La question de l'utilisateur.
            k (int, optional): Le nombre de documents à récupérer. Par défaut 5.
            
        Returns:
            List[Dict[str, Any]]: Les fragments récupérés enrichis d'un score de pertinence.
        """
        # Encodage sémantique
        query_embedding = self.embedding_model.encode([question], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Recherche FAISS (sémantique)
        sem_scores_batch, sem_indices_batch = self.index.search(query_embedding, k * 2)
        sem_scores = sem_scores_batch[0]
        sem_indices = sem_indices_batch[0]
        
        results = []
        search_method = self.search_config.get("search_method", "semantic")
        
        if search_method == "hybrid" and self.bm25 is not None:
            # Recherche hybride : fusion sémantique + BM25
            # Réf. Gao et al. (2024) : la recherche hybride surpasse
            # systématiquement chaque méthode seule.
            n = len(self.chunks)
            alpha = self.search_config.get("alpha", 0.7)

            # Scores sémantiques sur top-50 candidats
            n_candidates = min(n, 50)
            sem_scores_all, sem_indices_all = self.index.search(query_embedding, n_candidates)
            sem_full = np.zeros(n)
            for sc, idx in zip(sem_scores_all[0], sem_indices_all[0]):
                if 0 <= idx < n:
                    sem_full[idx] = sc

            # Scores BM25 sur tout le corpus
            bm25_full = self.bm25.score(question)

            # Normalisation min-max
            def _norm(arr):
                mn, mx = arr.min(), arr.max()
                return (arr - mn) / (mx - mn + 1e-10)

            hybrid_scores = alpha * _norm(sem_full) + (1 - alpha) * _norm(bm25_full)
            top_k_idx = np.argsort(hybrid_scores)[::-1][:k]

            for idx in top_k_idx:
                if 0 <= idx < n:
                    chunk = self.chunks[idx].copy()
                    chunk["retrieval_score"] = float(hybrid_scores[idx])
                    results.append(chunk)
                
        else:
            # Sémantique uniquement
            for idx, score in zip(sem_indices[:k], sem_scores[:k]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk["retrieval_score"] = float(score)
                    results.append(chunk)
                    
        return results

    def build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Construit le prompt pour le modèle de langage.
        
        Args:
            question (str): La question de l'utilisateur.
            contexts (List[Dict[str, Any]]): Les passages récupérés.
            
        Returns:
            str: Le prompt formaté.
        """
        context_str = ""
        for i, ctx in enumerate(contexts):
            source = ctx.get("doc_title", ctx.get("doc_source", f"Document {i+1}"))
            text = ctx.get("chunk_text", "")
            context_str += f"\n--- Source: {source} ---\n{text}\n"
            
        prompt = (
            "Tu es un assistant expert en documentation technique (Python, Scikit-learn, LangChain). "
            "Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur les passages fournis ci-dessous.\n"
            "Règles importantes :\n"
            "1. Si les passages ne contiennent pas la réponse, dis simplement 'Je ne sais pas d'après les documents fournis'.\n"
            "2. Cite tes sources en t'appuyant sur les noms de documents indiqués.\n"
            "3. Rédige ta réponse en français de manière claire et concise.\n\n"
            f"Passages extraits :{context_str}\n\n"
            f"Question : {question}\n\n"
            "Réponse :"
        )
        return prompt

    def generate(self, prompt: str) -> str:
        """
        Génère une réponse via le client LLM.
        
        Args:
            prompt (str): Le prompt complet.
            
        Returns:
            str: La réponse textuelle générée.
        """
        return self.llm_client.generate(prompt)

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Exécute le pipeline complet (Retrieval -> Prompt -> Generation).
        
        Args:
            question (str): La question posée.
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant question, réponse et sources.
        """
        k = LLM_CONFIG.get("top_k_retrieval", 5)
        contexts = self.retrieve(question, k=k)
        
        if not contexts:
            return {
                "question": question,
                "answer": "Aucun document pertinent n'a été trouvé pour répondre à votre question.",
                "sources": [],
                "retrieval_scores": []
            }
            
        prompt = self.build_prompt(question, contexts)
        answer_text = self.generate(prompt)
        
        sources = [{"doc_title": c.get("doc_title"), "doc_source": c.get("doc_source")} for c in contexts]
        scores = [c.get("retrieval_score", 0.0) for c in contexts]
        
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "retrieval_scores": scores
        }

def interactive_mode(pipeline: RAGPipeline) -> None:
    """
    Lance le pipeline en mode interactif pour poser des questions en direct.
    
    Args:
        pipeline (RAGPipeline): Le pipeline RAG initialisé.
    """
    print("\n" + "="*50)
    print("Mode interactif RAG - Tapez 'quit' ou 'exit' pour quitter.")
    print("="*50)
    
    while True:
        try:
            question = input("\nVotre question: ")
            if question.lower().strip() in ['quit', 'exit']:
                break
                
            if not question.strip():
                continue
                
            print("Recherche et génération en cours...")
            result = pipeline.answer(question)
            
            print("\n" + "-"*50)
            print(f"RÉPONSE:\n{result['answer']}")
            print("-"*50)
            print("SOURCES UTILISÉES:")
            
            seen = set()
            for src in result['sources']:
                title = src.get('doc_title', 'Inconnu')
                url = src.get('doc_source', 'Non spécifiée')
                identifier = f"{title} ({url})"
                if identifier not in seen:
                    print(f"- {identifier}")
                    seen.add(identifier)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question : {e}")

def batch_mode(pipeline: RAGPipeline, questions: List[str], output_path: Path) -> None:
    """
    Traite une liste de questions en lot et sauvegarde les résultats au format JSON.
    
    Args:
        pipeline (RAGPipeline): L'instance du pipeline.
        questions (List[str]): Liste de questions à traiter.
        output_path (Path): Chemin du fichier de sortie.
    """
    results = []
    
    for i, q in enumerate(questions):
        logger.info(f"Traitement de la question {i+1}/{len(questions)} : {q}")
        res = pipeline.answer(q)
        results.append(res)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Traitement par lots terminé. Résultats sauvegardés dans {output_path}")

def main():
    """
    Point d'entrée principal de l'étape 5.

    Pipeline complet : chargement de l'index → détection du LLM →
    démonstration sur 3 questions → mode interactif.
    """
    print("\n" + "=" * 65)
    print("🤖  ÉTAPE 5 — RECHERCHE ET GÉNÉRATION RAG")
    print("=" * 65)
    print("\n  Ce module complète le pipeline RAG en ajoutant la génération")
    print("  de réponses via un LLM (Mistral 7B Instruct).")
    print("  Réf. Lewis et al. (2020), Jiang et al. (2023).\n")

    # Charger l'index FAISS et les chunks
    index, chunks = load_index()
    print(f"  → {len(chunks)} chunks chargés, index FAISS de dimension {index.d}.")

    # Charger la configuration optimale depuis le benchmark
    search_config = load_search_config()
    model_name = search_config.get("embedding_model", "all-MiniLM-L6-v2")

    logger.info(f"Chargement du modèle d'embedding : {model_name}...")
    embedding_model = SentenceTransformer(model_name)
    print(f"  → Modèle d'embedding : {model_name}")
    print(f"  → Méthode de recherche : {search_config['search_method']}")

    # Détecter et charger le LLM
    llm_client = auto_detect_client()

    # Créer le pipeline
    pipeline = RAGPipeline(
        llm_client=llm_client,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        search_config=search_config,
    )

    # Questions de démonstration
    demo_questions = [
        "What is a Python decorator and how to use it?",
        "How to perform cross-validation with Scikit-learn?",
        "How to create a custom retriever in LangChain?",
    ]

    print("\n" + "─" * 50)
    print("🧪  Démonstration sur 3 questions")
    print("─" * 50)

    for q in demo_questions:
        print(f"\n  ❓ {q}")
        try:
            res = pipeline.answer(q)
            # Afficher les 300 premiers caractères
            answer_preview = res["answer"][:300]
            if len(res["answer"]) > 300:
                answer_preview += "..."
            print(f"  💬 {answer_preview}")
            sources = set(s.get("doc_source", "?") for s in res["sources"])
            print(f"  📚 Sources : {', '.join(sources)}")
        except Exception as e:
            logger.error(f"Erreur lors de la démo pour '{q}': {e}")

    print("\n" + "=" * 65)
    print("🎉  Étape 5 prête — Lancement du mode interactif")
    print("=" * 65)
    interactive_mode(pipeline)


if __name__ == "__main__":
    main()
