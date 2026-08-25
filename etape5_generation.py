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
import sys
import torch
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# Fix Windows UnicodeEncodeError : forcer l'encodage UTF-8 sur stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python < 3.7 : pas disponible, on continue

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
        "embedding_query_prefix": "",
        "candidate_k": LLM_CONFIG.get("top_k_retrieval", 5),
        "final_k": LLM_CONFIG.get("top_k_retrieval", 5),
        "reranker": {"enabled": False},
    }

    manifest_path = VECTORSTORE_DIR / "index_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index_config = manifest.get("index_config", {})
            for key in ("embedding_model", "search_method", "alpha", "embedding_query_prefix", "candidate_k", "final_k", "reranker", "retrieval_profile"):
                if key in index_config:
                    config[key] = index_config[key]
            logger.info(f"Configuration chargée depuis le manifeste index : {config}")
            return config
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Manifeste d'index invalide, repli benchmark : {exc}")
    
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
        self._load_error = None
        
    def _load_model(self) -> None:
        """Charge le modèle et le tokenizer de manière paresseuse."""
        if self._load_error is not None:
            raise RuntimeError(f"Le modèle HF n'a pas pu être chargé : {self._load_error}")
        if self.model is None:
            logger.info(f"Chargement du modèle HF {self.model_name} en 4-bit...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                self.model.eval()
                logger.info("Modèle HF chargé avec succès.")
            except Exception as exc:
                # Do not retry a failed load for every evaluation sample. This
                # was the source of repeated OOMs in the old Kaggle notebook.
                self._load_error = exc
                raise
            
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
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=LLM_CONFIG.get("max_input_tokens", 3072),
        )
        input_device = next(self.model.parameters()).device
        inputs = {name: value.to(input_device) for name, value in inputs.items()}

        with torch.inference_mode():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": LLM_CONFIG["max_new_tokens"],
                "repetition_penalty": LLM_CONFIG["repetition_penalty"],
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if LLM_CONFIG.get("temperature", 0.0) > 0:
                generation_kwargs.update({
                    "temperature": LLM_CONFIG["temperature"],
                    "top_p": LLM_CONFIG["top_p"],
                    "do_sample": True,
                })
            else:
                generation_kwargs["do_sample"] = False
            outputs = self.model.generate(**generation_kwargs)
            
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
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
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=LLM_CONFIG.get("request_timeout_seconds", 180),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête vers Ollama : {e}")
            raise RuntimeError(f"Impossible de contacter Ollama : {e}") from e

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
        self._reranker = None
        
        # Initialisation de BM25 si la recherche hybride est configurée
        if self.search_config.get("search_method") == "hybrid":
            logger.info("Initialisation de SimpleBM25 pour la recherche hybride...")
            texts = [c.get("chunk_text", "") for c in self.chunks]
            self.bm25 = SimpleBM25(texts)
        else:
            self.bm25 = None

    def _load_reranker(self):
        """Load the cross-encoder only when a profile explicitly enables it."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            cfg = self.search_config.get("reranker", {})
            model_name = cfg["model_name"]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Chargement du reranker {model_name} sur {device}...")
            self._reranker = CrossEncoder(model_name, max_length=cfg.get("max_length", 512), device=device)
        return self._reranker

    def _unload_reranker(self) -> None:
        if self._reranker is not None:
            del self._reranker
            self._reranker = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _rerank(self, question: str, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        cfg = self.search_config.get("reranker", {})
        if not cfg.get("enabled") or not candidates:
            return candidates[:k]
        reranker = self._load_reranker()
        pairs = [(question, candidate.get("chunk_text", "")) for candidate in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)
        ranked = []
        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["candidate_retrieval_score"] = item.get("retrieval_score")
            item["reranker_score"] = float(score)
            item["retrieval_score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        if cfg.get("unload_after_query", False):
            self._unload_reranker()
        return ranked[:k]

    def retrieve(
        self,
        question: str,
        k: int = 5,
        source_domain: Optional[str] = None,
        source_urls: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les fragments documentaires les plus pertinents pour une question.
        
        Args:
            question (str): La question de l'utilisateur.
            k (int, optional): Le nombre de documents à récupérer. Par défaut 5.
            source_domain (str, optional): Domaine documentaire imposé pour
                l'annotation d'un jeu de test (`python`, `scikit_learn` ou
                `langchain`). Les questions ordinaires n'utilisent aucun filtre.
            source_urls (list[str], optional): URLs documentaires officielles
                imposées lors de la recherche de preuves pour l'annotation.
            
        Returns:
            List[Dict[str, Any]]: Les fragments récupérés enrichis d'un score de pertinence.
        """
        # Encodage sémantique
        query_prefix = self.search_config.get("embedding_query_prefix", "")
        query_embedding = self.embedding_model.encode([query_prefix + question], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Recherche FAISS (sémantique)
        candidate_k = max(k, int(self.search_config.get("candidate_k", k)))
        semantic_pool_k = min(
            len(self.chunks),
            max(candidate_k, candidate_k * 50 if source_urls else candidate_k),
        )
        sem_scores_batch, sem_indices_batch = self.index.search(query_embedding, semantic_pool_k)
        sem_scores = sem_scores_batch[0]
        sem_indices = sem_indices_batch[0]
        
        results = []
        search_method = self.search_config.get("search_method", "semantic")
        
        def _matches_domain(chunk: Dict[str, Any]) -> bool:
            if source_domain is None:
                return True
            identifier = str(chunk.get("chunk_id", ""))
            prefixes = {"python": "python_", "scikit_learn": "sklearn_", "langchain": "langchain_"}
            return identifier.startswith(prefixes.get(source_domain, "__no_match__"))

        def _matches_source_url(chunk: Dict[str, Any]) -> bool:
            if not source_urls:
                return True
            chunk_url = str(chunk.get("doc_url", "")).rstrip("/")
            return chunk_url in {str(url).rstrip("/") for url in source_urls}

        def _matches_constraints(chunk: Dict[str, Any]) -> bool:
            return _matches_domain(chunk) and _matches_source_url(chunk)

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
            ranked_indices = np.argsort(hybrid_scores)[::-1]
            top_k_idx = [idx for idx in ranked_indices if _matches_constraints(self.chunks[idx])][:candidate_k]

            for idx in top_k_idx:
                if 0 <= idx < n:
                    chunk = self.chunks[idx].copy()
                    chunk["retrieval_score"] = float(hybrid_scores[idx])
                    results.append(chunk)
                
        else:
            # Sémantique uniquement
            for idx, score in zip(sem_indices, sem_scores):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    if not _matches_constraints(chunk):
                        continue
                    chunk["retrieval_score"] = float(score)
                    results.append(chunk)
                    if len(results) >= candidate_k:
                        break
                    
        return self._rerank(question, results, k)

    def get_prompt_contexts(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return the exact ranked text windows that will be inserted into the prompt."""
        max_chars = LLM_CONFIG.get("max_context_chars_per_chunk", 2600)
        return [
            {
                "rank": index,
                "chunk_id": context.get("chunk_id"),
                "text": context.get("chunk_text", "")[:max_chars],
                "source_label": context.get("doc_title", context.get("doc_source", f"Document {index}")),
            }
            for index, context in enumerate(contexts, start=1)
            if context.get("chunk_text", "").strip()
        ]

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
        for context in self.get_prompt_contexts(contexts):
            context_str += f"\n[S{context['rank']}] Source: {context['source_label']}\n{context['text']}\n"

        prompt = (
            "Tu es un assistant expert en documentation technique sur Python, Scikit-learn et LangChain. "
            "Réponds uniquement à partir des passages fournis.\n"
            "Règles :\n"
            "1. Si les passages ne suffisent pas, réponds exactement : "
            "Je ne sais pas d'après les documents fournis.\n"
            "2. Cite les passages utilisés avec leurs identifiants [S1], [S2], etc.\n"
            "3. N'invente ni API, ni paramètre, ni version absente des passages.\n"
            "4. Rédige une réponse claire et concise dans la langue de la question.\n\n"
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
        k = int(self.search_config.get("final_k", LLM_CONFIG.get("top_k_retrieval", 5)))
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
        prompt_context_records = self.get_prompt_contexts(contexts)
        
        sources = [{"doc_title": c.get("doc_title"), "doc_source": c.get("doc_source")} for c in contexts]
        scores = [c.get("retrieval_score", 0.0) for c in contexts]
        
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "retrieval_scores": scores,
            "retrieved_chunks": contexts,
            "prompt_contexts": [record["text"] for record in prompt_context_records],
            "prompt_context_metadata": prompt_context_records,
        }

def load_pipeline() -> RAGPipeline:
    """Charge une seule instance partageable du pipeline RAG."""
    index, chunks = load_index()
    search_config = load_search_config()
    model_name = search_config.get("embedding_model", "all-MiniLM-L6-v2")
    logger.info(f"Chargement du modèle d'embedding : {model_name}...")
    embedding_model = SentenceTransformer(model_name)
    llm_client = auto_detect_client()
    return RAGPipeline(
        llm_client=llm_client,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        search_config=search_config,
    )


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

    Pipeline complet : chargement de l'index -> détection du LLM ->
    démonstration sur 3 questions -> mode interactif (si terminal disponible).
    """
    print("\n" + "=" * 65)
    print("[ETAPE 5] RECHERCHE ET GENERATION RAG")
    print("=" * 65)
    print("\n  Ce module complète le pipeline RAG en ajoutant la génération")
    print("  de réponses via un LLM (Mistral 7B Instruct).")
    print("  Réf. Lewis et al. (2020), Jiang et al. (2023).\n")

    pipeline = load_pipeline()
    print(f"  → {len(pipeline.chunks)} chunks chargés, index FAISS de dimension {pipeline.index.d}.")
    print(f"  → Modèle d'embedding : {pipeline.search_config.get('embedding_model', 'inconnu')}")
    print(f"  → Méthode de recherche : {pipeline.search_config.get('search_method', 'semantic')}")

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
    print("[OK] Etape 5 prete")
    print("=" * 65)

    # Mode interactif : uniquement si un terminal est disponible.
    # En environnement non-interactif (Kaggle kernel, pipeline CI, etc.)
    # sys.stdin.isatty() retourne False => on n'essaie pas de lire l'input
    # ce qui eviterait de bloquer indefiniment.
    if sys.stdin.isatty():
        interactive_mode(pipeline)
    else:
        print("\n  [INFO] Mode non-interactif detecte (Kaggle / CI).")
        print("  Le mode interactif est desactive pour eviter un blocage.")
        print("  Utilisez batch_mode() ou appelez pipeline.answer(question) directement.")


if __name__ == "__main__":
    main()
