"""Build a 120-record review dataset from already generated Kaggle candidates.

This script does *not* freeze a final Ragas test set.  It creates a complete,
source-traceable review artifact whose evidence IDs come from the user's
uploaded candidate file.  Every row remains explicitly pending human approval.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_schema import read_jsonl, validate_gold_records, write_jsonl


WORKSPACE = Path("/home/ubuntu")
DEFAULT_DRAFT = PROJECT_ROOT / "evaluation" / "datasets" / "test_dataset_v1_source_grounded_draft.jsonl"
DEFAULT_CANDIDATES = WORKSPACE / "upload" / "candidats_120_questions(1).jsonl"
DEFAULT_DECISIONS = WORKSPACE / "candidate_audit" / "consolidated_evidence_decisions_v2.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "datasets" / "test_dataset_v1_candidate_aligned_review.jsonl"
DEFAULT_CSV = PROJECT_ROOT / "evaluation" / "datasets" / "test_dataset_v1_candidate_aligned_review.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "evaluation" / "datasets" / "test_dataset_v1_candidate_aligned_review_summary.json"


# Each replacement is deliberately narrower than the original unsupported
# question and is phrased only from a passage actually supplied by Kaggle.
REVISIONS = {
    "test_python_004": {
        "user_input": "Qu'est-ce que le duck typing en Python ?",
        "reference": "Le duck typing est un style qui ne vérifie pas le type d'un objet : il appelle ou utilise directement la méthode ou l'attribut attendu par l'interface.",
        "chunk_ids": ["python_glossary_rst_b2b8da270c54_chunk_0008"],
    },
    "test_python_015": {
        "user_input": "Pourquoi le module copy est-il utile pour certaines collections mutables ?",
        "reference": "Lorsqu'une collection est mutable ou contient des éléments mutables, une copie peut permettre de modifier une copie sans modifier l'autre.",
        "chunk_ids": ["python_library_copy_rst_d8bae0fd6604_chunk_0000"],
    },
    "test_python_029": {
        "user_input": "Comment une annotation peut-elle décrire le type d'éléments d'un ensemble ?",
        "reference": "Une annotation telle que set[bytes] peut indiquer que les éléments d'un ensemble sont de type bytes.",
        "chunk_ids": ["python_library_stdtypes_rst_d34a8da841ea_chunk_0108"],
    },
    "test_python_034": {
        "user_input": "Quels objets peut-on annoter avec collections.abc.Callable ?",
        "reference": "collections.abc.Callable peut annoter des fonctions ou d'autres objets appelables.",
        "chunk_ids": ["python_library_typing_rst_5ee6444b07cb_chunk_0004"],
    },
    "test_python_038": {
        "user_input": "Dans l'AST Python, que contiennent les champs keys et values d'un nœud dictionnaire ?",
        "reference": "Ils contiennent des listes de nœuds représentant respectivement les clés et les valeurs, dans l'ordre correspondant.",
        "chunk_ids": ["python_library_ast_rst_17b2ca98b5e8_chunk_0007"],
    },
    "test_scikit_learn_005": {
        "user_input": "Sur quel type d'estimations la log loss est-elle définie ?",
        "reference": "La log loss est définie sur des estimations de probabilités.",
        "chunk_ids": ["sklearn_modules_model_evaluation_rst_5f512b0b9c3a_chunk_0036"],
    },
    "test_scikit_learn_006": {
        "user_input": "Quelles méthodes un transformateur scikit-learn peut-il fournir pour modifier les données ?",
        "reference": "Un transformateur applique transform aux données ; lorsque l'ajustement et la transformation sont plus efficaces ensemble, il peut fournir fit_transform.",
        "chunk_ids": ["sklearn_developers_develop_rst_c26065a88e49_chunk_0001"],
    },
    "test_scikit_learn_016": {
        "user_input": "Comment GridSearchCV et RandomizedSearchCV peuvent-ils évaluer plusieurs métriques ?",
        "reference": "Les deux acceptent plusieurs métriques via le paramètre scoring, sous forme de liste de noms ou de dictionnaire ; avec plusieurs métriques, refit doit désigner la métrique utilisée pour le réajustement.",
        "chunk_ids": ["sklearn_modules_grid_search_rst_9c887a577866_chunk_0018"],
    },
    "test_scikit_learn_023": {
        "user_input": "Quel contrôle ShuffleSplit offre-t-il pour une validation croisée ?",
        "reference": "ShuffleSplit permet un contrôle plus fin du nombre d'itérations et de la proportion d'échantillons de chaque côté de la séparation entraînement/test.",
        "chunk_ids": ["sklearn_modules_cross_validation_rst_3862e505f662_chunk_0013"],
    },
    "test_scikit_learn_031": {
        "user_input": "Quelles contraintes portent sur les étapes d'un Pipeline scikit-learn ?",
        "reference": "Toutes les étapes sauf la dernière doivent être des transformateurs avec une méthode transform ; la dernière peut être de tout type compatible, notamment un classifieur.",
        "chunk_ids": ["sklearn_modules_compose_rst_5a7d65afadd9_chunk_0001"],
    },
    "test_langchain_001": {
        "user_input": "Comment LangChain est-il décrit dans la documentation d'intégration ?",
        "reference": "LangChain y est décrit comme un framework destiné à construire des applications de raisonnement conscientes du contexte.",
        "chunk_ids": ["langchain_src_oss_python_integrations_embeddings_ollama_mdx_01d3036bad7b_chunk_0006"],
    },
    "test_langchain_005": {
        "user_input": "Quel rôle Bigtable joue-t-il dans l'exemple SimpleKVStoreRetriever ?",
        "reference": "Bigtable y sert de couche de persistance documentaire qui récupère les documents correspondant à un préfixe de requête.",
        "chunk_ids": ["langchain_src_oss_python_integrations_stores_bigtable_mdx_0e9506fb1b10_chunk_0018"],
    },
    "test_langchain_015": {
        "user_input": "Quelle méthode convertit un vector store en retriever dans l'exemple LangChain ?",
        "reference": "La méthode as_retriever convertit le vector store en VectorStoreRetriever et permet de régler la stratégie et les paramètres de recherche.",
        "chunk_ids": ["langchain_src_oss_python_integrations_vectorstores_pinecone_sparse_mdx_9460a82e9f32_chunk_0009"],
    },
    "test_langchain_017": {
        "user_input": "À quelles étapes d'un flux RAG les modèles d'embeddings sont-ils utilisés ?",
        "reference": "Ils sont utilisés lors de l'indexation des données puis plus tard lors de leur récupération.",
        "chunk_ids": ["langchain_src_oss_python_integrations_embeddings_mistralai_mdx_5d4a143b1226_chunk_0005"],
    },
    "test_langchain_018": {
        "user_input": "Quel paramètre limite le nombre de nouveaux jetons générés dans l'exemple HuggingFacePipeline ?",
        "reference": "L'exemple règle max_new_tokens à 512.",
        "chunk_ids": ["langchain_src_oss_python_integrations_chat_huggingface_mdx_629f265eacce_chunk_0010"],
    },
    "test_langchain_027": {
        "user_input": "Comment convertir un modèle RAGatouille en retriever compatible LangChain ?",
        "reference": "L'exemple appelle RAG.as_langchain_retriever(k=3), puis le retriever obtenu peut être invoqué avec une requête.",
        "chunk_ids": ["langchain_src_oss_python_integrations_retrievers_ragatouille_mdx_671ef58388ef_chunk_0010"],
    },
    "test_langchain_031": {
        "user_input": "Comment l'exemple configure-t-il un identifiant de fil pour une invocation d'agent ?",
        "reference": "Il place thread_id dans le dictionnaire config sous la clé configurable, par exemple {\"configurable\": {\"thread_id\": \"abc123\"}}.",
        "chunk_ids": ["langchain_src_oss_python_integrations_chat_anthropic_mdx_a087d6ccf479_chunk_0049"],
    },
    "test_langchain_033": {
        "user_input": "Comment rendre un vector store plus facile à utiliser dans une chaîne LangChain ?",
        "reference": "L'exemple transforme le vector store en retriever avec as_retriever, puis appelle invoke sur ce retriever.",
        "chunk_ids": ["langchain_src_oss_python_integrations_vectorstores_turbopuffer_mdx_718b7bad3c6d_chunk_0009"],
    },
    "test_langchain_034": {
        "user_input": "Quelle contrainte le prompt de l'exemple de chaîne RAG impose-t-il à la réponse ?",
        "reference": "Il demande de répondre à la question uniquement à partir du contexte fourni.",
        "chunk_ids": ["langchain_src_oss_python_integrations_retrievers_perplexity_search_mdx_7b4b2bc1cc32_chunk_0007"],
    },
    "test_langchain_035": {
        "user_input": "Quelle recommandation la documentation donne-t-elle avant de choisir un modèle d'embeddings à partir du classement MTEB ?",
        "reference": "Elle recommande d'exécuter une petite évaluation sur ses propres données avant de s'engager, car les résultats du classement ne se transfèrent pas toujours.",
        "chunk_ids": ["langchain_src_oss_python_integrations_embeddings_index_mdx_aa9161d191c8_chunk_0005"],
    },
    "test_langchain_038": {
        "user_input": "Comment l'exemple crée-t-il un agent qui peut répondre à une question météo ?",
        "reference": "Il appelle create_agent avec un modèle, la liste tools contenant weather_tool et un format de réponse, puis invoque l'agent avec un message utilisateur.",
        "chunk_ids": ["langchain_src_oss_python_integrations_chat_anthropic_mdx_a087d6ccf479_chunk_0056"],
    },
    "test_langchain_040": {
        "user_input": "Pourquoi le découpage des documents est-il important pour la récupération documentaire ?",
        "reference": "Un découpage approprié est critique pour la récupération ; plusieurs techniques existent, notamment selon les espaces ou le découpage récursif par longueur de caractères.",
        "chunk_ids": ["langchain_src_oss_python_integrations_document_loaders_docugami_mdx_f8991a269783_chunk_0002"],
    },
}


def _candidate_lookup(candidate_rows: list[dict]) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    by_question = {
        row["question_id"]: {chunk["chunk_id"]: chunk for chunk in row.get("candidate_chunks", [])}
        for row in candidate_rows
    }
    by_id = {
        chunk["chunk_id"]: chunk
        for row in candidate_rows
        for chunk in row.get("candidate_chunks", [])
    }
    return by_question, by_id


def build_candidate_aligned_review_dataset(
    draft_path: Path = DEFAULT_DRAFT,
    candidates_path: Path = DEFAULT_CANDIDATES,
    decisions_path: Path = DEFAULT_DECISIONS,
    output_path: Path = DEFAULT_OUTPUT,
    csv_path: Path = DEFAULT_CSV,
    summary_path: Path = DEFAULT_SUMMARY,
) -> list[dict]:
    drafts = read_jsonl(draft_path)
    candidates, chunks_by_id = _candidate_lookup(read_jsonl(candidates_path))
    decisions = {row["question_id"]: row for row in read_jsonl(decisions_path)}
    records = []
    revision_count = 0

    for draft in drafts:
        question_id = draft["question_id"]
        decision = decisions.get(question_id)
        if decision is None:
            raise ValueError(f"Missing evidence decision for {question_id}")
        revision = REVISIONS.get(question_id)
        if revision:
            revision_count += 1
            selected_ids = revision["chunk_ids"]
            question = revision["user_input"]
            reference = revision["reference"]
            provenance = "question_reference_rewritten_from_available_candidate"
        else:
            selected_ids = decision["selected_chunk_ids"]
            question = draft["user_input"]
            reference = draft["reference"]
            provenance = "original_question_reference_retained"
        if not selected_ids:
            raise ValueError(f"No evidence IDs for {question_id}")
        chunk_map = candidates.get(question_id, {})
        evidence_chunks = {
            chunk_id: chunk_map.get(chunk_id, chunks_by_id.get(chunk_id))
            for chunk_id in selected_ids
        }
        missing = [chunk_id for chunk_id, chunk in evidence_chunks.items() if chunk is None]
        if missing:
            raise ValueError(f"Selected chunks unavailable from all uploaded candidates for {question_id}: {missing}")
        urls = sorted({chunk["doc_url"] for chunk in evidence_chunks.values() if chunk.get("doc_url")})
        record = {
            **draft,
            "user_input": question,
            "reference": reference,
            "reference_context_ids": selected_ids,
            "reference_source_urls": urls,
            "annotation": {
                "review_status": "ai_prevalidated_pending_human_review",
                "annotator": "Manus AI",
                "reviewer": "",
                "notes": (
                    "Evidence IDs originate from the uploaded Kaggle candidate file. "
                    f"{provenance}. Review pass {decision['review_pass']}; "
                    f"evidence confidence {decision['confidence']}. "
                    "Human researcher approval is still required before final Ragas conversion."
                ),
            },
        }
        records.append(record)

    if len(records) != 120:
        raise ValueError(f"Expected 120 records, found {len(records)}")
    if len(REVISIONS) != revision_count:
        raise ValueError("Not all declared revisions were applied")
    validate_gold_records(records, require_validated=False)
    write_jsonl(output_path, records)

    fields = [
        "question_id", "domain", "question_type", "difficulty", "user_input",
        "reference", "reference_context_ids", "reference_source_urls", "changed_from_draft",
        "annotation_status", "researcher_confirmation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "question_id": record["question_id"],
                "domain": record["domain"],
                "question_type": record["question_type"],
                "difficulty": record["difficulty"],
                "user_input": record["user_input"],
                "reference": record["reference"],
                "reference_context_ids": json.dumps(record["reference_context_ids"], ensure_ascii=False),
                "reference_source_urls": json.dumps(record["reference_source_urls"], ensure_ascii=False),
                "changed_from_draft": str(record["question_id"] in REVISIONS).lower(),
                "annotation_status": record["annotation"]["review_status"],
                "researcher_confirmation": "",
            })

    summary = {
        "record_count": len(records),
        "revised_record_count": revision_count,
        "retained_record_count": len(records) - revision_count,
        "domain_counts": {domain: sum(record["domain"] == domain for record in records) for domain in ("python", "scikit_learn", "langchain")},
        "status": "ai_prevalidated_pending_human_review",
        "final_ragas_eligible": False,
        "reason": "The record-level evidence is sourced from uploaded Kaggle candidates, but no human researcher has signed the annotations yet.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    rows = build_candidate_aligned_review_dataset()
    print(f"Built {len(rows)} candidate-aligned review records at {DEFAULT_OUTPUT}")
