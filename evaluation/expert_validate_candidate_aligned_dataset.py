"""Run a transparent AI-expert review over the candidate-aligned 120-question dataset.

The reviewer receives only the French question, reference answer, selected
candidate chunks, and their source URLs.  It cannot invent outside evidence.
Its output is an auditable *AI expert review*, never an independent human
annotation.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_schema import read_jsonl, write_jsonl


WORKSPACE = Path("/home/ubuntu")
DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "test_dataset_v1_candidate_aligned_review.jsonl"
CANDIDATES = WORKSPACE / "upload" / "candidats_120_questions(1).jsonl"
OUTPUT = PROJECT_ROOT / "evaluation" / "datasets" / "test_dataset_v1_ai_expert_review.jsonl"
SUMMARY = PROJECT_ROOT / "evaluation" / "datasets" / "test_dataset_v1_ai_expert_review_summary.json"
MODEL = os.getenv("RAG_DATASET_REVIEW_MODEL", "gpt-5")
MAX_WORKERS = int(os.getenv("RAG_DATASET_REVIEW_WORKERS", "2"))
RETRY_ERRORS_ONLY = os.getenv("RAG_DATASET_REVIEW_RETRY_ERRORS_ONLY", "true").lower() == "true"

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_dataset_expert_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string"},
                "verdict": {"type": "string", "enum": ["approve", "revise", "reject"]},
                "confidence": {"type": "number"},
                "support_type": {
                    "type": "string",
                    "enum": ["direct", "partially_direct", "insufficient"],
                },
                "critique_fr": {"type": "string"},
                "proposed_user_input": {"type": "string"},
                "proposed_reference": {"type": "string"},
            },
            "required": [
                "question_id", "verdict", "confidence", "support_type",
                "critique_fr", "proposed_user_input", "proposed_reference",
            ],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """Tu es un relecteur expert en RAG et en documentation technique.
Évalue strictement un enregistrement de dataset. Tu dois utiliser exclusivement
les passages candidats fournis. Approuve seulement si : (1) la réponse répond
directement à la question, (2) chaque affirmation importante est explicitement
soutenue par un passage sélectionné, (3) l'URL correspondante est officielle et
(4) aucun détail externe n'a été ajouté. Si une légère reformulation de la
question ou de la réponse permet de devenir exactement fidèle au passage,
choisis revise et propose cette reformulation en français. Si aucune formulation
raisonnable ne peut être soutenue, choisis reject. Ne change jamais les IDs de
chunks et ne prétends jamais être un relecteur humain indépendant."""


def _load_candidate_chunks() -> dict[str, dict]:
    chunks = {}
    for row in read_jsonl(CANDIDATES):
        for chunk in row.get("candidate_chunks", []):
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                chunks.setdefault(chunk_id, chunk)
    return chunks


def _payload(record: dict, chunks_by_id: dict[str, dict]) -> str:
    evidence = []
    for chunk_id in record["reference_context_ids"]:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Evidence chunk unavailable: {record['question_id']} / {chunk_id}")
        evidence.append({
            "chunk_id": chunk_id,
            "doc_url": chunk.get("doc_url", ""),
            "text_preview": chunk.get("text_preview", ""),
        })
    return json.dumps({
        "question_id": record["question_id"],
        "domain": record["domain"],
        "question": record["user_input"],
        "reference_answer": record["reference"],
        "selected_evidence": evidence,
    }, ensure_ascii=False)


def _review_one(record: dict, chunks_by_id: dict[str, dict]) -> dict:
    client = OpenAI()
    content = _payload(record, chunks_by_id)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=RESPONSE_SCHEMA,
                max_completion_tokens=900,
            )
            content = response.choices[0].message.content if response.choices else None
            if not content:
                raise RuntimeError("Empty structured response from expert-review model")
            result = json.loads(content)
            if result["question_id"] != record["question_id"]:
                raise ValueError("Question ID mismatch in expert review")
            result["review_model"] = MODEL
            result["review_protocol"] = "ai_expert_direct_evidence_v1"
            return result
        except Exception as exc:  # transient proxy failures are retried
            if attempt == 2:
                return {
                    "question_id": record["question_id"],
                    "verdict": "reject",
                    "confidence": 0.0,
                    "support_type": "insufficient",
                    "critique_fr": f"Erreur de revue à conserver : {type(exc).__name__}: {exc}",
                    "proposed_user_input": "",
                    "proposed_reference": "",
                    "review_model": MODEL,
                    "review_protocol": "ai_expert_direct_evidence_v1",
                }
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def main() -> None:
    records = read_jsonl(DATASET)
    if len(records) != 120:
        raise ValueError(f"Expected 120 records, found {len(records)}")
    chunks_by_id = _load_candidate_chunks()
    prior_reviews = {
        row["question_id"]: row
        for row in read_jsonl(OUTPUT)
    } if OUTPUT.exists() and RETRY_ERRORS_ONLY else {}
    records_to_review = [
        record for record in records
        if "Erreur de revue" in prior_reviews.get(record["question_id"], {}).get("critique_fr", "")
        or record["question_id"] not in prior_reviews
    ]
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fresh_reviews = list(executor.map(lambda record: _review_one(record, chunks_by_id), records_to_review))
    fresh_by_id = {review["question_id"]: review for review in fresh_reviews}
    reviews = [fresh_by_id.get(record["question_id"], prior_reviews[record["question_id"]]) for record in records]
    write_jsonl(OUTPUT, reviews)
    counts = {key: sum(review["verdict"] == key for review in reviews) for key in ("approve", "revise", "reject")}
    summary = {
        "dataset": str(DATASET),
        "candidate_source": str(CANDIDATES),
        "review_model": MODEL,
        "review_protocol": "ai_expert_direct_evidence_v1",
        "record_count": len(reviews),
        "verdict_counts": counts,
        "non_approved_question_ids": [review["question_id"] for review in reviews if review["verdict"] != "approve"],
        "limitation": "This is an AI-expert review. It is not represented as independent human annotation.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
