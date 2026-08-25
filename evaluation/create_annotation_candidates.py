"""Create human-reviewable candidate chunks for final-test source annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_schema import read_jsonl, write_jsonl


def create_candidates(pipeline, dataset_path: Path, output_path: Path, k: int = 10) -> None:
    """Persist retrieval candidates; reviewers choose IDs rather than accepting them blindly."""
    rows = []
    for position, record in enumerate(read_jsonl(dataset_path), start=1):
        contexts = pipeline.retrieve(record["user_input"], k=k)
        rows.append({
            "question_id": record["question_id"],
            "domain": record["domain"],
            "user_input": record["user_input"],
            "reference": record["reference"],
            "reference_source_urls": record["reference_source_urls"],
            "candidate_chunks": [
                {
                    "rank": rank,
                    "chunk_id": context.get("chunk_id"),
                    "document_id": context.get("document_id"),
                    "doc_url": context.get("doc_url"),
                    "doc_section": context.get("doc_section"),
                    "retrieval_score": context.get("retrieval_score"),
                    "text_preview": context.get("chunk_text", "")[:700],
                }
                for rank, context in enumerate(contexts, start=1)
            ],
            "review_instruction": "Select only chunks that directly support the reference answer; do not accept candidates automatically.",
        })
        print(f"Prepared {position}/{len(rows)}: {record['question_id']}")
    write_jsonl(output_path, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    from etape5_generation import load_pipeline

    create_candidates(load_pipeline(), args.dataset, args.output, args.k)
