"""Convert manually completed final-test CSV annotations into validated JSONL."""

import argparse
import csv
import json
from pathlib import Path

from evaluation.dataset_schema import validate_gold_records, write_jsonl


JSON_COLUMNS = ("reference_context_ids", "reference_source_urls", "source_versions")


def convert(input_path: Path, output_path: Path, expected_count: int = 120) -> None:
    records = []
    with input_path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                record = {
                    "question_id": row["question_id"].strip(),
                    "split": row["split"].strip(),
                    "domain": row["domain"].strip(),
                    "question_type": row["question_type"].strip(),
                    "difficulty": row["difficulty"].strip(),
                    "language": row["language"].strip(),
                    "user_input": row["user_input"].strip(),
                    "reference": row["reference"].strip(),
                    "reference_context_ids": json.loads(row["reference_context_ids"]),
                    "reference_source_urls": json.loads(row["reference_source_urls"]),
                    "source_versions": json.loads(row["source_versions"]),
                    "annotation": {
                        "review_status": row["annotation_status"].strip(),
                        "annotator": row["annotator"].strip(),
                        "reviewer": row["reviewer"].strip(),
                        "notes": row["notes"].strip(),
                    },
                }
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"CSV annotation invalide, ligne {row_number}: {exc}") from exc
            records.append(record)
    if len(records) != expected_count:
        raise ValueError(f"Le test final doit contenir {expected_count} questions, reçu {len(records)}.")
    validate_gold_records(records, require_validated=True)
    write_jsonl(output_path, records)
    print(f"Validated final test set written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=120)
    args = parser.parse_args()
    convert(args.input, args.output, args.expected_count)
