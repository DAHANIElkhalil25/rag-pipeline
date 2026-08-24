"""Validation and serialization for the final RAG evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_GOLD_FIELDS = {
    "question_id", "split", "domain", "question_type", "difficulty",
    "language", "user_input", "reference", "reference_context_ids",
    "reference_source_urls", "source_versions", "annotation",
}
VALID_SPLITS = {"dev", "validation", "test"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records with a helpful error message."""
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON invalide dans {path}, ligne {line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records) + "\n",
        encoding="utf-8",
    )


def validate_gold_records(records: list[dict[str, Any]], require_validated: bool) -> None:
    """Validate the manual ground-truth dataset before a Ragas experiment starts."""
    if not records:
        raise ValueError("Le jeu d'évaluation est vide.")

    seen_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = REQUIRED_GOLD_FIELDS.difference(record)
        if missing:
            raise ValueError(f"Question {index}: champs obligatoires absents: {sorted(missing)}")
        question_id = str(record["question_id"]).strip()
        if not question_id or question_id in seen_ids:
            raise ValueError(f"Question {index}: question_id vide ou dupliqué: {question_id!r}")
        seen_ids.add(question_id)
        if record["split"] not in VALID_SPLITS:
            raise ValueError(f"{question_id}: split invalide {record['split']!r}")
        if not str(record["user_input"]).strip() or not str(record["reference"]).strip():
            raise ValueError(f"{question_id}: question et référence doivent être renseignées.")
        if not isinstance(record["reference_context_ids"], list):
            raise ValueError(f"{question_id}: reference_context_ids doit être une liste.")
        if require_validated:
            status = record.get("annotation", {}).get("review_status")
            if status != "validated":
                raise ValueError(f"{question_id}: le test final exige review_status='validated'.")
            if not record["reference_context_ids"]:
                raise ValueError(f"{question_id}: le test final exige des reference_context_ids annotés.")


def deterministic_id_scores(retrieved_ids: list[str], reference_ids: list[str]) -> dict[str, float | None]:
    """Compute transparent ID-based precision and recall when annotations exist."""
    reference_set = set(reference_ids)
    retrieved_set = set(retrieved_ids)
    if not reference_set:
        return {"id_context_precision": None, "id_context_recall": None}
    overlap = retrieved_set.intersection(reference_set)
    return {
        "id_context_precision": len(overlap) / len(retrieved_ids) if retrieved_ids else 0.0,
        "id_context_recall": len(overlap) / len(reference_set),
    }
