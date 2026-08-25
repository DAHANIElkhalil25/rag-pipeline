"""Regression tests for final-system evaluation artifacts without ML dependencies."""

import json

import pytest

from evaluation.create_annotation_candidates import create_candidates
from evaluation.dataset_schema import deterministic_id_scores, validate_gold_records


def make_record(split="dev", review_status="needs_context_annotation", context_ids=None):
    return {
        "question_id": "sample_001",
        "split": split,
        "domain": "python",
        "question_type": "factual",
        "difficulty": "easy",
        "language": "fr",
        "user_input": "Quelle est la portée d'une variable ?",
        "reference": "La portée définit où une variable peut être utilisée.",
        "reference_context_ids": context_ids if context_ids is not None else [],
        "reference_source_urls": [],
        "source_versions": {},
        "annotation": {"review_status": review_status},
    }


def test_development_records_do_not_require_final_context_annotation():
    validate_gold_records([make_record()], require_validated=False)


def test_final_records_require_reviewer_validation_and_context_ids():
    with pytest.raises(ValueError, match="review_status='validated'"):
        validate_gold_records([make_record(split="test")], require_validated=True)

    with pytest.raises(ValueError, match="reference_context_ids"):
        validate_gold_records(
            [make_record(split="test", review_status="validated", context_ids=[])],
            require_validated=True,
        )

    validate_gold_records(
        [make_record(split="test", review_status="validated", context_ids=["python_scope_chunk_001"])],
        require_validated=True,
    )


def test_deterministic_context_id_scores_are_rank_agnostic_and_transparent():
    scores = deterministic_id_scores(
        ["chunk_1", "chunk_2", "chunk_3"],
        ["chunk_2", "chunk_4"],
    )
    assert scores["id_context_precision"] == pytest.approx(1 / 3)
    assert scores["id_context_recall"] == pytest.approx(1 / 2)


def test_deterministic_scores_are_absent_when_gold_context_ids_are_not_annotated():
    assert deterministic_id_scores(["chunk_1"], []) == {
        "id_context_precision": None,
        "id_context_recall": None,
    }


class _CandidatePipeline:
    def __init__(self):
        self.calls = []
        self.chunks = [{"doc_url": "https://docs.python.org/3/glossary.html"}]

    def retrieve(self, question, k, source_domain, source_urls):
        self.calls.append({
            "question": question,
            "k": k,
            "source_domain": source_domain,
            "source_urls": source_urls,
        })
        return [{
            "chunk_id": "python_glossary_chunk_001",
            "document_id": "python_glossary",
            "doc_url": source_urls[0],
            "doc_section": "other",
            "retrieval_score": 0.9,
            "chunk_text": "An iterator yields successive values and raises StopIteration.",
        }]


def test_annotation_candidates_forward_the_official_source_url_to_retrieval(tmp_path):
    dataset_path = tmp_path / "draft.jsonl"
    output_path = tmp_path / "candidates.jsonl"
    source_url = "https://docs.python.org/3/glossary.html"
    dataset_path.write_text(json.dumps({
        "question_id": "test_python_001",
        "domain": "python",
        "user_input": "Définir un itérateur",
        "reference": "Un itérateur fournit des valeurs successives.",
        "reference_source_urls": [source_url],
    }) + "\n", encoding="utf-8")
    pipeline = _CandidatePipeline()

    create_candidates(pipeline, dataset_path, output_path, k=4)

    assert pipeline.calls == [{
        "question": "Définir un itérateur",
        "k": 4,
        "source_domain": "python",
        "source_urls": [source_url],
    }]
