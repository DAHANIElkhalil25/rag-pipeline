"""Regression tests for final-system evaluation artifacts without ML dependencies."""

import asyncio
import json
from pathlib import Path

import pytest

from evaluation.create_annotation_candidates import create_candidates
from evaluation.dataset_schema import deterministic_id_scores, read_jsonl, validate_gold_records
from evaluation.build_candidate_aligned_review_dataset import REVISIONS
from evaluation.ragas_runner import _sentence_transformer_ragas_embeddings_class, build_judge, build_metrics
from core.scope_guard import evaluate_scope, explicit_scope_refusal


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


def test_candidate_aligned_revisions_are_complete_and_each_has_explicit_evidence_ids():
    assert len(REVISIONS) == 22
    assert all(revision["user_input"].strip() for revision in REVISIONS.values())
    assert all(revision["reference"].strip() for revision in REVISIONS.values())
    assert all(revision["chunk_ids"] for revision in REVISIONS.values())


def test_frozen_final_dataset_is_complete_balanced_and_schema_validated():
    dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "test_dataset_v1.jsonl"
    rows = read_jsonl(dataset_path)
    validate_gold_records(rows, require_validated=True)
    assert len(rows) == 120
    assert {domain: sum(row["domain"] == domain for row in rows) for domain in ("python", "scikit_learn", "langchain")} == {
        "python": 40,
        "scikit_learn": 40,
        "langchain": 40,
    }
    assert all(row["reference_context_ids"] for row in rows)


class _FakeVectors:
    def tolist(self):
        return [[0.1, 0.2], [0.3, 0.4]]


class _FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        self.calls.append((texts, normalize_embeddings, convert_to_numpy))
        return _FakeVectors()


def test_ragas_sentence_transformer_adapter_is_concrete_and_supports_modern_async_methods():
    adapter_class = _sentence_transformer_ragas_embeddings_class()
    assert adapter_class.__abstractmethods__ == frozenset()
    model = _FakeSentenceTransformer()
    adapter = adapter_class(model_name="mock-model", model=model)
    assert adapter.embed_texts(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert asyncio.run(adapter.aembed_text("question")) == [0.1, 0.2]
    assert model.calls == [
        (["a", "b"], True, True),
        (["question"], True, True),
    ]


def test_ragas_metrics_construct_with_the_independent_embedding_adapter():
    metrics = build_metrics(
        build_judge("openai", "gpt-4o-mini", "test-key-used-for-construction-only"),
        "mock-model",
        embedding_model=_FakeSentenceTransformer(),
    )
    assert set(metrics) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "factual_correctness",
    }


def test_openai_ragas_judge_uses_an_asynchronous_client():
    judge = build_judge("openai", "gpt-4o-mini", "test-key-used-for-construction-only")
    assert judge.is_async is True


def test_mistral_ragas_judge_uses_an_openai_compatible_asynchronous_client():
    judge = build_judge("mistral", "mistral-small-latest", "test-key-used-for-construction-only")
    assert judge.is_async is True
    assert judge.model == "mistral-small-latest"


SCOPE_GUARD_CONFIG = {
    "enabled": True,
    "minimum_top_score": None,
    "unsupported_aliases": {"pd": "pandas"},
    "unsupported_libraries": ["pandas", "django"],
}


def test_scope_guard_refuses_explicit_library_outside_the_documented_corpus():
    decision = evaluate_scope(
        "fait quoi pd.head ?",
        [{"retrieval_score": 0.95, "chunk_text": "Un passage trompeur."}],
        SCOPE_GUARD_CONFIG,
    )
    assert decision["allow_answer"] is False
    assert decision["reason"] == "unsupported_library"
    assert decision["detected_topic"] == "pandas"


def test_explicit_scope_precheck_refuses_pandas_before_retrieval():
    decision = explicit_scope_refusal("Comment fonctionne pandas.DataFrame.head() ?", SCOPE_GUARD_CONFIG)
    assert decision is not None
    assert decision["reason"] == "unsupported_library"

    assert explicit_scope_refusal("Comment fonctionne yield en Python ?", SCOPE_GUARD_CONFIG) is None


def test_scope_guard_refuses_when_no_document_is_retrieved():
    decision = evaluate_scope("Question inconnue", [], SCOPE_GUARD_CONFIG)
    assert decision["allow_answer"] is False
    assert decision["reason"] == "no_retrieved_context"
    assert decision["confidence"] == 0.0


def test_scope_guard_keeps_a_supported_question_with_retrieved_evidence():
    decision = evaluate_scope(
        "Quel est le rôle de yield en Python ?",
        [{"reranker_score": 0.12, "chunk_text": "yield suspend une fonction génératrice."}],
        SCOPE_GUARD_CONFIG,
    )
    assert decision == {
        "allow_answer": True,
        "reason": "sufficient_retrieved_context",
        "confidence": 0.12,
    }
