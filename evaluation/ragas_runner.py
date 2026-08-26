"""Final Ragas v0.4 evaluation runner with resumable Kaggle artifacts.

The baseline custom evaluator remains in ``legacy/``. This module is the sole
entrypoint for final-system evaluation and writes every observed sample before
aggregate metrics are calculated.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from config import EVALUATION_RUNS_DIR, RAGAS_CONFIG
from evaluation.dataset_schema import (
    deterministic_id_scores,
    read_jsonl,
    validate_gold_records,
    write_jsonl,
)

if TYPE_CHECKING:
    from etape5_generation import RAGPipeline


METRIC_FIELDS = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "retrieved_contexts", "reference"),
    "context_recall": ("user_input", "retrieved_contexts", "reference"),
    "factual_correctness": ("response", "reference"),
}


def _sentence_transformer_ragas_embeddings_class():
    """Return a concrete *modern* Ragas embedding adapter.

    Ragas 0.4.3 collection metrics require ``BaseRagasEmbedding`` (singular),
    whereas the packaged Hugging Face class implements the older, abstract
    ``BaseRagasEmbeddings`` interface and is additionally constrained by a
    Pydantic dataclass. This adapter implements the modern official contract,
    reuses the SentenceTransformer already loaded by the RAG pipeline, and
    avoids both compatibility failures and a duplicate GPU model load.
    """
    from ragas.embeddings import BaseRagasEmbedding

    class SentenceTransformerRagasEmbeddings(BaseRagasEmbedding):
        def __init__(self, model_name: str, model: Any | None = None):
            super().__init__()
            self.model_name = model_name
            if model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name)
            self.model = model

        def embed_text(self, text: str, **_: Any) -> list[float]:
            return self.embed_texts([text])[0]

        def embed_texts(self, texts: list[str], **_: Any) -> list[list[float]]:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vectors.tolist()

        async def aembed_text(self, text: str, **_: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

        async def aembed_texts(self, texts: list[str], **_: Any) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts)

    return SentenceTransformerRagasEmbeddings


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _manifest(run_id: str, dataset_path: Path, provider: str, judge_model: str, pipeline: RAGPipeline) -> dict[str, Any]:
    return {
        "schema_version": "final_ragas_run_v1",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "judge_provider": provider,
        "judge_model": judge_model,
        "ragas_config": RAGAS_CONFIG,
        "generation_model": getattr(pipeline.llm_client, "model_name", None),
        "retrieval_config": pipeline.search_config,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("ragas", "torch", "transformers", "sentence-transformers", "faiss-cpu")
        },
        "ragas_embedding_adapter": "sentence_transformer_base_ragas_adapter_v2",
    }


def build_judge(provider: str, model: str, api_key: str):
    """Build a current Ragas judge from a supported provider client."""
    if not api_key:
        raise ValueError(
            "Aucune clé pour le modèle juge. Configurez une variable Kaggle Secret "
            "(OPENAI_API_KEY ou MISTRAL_API_KEY) avant l'évaluation finale."
        )

    from ragas.llms import llm_factory

    provider = provider.lower().strip()
    if provider == "openai":
        from openai import OpenAI
        return llm_factory(model, provider="openai", client=OpenAI(api_key=api_key))
    if provider == "mistral":
        from mistralai import Mistral
        return llm_factory(model, provider="mistral", client=Mistral(api_key=api_key))
    raise ValueError("Provider non supporté. Utilisez 'openai' ou 'mistral'.")


def build_metrics(judge_llm, embedding_model_name: str, embedding_model: Any | None = None) -> dict[str, Any]:
    """Instantiate official Ragas collection metrics for a final experiment."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    evaluator_embeddings = _sentence_transformer_ragas_embeddings_class()(
        model_name=embedding_model_name,
        model=embedding_model,
    )
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=evaluator_embeddings),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
        "factual_correctness": FactualCorrectness(llm=judge_llm),
    }


def _metric_result_value(result: Any) -> tuple[float | None, str | None]:
    value = getattr(result, "value", result)
    value = None if value is None else float(value)
    if value is not None and not math.isfinite(value):
        value = None
    reason = getattr(result, "reason", None)
    return value, str(reason) if reason else None


def _sample_from_pipeline(gold: dict[str, Any], pipeline: RAGPipeline) -> dict[str, Any]:
    started = time.perf_counter()
    result = pipeline.answer(gold["user_input"])
    chunks = result.get("retrieved_chunks", [])
    retrieved_contexts = result.get("prompt_contexts") or [
        chunk.get("chunk_text", "") for chunk in chunks if chunk.get("chunk_text", "").strip()
    ]
    retrieved_context_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks if chunk.get("chunk_id")]
    return {
        **gold,
        "response": result.get("answer", ""),
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "doc_source": chunk.get("doc_source"),
                "doc_url": chunk.get("doc_url"),
                "rank": rank,
                "retrieval_score": chunk.get("retrieval_score"),
                "chunk_content_sha256": chunk.get("chunk_content_sha256"),
            }
            for rank, chunk in enumerate(chunks, start=1)
        ],
        "prompt_context_metadata": result.get("prompt_context_metadata", []),
        "run_timing": {"pipeline_seconds": round(time.perf_counter() - started, 4)},
    }


async def _score_sample(sample: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, float | None] = {}
    reasons: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    for name, metric in metrics.items():
        args = {field: sample[field] for field in METRIC_FIELDS[name]}
        try:
            result = await metric.ascore(**args)
            scores[name], reasons[name] = _metric_result_value(result)
        except Exception as exc:  # Error is persisted; it is never converted to a mid-score.
            scores[name] = None
            reasons[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
    sample["ragas_metrics"] = scores
    sample["ragas_reasons"] = reasons
    sample["ragas_errors"] = errors
    sample["id_metrics"] = deterministic_id_scores(
        sample.get("retrieved_context_ids", []), sample.get("reference_context_ids", [])
    )
    sample["status"] = "valid" if not errors else "partial" if any(value is not None for value in scores.values()) else "failed"
    return sample


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [*METRIC_FIELDS, "id_context_precision", "id_context_recall"]
    overall: dict[str, dict[str, float | int | None]] = {}
    by_domain: dict[str, dict[str, dict[str, float | int | None]]] = {}

    def aggregate(records: list[dict[str, Any]], metric_name: str) -> dict[str, float | int | None]:
        values = []
        for record in records:
            group = record.get("id_metrics", {}) if metric_name.startswith("id_") else record.get("ragas_metrics", {})
            value = group.get(metric_name)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return {
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
            "n_valid": len(values),
            "n_total": len(records),
            "coverage": len(values) / len(records) if records else 0.0,
        }

    for metric_name in metric_names:
        overall[metric_name] = aggregate(samples, metric_name)
    for domain in sorted({record["domain"] for record in samples}):
        domain_records = [record for record in samples if record["domain"] == domain]
        by_domain[domain] = {metric_name: aggregate(domain_records, metric_name) for metric_name in metric_names}
    return {
        "metric_summary": overall,
        "by_domain": by_domain,
        "sample_status": {
            status: sum(record.get("status") == status for record in samples)
            for status in ("valid", "partial", "failed")
        },
    }


def _write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    fields = [
        "question_id", "split", "domain", "question_type", "difficulty", "language",
        "user_input", "reference", "response", "status", "retrieved_context_ids",
        *METRIC_FIELDS.keys(), "id_context_precision", "id_context_recall", "ragas_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {field: sample.get(field) for field in fields}
            row.update(sample.get("ragas_metrics", {}))
            row.update(sample.get("id_metrics", {}))
            row["retrieved_context_ids"] = json.dumps(sample.get("retrieved_context_ids", []), ensure_ascii=False)
            row["ragas_errors"] = json.dumps(sample.get("ragas_errors", {}), ensure_ascii=False)
            writer.writerow(row)


async def run_final_evaluation(
    pipeline: RAGPipeline,
    dataset_path: Path,
    run_id: str,
    provider: str,
    judge_model: str,
    api_key: str,
    output_root: Path = EVALUATION_RUNS_DIR,
) -> dict[str, Any]:
    """Run a resumable final-system Ragas experiment and save raw artifacts."""
    gold_records = read_jsonl(dataset_path)
    is_final_test = all(record.get("split") == "test" for record in gold_records)
    validate_gold_records(gold_records, require_validated=is_final_test)

    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "samples.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    csv_path = run_dir / "samples.csv"

    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(_manifest(run_id, dataset_path, provider, judge_model, pipeline), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    completed = {record["question_id"]: record for record in read_jsonl(samples_path)} if samples_path.exists() else {}
    judge_llm = build_judge(provider, judge_model, api_key)
    embedding_model_name = pipeline.search_config.get("embedding_model", "all-MiniLM-L6-v2")
    metrics = build_metrics(
        judge_llm,
        embedding_model_name,
        embedding_model=getattr(pipeline, "embedding_model", None),
    )
    checkpoint_every = int(RAGAS_CONFIG.get("checkpoint_every", 10))

    for position, gold in enumerate(gold_records, start=1):
        if gold["question_id"] in completed:
            continue
        sample = _sample_from_pipeline(gold, pipeline)
        sample = await _score_sample(sample, metrics)
        completed[sample["question_id"]] = sample
        if position % checkpoint_every == 0 or position == len(gold_records):
            ordered = [completed[item["question_id"]] for item in gold_records if item["question_id"] in completed]
            write_jsonl(samples_path, ordered)

    ordered_samples = [completed[item["question_id"]] for item in gold_records]
    report = _summary(ordered_samples)
    report["run_id"] = run_id
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(csv_path, ordered_samples)
    return {"run_dir": str(run_dir), "summary": report, "samples": ordered_samples}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the final official-Ragas-compatible evaluation.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--provider", choices=("openai", "mistral"), required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--api-key-env", required=True)
    args = parser.parse_args()

    from etape5_generation import load_pipeline

    api_key = os.getenv(args.api_key_env, "")
    result = asyncio.run(
        run_final_evaluation(
            pipeline=load_pipeline(),
            dataset_path=args.dataset,
            run_id=args.run_id,
            provider=args.provider,
            judge_model=args.judge_model,
            api_key=api_key,
        )
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
