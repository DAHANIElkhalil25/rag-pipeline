import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def evaluation_module():
    config = types.ModuleType("config")
    config.PROCESSED_DIR = Path(".")
    config.VECTORSTORE_DIR = Path(".")
    config.BENCHMARK_DIR = Path(".")
    config.EVALUATION_DIR = Path(".")
    config.LLM_CONFIG = {"top_k_retrieval": 5}
    config.EVALUATION_CONFIG = {
        "max_claims_per_answer": 8,
        "max_contexts_per_sample": 5,
        "cache_judgments": True,
    }
    config.logger = logging.getLogger("test_evaluation")
    old_config = sys.modules.get("config")
    old_step5 = sys.modules.get("etape5_generation")
    sys.modules["config"] = config

    step5 = types.ModuleType("etape5_generation")
    step5.load_index = lambda: (None, [])
    step5.load_search_config = lambda: {}
    step5.auto_detect_client = lambda: None
    step5.RAGPipeline = object
    sys.modules["etape5_generation"] = step5

    spec = importlib.util.spec_from_file_location(
        "evaluation_under_test",
        Path(__file__).parents[1] / "etape6_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module

    if old_config is None:
        sys.modules.pop("config", None)
    else:
        sys.modules["config"] = old_config
    if old_step5 is None:
        sys.modules.pop("etape5_generation", None)
    else:
        sys.modules["etape5_generation"] = old_step5


def test_boolean_parser_rejects_quoted_false_as_true(evaluation_module):
    assert evaluation_module.parse_boolean_from_llm('{"supported": "false"}') is False
    assert evaluation_module.parse_boolean_from_llm('{"supported": false}') is False


def test_boolean_parser_rejects_ambiguous_prose(evaluation_module):
    with pytest.raises(ValueError):
        evaluation_module.parse_boolean_from_llm("The context does not clearly support this claim.")


def test_context_precision_uses_evaluated_contexts_and_rank(evaluation_module):
    class OrderedLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            return "false" if self.calls == 1 else "true"

    class Pipeline:
        embedding_model = None

    evaluator = evaluation_module.RAGASEvaluator(Pipeline(), OrderedLLM())
    score = evaluator.evaluate_context_precision(
        "question", ["first", "second"], "reference"
    )
    assert score == pytest.approx(0.5)


def test_context_precision_does_not_divide_by_unjudged_contexts(evaluation_module):
    class AlwaysTrueLLM:
        def generate(self, prompt):
            return "true"

    class Pipeline:
        embedding_model = None

    evaluator = evaluation_module.RAGASEvaluator(Pipeline(), AlwaysTrueLLM())
    score = evaluator.evaluate_context_precision(
        "question", [f"context {i}" for i in range(6)], "reference"
    )
    assert score == pytest.approx(1.0)


def test_report_serializes_invalid_metrics_as_empty_csv_fields(evaluation_module, tmp_path):
    old_dir = evaluation_module.EVALUATION_DIR
    evaluation_module.EVALUATION_DIR = tmp_path
    try:
        results = {
            "mean_metrics": {"faithfulness": None},
            "details": [{
                "question": "q",
                "source_filter": "Python",
                "status": "partial",
                "error": None,
                "generated_answer": "a",
                "contexts": [],
                "metrics": {
                    "faithfulness": None,
                    "answer_relevancy": 0.5,
                    "context_precision": 0.0,
                    "context_recall": None,
                },
            }],
        }
        evaluation_module.generate_evaluation_report(results, 1.25)
        assert (tmp_path / "ragas_report.json").exists()
        csv_text = (tmp_path / "ragas_details.csv").read_text(encoding="utf-8")
        assert "Status" in csv_text
        assert "partial" in csv_text
    finally:
        evaluation_module.EVALUATION_DIR = old_dir
