"""Initialise les fichiers de données d'évaluation sans modifier la baseline."""

import csv
import json
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATASETS_DIR = ROOT / "datasets"
DEV_PATH = DATASETS_DIR / "dev_dataset_v1.jsonl"
TEST_TEMPLATE_PATH = DATASETS_DIR / "test_dataset_v1_annotation_template.csv"


def load_baseline_qa() -> list[dict]:
    """Read the existing constant without importing the ML-dependent evaluator module."""
    tree = ast.parse((PROJECT_ROOT / "etape6_evaluation.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "GROUND_TRUTH_QA":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GROUND_TRUTH_QA":
                    return ast.literal_eval(node.value)
    raise ValueError("GROUND_TRUTH_QA introuvable dans etape6_evaluation.py")


def create_development_set() -> None:
    """Convertit les 20 paires baseline en jeu de développement, jamais en test final."""
    records = []
    for index, item in enumerate(load_baseline_qa(), start=1):
        domain = str(item.get("source_filter", "unknown")).lower().replace("-", "_")
        records.append({
            "question_id": f"dev_{domain}_{index:03d}",
            "split": "dev",
            "domain": domain,
            "question_type": "conceptual_or_factual",
            "difficulty": "medium",
            "language": "fr",
            "user_input": item["question"],
            "reference": item["reference_answer"],
            "reference_context_ids": [],
            "reference_source_urls": [],
            "source_versions": {},
            "annotation": {
                "review_status": "needs_context_annotation",
                "origin": "baseline_ground_truth_qa",
                "notes": "Development only: do not use this record in final reporting."
            }
        })
    DEV_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def create_final_annotation_template() -> None:
    """Crée 120 emplacements équilibrés à annoter manuellement, sans fabriquer de vérité terrain."""
    domains = ["python", "scikit_learn", "langchain"]
    question_types = ["factual", "conceptual", "procedural", "troubleshooting", "multi_step"]
    difficulties = ["easy"] * 12 + ["medium"] * 20 + ["hard"] * 8
    fields = [
        "question_id", "split", "domain", "question_type", "difficulty", "language",
        "user_input", "reference", "reference_context_ids", "reference_source_urls",
        "source_versions", "annotation_status", "annotator", "reviewer", "notes",
    ]
    rows = []
    for domain in domains:
        for index, difficulty in enumerate(difficulties, start=1):
            question_type = question_types[(index - 1) // 8]
            rows.append({
                "question_id": f"test_{domain}_{index:03d}",
                "split": "test",
                "domain": domain,
                "question_type": question_type,
                "difficulty": difficulty,
                "language": "fr",
                "user_input": "",
                "reference": "",
                "reference_context_ids": "[]",
                "reference_source_urls": "[]",
                "source_versions": "{}",
                "annotation_status": "draft",
                "annotator": "",
                "reviewer": "",
                "notes": "Complete manually from official documentation before freezing the test set.",
            })
    with TEST_TEMPLATE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    create_development_set()
    create_final_annotation_template()
    print(f"Created {DEV_PATH}")
    print(f"Created {TEST_TEMPLATE_PATH}")
