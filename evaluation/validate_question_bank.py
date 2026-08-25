import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_schema import read_jsonl


PATH = Path(__file__).resolve().parent / "datasets" / "test_dataset_v1_source_grounded_draft.jsonl"
records = read_jsonl(PATH)
if len(records) != 120:
    raise SystemExit(f"Expected 120 records, found {len(records)}")

domains = collections.Counter(record["domain"] for record in records)
if domains != {"python": 40, "scikit_learn": 40, "langchain": 40}:
    raise SystemExit(f"Unexpected domain balance: {dict(domains)}")

types = collections.Counter(record["question_type"] for record in records)
if types != {"factual": 24, "conceptual": 24, "procedural": 24, "troubleshooting": 24, "multi_step": 24}:
    raise SystemExit(f"Unexpected question-type balance: {dict(types)}")

for record in records:
    if not record["reference"].strip() or not record["reference_source_urls"]:
        raise SystemExit(f"Missing reference evidence for {record['question_id']}")
    if not all(url.startswith("https://") for url in record["reference_source_urls"]):
        raise SystemExit(f"Non-HTTPS source URL for {record['question_id']}")
    if record["annotation"]["review_status"] != "source_grounded_draft":
        raise SystemExit(f"Unexpected draft status for {record['question_id']}")

print(f"Valid source-grounded draft bank: {len(records)} questions; domains={dict(domains)}; types={dict(types)}")
