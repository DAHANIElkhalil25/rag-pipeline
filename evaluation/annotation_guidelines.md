# Final Evaluation Annotation Guide

The final test set contains 120 planned records: 40 Python, 40 Scikit-learn, and 40 LangChain questions. Each domain contains 8 factual, 8 conceptual, 8 procedural, 8 troubleshooting, and 8 multi-step questions. The intended difficulty balance per domain is 12 easy, 20 medium, and 8 hard items.

Each item must be written from the official documentation version recorded in `source_versions`. The reference answer should be concise, complete for the question, and traceable to the cited official URLs. It should not mention details that are absent from the cited material.

For every reference answer, annotate the smallest set of final-index chunk IDs that jointly supports the answer. Record those IDs in `reference_context_ids`; record the associated documentation pages in `reference_source_urls`. If the required information is absent from the index, flag the record as `corpus_gap` rather than guessing a chunk ID.

The annotator completes the item. A reviewer verifies the question is unambiguous, the reference is correct, and all context IDs support it. Only an item whose `annotation_status` is `validated` may enter `test_dataset_v1.jsonl`. The test set must be frozen before final retrieval or prompt tuning begins.
