# Official Ragas Compatibility Notes

The final evaluator is pinned to Ragas 0.4.3. The Ragas v0.4 migration guide documents the collections-based metric API, where metrics accept named fields with `ascore(...)` and return structured `MetricResult` objects. This project uses the verified signatures for Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, and FactualCorrectness.

For the evaluator LLM, Ragas documents `llm_factory()` adapters for supported providers including OpenAI and Mistral. The final Kaggle workflow therefore keeps the local Mistral generator separate and reads an evaluator-provider key only from a Kaggle Secret. The evaluation runner persists the judge provider/model in every run manifest.

The Ragas embeddings reference documents `HuggingfaceEmbeddings`, used for AnswerRelevancy with the final pipeline's embedding-model identifier.

## Sources

1. [Ragas v0.3 to v0.4 migration](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/)
2. [Ragas LLM adapters](https://docs.ragas.io/en/stable/howtos/llm-adapters/)
3. [Ragas LangChain evaluation integration](https://docs.ragas.io/en/stable/howtos/integrations/langchain/)
4. [Ragas embeddings reference](https://docs.ragas.io/en/stable/references/embeddings/)
5. [Ragas 0.4.3 release on PyPI](https://pypi.org/project/ragas/)
