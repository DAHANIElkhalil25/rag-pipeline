# Retrieval Improvement Evidence

The final RAG system will compare the existing English-tagged MiniLM dense retriever with `intfloat/multilingual-e5-base` because final user questions are in French while the core documentation is largely English. The model card documents 100-language support and requires the `query: ` prefix for questions and the `passage: ` prefix for indexed passages in asymmetric retrieval.

The reranking experiment will retrieve a larger candidate set with hybrid search and then score query-passage pairs with `BAAI/bge-reranker-v2-m3`. Its model card identifies it as a multilingual reranker; it produces direct query-passage relevance scores and supports mixed-language retrieval. To fit the Kaggle T4 workflow, the reranker will be loaded only after candidate retrieval and released before local Mistral generation if GPU memory pressure occurs.

The controlled development experiments compare the current retrieval baseline with multilingual dense retrieval, and then the selected dense model with top-20 hybrid retrieval followed by reranking to top-5. The development set remains the 20 historical questions; the 120-question test set remains frozen until final evaluation.

## Sources

1. [Multilingual E5-base model card](https://huggingface.co/intfloat/multilingual-e5-base)
2. [BGE reranker v2 M3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
3. [Sentence Transformers cross-encoder documentation](https://sbert.net/docs/cross_encoder/pretrained_models.html)
4. [Scikit-learn Pipeline API reference](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
5. [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
6. [Python glossary](https://docs.python.org/3/glossary.html)
7. [LangChain retriever integrations](https://docs.langchain.com/oss/python/integrations/retrievers)
