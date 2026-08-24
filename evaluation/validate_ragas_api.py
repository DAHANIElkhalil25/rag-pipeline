import inspect

import ragas
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)


metric_types = [Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness]
print(f"ragas={getattr(ragas, '__version__', 'unknown')}")
print(f"HuggingfaceEmbeddings={HuggingfaceEmbeddings.__name__}")
for metric_type in metric_types:
    print(f"{metric_type.__name__}.ascore={inspect.signature(metric_type.ascore)}")
