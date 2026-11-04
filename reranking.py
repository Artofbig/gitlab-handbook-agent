"""Small local ONNX cross-encoder; consumes English research queries."""
import math
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class LocalReranker:
    def __init__(self, model=DEFAULT_RERANK_MODEL, cache=Path(".cache/models"),
                 threads=4, batch_size=8, encoder_factory=TextCrossEncoder):
        if threads < 1 or batch_size < 1:
            raise ValueError("Reranker threads and batch_size must be positive")
        self.model_name, self.cache = model, cache
        self.threads, self.batch_size = threads, batch_size
        self.encoder_factory, self._encoder = encoder_factory, None

    def __call__(self, queries, hits, top_k):
        if top_k < 1:
            raise ValueError("Reranker top_k must be positive")
        if not hits:
            return []
        # Handbook is English, so use Research's English query, not the Russian dense query.
        english = list(dict.fromkeys(query.sparse_query for query in queries))
        if not english:
            raise ValueError("Reranking requires at least one English query")
        if self._encoder is None:
            self._encoder = self.encoder_factory(
                model_name=self.model_name, cache_dir=str(self.cache), threads=self.threads,
                providers=["CPUExecutionProvider"])
        scores = [-math.inf] * len(hits)
        texts = [f"{hit['metadata'].get('title', '')}\n{hit['page_content']}" for hit in hits]
        for query in english:
            values = list(self._encoder.rerank(query, texts, batch_size=self.batch_size))
            if len(values) != len(hits) or any(not math.isfinite(float(v)) for v in values):
                raise ValueError("Reranker returned invalid scores")
            scores = [max(old, float(new)) for old, new in zip(scores, values)]
        # Stable ties retain the hybrid/round-robin order. Scores are not probabilities.
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [{**hits[i], "rerank_score": scores[i], "hybrid_rank": i + 1}
                for i in order[:top_k]]
