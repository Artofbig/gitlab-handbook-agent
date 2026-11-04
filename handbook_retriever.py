"""Read-only bridge from the graph's Research Agent to existing hybrid Qdrant search."""
from pathlib import Path

from hybrid_search import Embedders, hybrid_search, read_state, resolve_collection


class HandbookRetriever:
    def __init__(self, client, alias="handbook", top_k=8, candidates=100,
                 cache=Path(".cache/models"), threads=4, embedder_factory=Embedders,
                 reranker=None, retrieval_k=30, rerank_limit=40):
        if not 0 < top_k <= retrieval_k <= candidates or threads < 1 or rerank_limit < top_k:
            raise ValueError("Require candidates >= retrieval_k >= top_k > 0, rerank_limit >= top_k and threads > 0")
        self.client, self.alias = client, alias
        self.top_k, self.candidates = top_k, candidates
        self.cache, self.threads = cache, threads
        self.embedder_factory = embedder_factory
        self.reranker, self.retrieval_k, self.rerank_limit = reranker, retrieval_k, rerank_limit
        self.last_debug = {}
        self._models = None
        self._config = None

    def __call__(self, queries):
        self.last_debug = {"stage": "index"}
        # Pin all queries of this research step to the same snapshot.
        collection = resolve_collection(self.client, self.alias)
        state = read_state(self.client, collection)
        if state["status"] != "ready":
            raise ValueError("Qdrant index is not ready")
        config = state["config"]
        model_config = (config["dense_model"], config["sparse_model"], config["dimension"], config.get("bm25"))
        if model_config != self._config:
            self.last_debug["stage"] = "embedding_load"
            options = {"bm25_config": config["bm25"]} if "bm25" in config else {}
            self._models = self.embedder_factory(config["dense_model"], config["sparse_model"],
                                                  self.cache, self.threads, **options)
            self._config = model_config
        # Round-robin across query results so the first query cannot consume the
        # whole context budget before another query's first result is considered.
        self.last_debug["stage"] = "hybrid_search"
        groups = [hybrid_search(self.client, collection, self._models,
                                query.dense_query, query.sparse_query,
                                self.retrieval_k, self.candidates) for query in queries]
        seen, hits = set(), []
        for index in range(max(map(len, groups), default=0)):
            for group in groups:
                if index < len(group) and group[index]["parent_id"] not in seen:
                    hit = group[index]
                    seen.add(hit["parent_id"])
                    hits.append(hit)
        pool = hits[:self.rerank_limit]
        self.last_debug.update(stage="rerank" if self.reranker else "selection", rerank_pool_size=len(pool))
        # Score every item in the bounded pool so debug data includes dropped candidates.
        ranked = self.reranker(queries, pool, len(pool)) if self.reranker and pool else pool
        selected = ranked[:self.top_k]
        self.last_debug = {
            "stage": "complete",
            "collection": collection, "candidates_per_branch": self.candidates,
            "parents_per_query": self.retrieval_k, "unique_parents": len(hits),
            "rerank_pool_size": len(pool), "returned_parents": len(selected),
            "reranker": getattr(self.reranker, "model_name", None),
            "query_language": "English" if self.reranker else None,
            "ranking": [{"parent_id": hit["parent_id"], "title": hit["metadata"].get("title"),
                         "rrf_score": hit["score"], "rerank_score": hit.get("rerank_score"),
                         "hybrid_rank": hit.get("hybrid_rank", index + 1), "rank": index + 1,
                         "selected": index < self.top_k} for index, hit in enumerate(ranked)],
        }
        return selected
