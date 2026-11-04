import copy
import math
import unittest
from unittest.mock import Mock, patch

from handbook_graph import SearchQuery
from handbook_retriever import HandbookRetriever
from reranking import LocalReranker


def hit(parent, score=0.1):
    return {"parent_id": parent, "page_content": "Original **policy** " + parent,
            "metadata": {"title": "Title " + parent}, "score": score, "matched_children": [parent + "-child"]}


class RerankingTests(unittest.TestCase):
    def test_english_queries_lazy_model_loading_and_original_payloads(self):
        encoder = Mock()
        encoder.rerank.side_effect = [[-3., 2., 2.], [1., 0., 0.], [-3., 2., 2.], [1., 0., 0.]]
        factory = Mock(return_value=encoder)
        rerank = LocalReranker(encoder_factory=factory, batch_size=2)
        queries = [SearchQuery(dense_query="Отпуск?", sparse_query="vacation"),
                   SearchQuery(dense_query="Уведомление?", sparse_query="notice")]
        hits = [hit("a"), hit("b"), hit("c")]
        original = copy.deepcopy(hits)
        factory.assert_not_called()
        result = rerank(queries, hits, 2)
        self.assertEqual([item["parent_id"] for item in result], ["b", "c"])
        self.assertEqual([item["hybrid_rank"] for item in result], [2, 3])
        self.assertEqual(result[0]["page_content"], original[1]["page_content"])
        self.assertEqual(hits, original)
        self.assertEqual([call.args[0] for call in encoder.rerank.call_args_list], ["vacation", "notice"])
        rerank(queries, hits, 2)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_invalid_scores_fail_and_empty_pool_does_not_load_model(self):
        factory = Mock()
        rerank = LocalReranker(encoder_factory=factory)
        query = [SearchQuery(dense_query="отпуск", sparse_query="vacation")]
        self.assertEqual(rerank(query, [], 2), [])
        factory.assert_not_called()
        for values in ([math.nan], [], [math.inf]):
            factory.return_value.rerank.return_value = values
            with self.assertRaisesRegex(ValueError, "invalid scores"):
                rerank(query, [hit("a")], 1)

    @patch("handbook_retriever.hybrid_search")
    @patch("handbook_retriever.read_state")
    @patch("handbook_retriever.resolve_collection", return_value="pinned-snapshot")
    def test_expand_deduplicate_rerank_bound_pool_and_capture_scores(self, resolve, state, search):
        state.return_value = {"status": "ready", "config": {"dense_model": "dense", "sparse_model": "sparse", "dimension": 2}}
        search.side_effect = [[hit("a"), hit("b")], [hit("a"), hit("c")]]
        encoder = Mock()
        encoder.rerank.return_value = [-3., 2., 5.]
        reranker = LocalReranker(encoder_factory=Mock(return_value=encoder))
        embedder = Mock()
        retriever = HandbookRetriever(Mock(), top_k=2, candidates=100, retrieval_k=30,
                                      rerank_limit=3, embedder_factory=embedder, reranker=reranker)
        queries = [SearchQuery(dense_query="отпуск", sparse_query="vacation")] * 2
        results = retriever(queries)
        self.assertEqual([item["parent_id"] for item in results], ["c", "b"])
        self.assertTrue(all(call.args[1] == "pinned-snapshot" and call.args[-2:] == (30, 100)
                            for call in search.call_args_list))
        self.assertEqual(retriever.last_debug["unique_parents"], 3)
        self.assertEqual([item["selected"] for item in retriever.last_debug["ranking"]], [True, True, False])
        self.assertEqual(retriever.last_debug["ranking"][0]["rerank_score"], 5.)
        self.assertEqual(retriever.last_debug["reranker"], reranker.model_name)
        encoder.rerank.assert_called_once()  # Duplicate English queries do not rescore.

    def test_invalid_pool_settings(self):
        with self.assertRaises(ValueError):
            HandbookRetriever(Mock(), top_k=8, retrieval_k=7)
        with self.assertRaises(ValueError):
            HandbookRetriever(Mock(), top_k=8, rerank_limit=7)


if __name__ == "__main__":
    unittest.main()
