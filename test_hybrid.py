import json
from pathlib import Path
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from qdrant_client import QdrantClient, models

from hybrid_search import Embedders, hybrid_search, index_snapshot, read_state, resolve_collection


class EmbeddingMemoryTests(unittest.TestCase):
    def test_splade_batch_is_independent_of_upload_batch(self):
        encoder = Embedders.__new__(Embedders)
        encoder.sparse_batch_size = 1
        encoder.dense = SimpleNamespace(passage_embed=Mock(return_value=[np.array([1., 0.])] * 8))
        sparse = SimpleNamespace(indices=np.array([7]), values=np.array([1.]))
        encoder.sparse = SimpleNamespace(embed=Mock(return_value=[sparse] * 8))
        texts = ["text"] * 8
        result = encoder.encode_documents(texts)
        self.assertEqual(len(result), 8)
        encoder.sparse.embed.assert_called_once_with(texts, batch_size=1)

    def test_memory_failure_retries_in_order_without_losing_items(self):
        encoder = Embedders.__new__(Embedders)

        def encode(texts):
            if len(texts) > 2:
                raise MemoryError("Simulated allocation failure")
            return [(text, "vector") for text in texts]

        encoder._encode_documents_once = Mock(side_effect=encode)
        texts = list(range(8))
        with self.assertLogs("hybrid_search", level="WARNING"):
            result = encoder.encode_documents(texts)
        self.assertEqual(result, [(text, "vector") for text in texts])
        self.assertEqual(encoder._encode_documents_once.call_count, 7)

    def test_single_text_failure_stops_instead_of_looping(self):
        encoder = Embedders.__new__(Embedders)
        encoder._encode_documents_once = Mock(side_effect=MemoryError("allocation failed"))
        with self.assertRaisesRegex(MemoryError, "even for one text"):
            encoder.encode_documents(["text"])
        self.assertEqual(encoder._encode_documents_once.call_count, 1)


class TestEmbedders:
    dimension = 2

    def config(self):
        return {"dense_model": "test-dense", "sparse_model": "test-sparse", "dimension": 2}

    def encode_documents(self, texts):
        return [([1., 0.], models.SparseVector(indices=[3], values=[1.])) if text == "dense match"
                else ([0., 1.], models.SparseVector(indices=[7], values=[2.])) for text in texts]

    def encode_query(self, query, sparse_query=None):
        return [1., 0.], models.SparseVector(indices=[7], values=[1.])


class HybridTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).parent / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.client = QdrantClient(":memory:")
        self.embeddings = TestEmbedders()
        self.parents = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.children = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.write_snapshot()

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def write_snapshot(self, single=False):
        n = 1 if single else 2
        parents, children = [], []
        for i in range(n):
            meta = {"parent_id": self.parents[i], "title": f"Policy {i}",
                    "source_url": f"https://example.com/{i}"}
            parents.append({"id": self.parents[i], "page_content": f"Full policy {i}", "metadata": meta})
            children.append({"id": self.children[i], "page_content": "dense match" if i == 0 else "sparse match",
                             "metadata": meta})
        for name, rows in (("parents", parents), ("children", children)):
            (self.root / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_actual_qdrant_fusion_and_parent_lookup(self):
        # Two dense matches crowd the sparse-only policy out of dense top-2.
        # It must be recovered through SPLADE's prefetch and RRF.
        duplicate = json.loads((self.root / "children.jsonl").read_text(encoding="utf-8").splitlines()[0])
        duplicate["id"] = str(uuid.uuid4())
        with (self.root / "children.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(duplicate) + "\n")
        result = index_snapshot(self.client, self.root, self.embeddings)
        self.assertEqual(result["indexed_children"], 3)
        hits = hybrid_search(self.client, "handbook", self.embeddings, "query", top_k=2, candidates=2)
        self.assertEqual({row["parent_id"] for row in hits}, set(self.parents))
        self.assertEqual({row["page_content"] for row in hits}, {"Full policy 0", "Full policy 1"})
        self.assertTrue(all(row["metadata"]["source_url"] for row in hits))
        config = self.client.get_collection(result["collection"]).config.params
        self.assertEqual(config.vectors["dense"].size, 2)
        self.assertIn("splade", config.sparse_vectors)
        # Removing the source files must not break parent retrieval.
        (self.root / "parents.jsonl").unlink()
        self.assertEqual(len(hybrid_search(self.client, "handbook", self.embeddings, "query", top_k=2)), 2)

    def test_rebuild_does_not_retain_deleted_children(self):
        first = index_snapshot(self.client, self.root, self.embeddings)
        self.write_snapshot(single=True)
        second = index_snapshot(self.client, self.root, self.embeddings)
        self.assertNotEqual(first["collection"], second["collection"])
        self.assertEqual(resolve_collection(self.client, "handbook"), second["collection"])
        hits = hybrid_search(self.client, "handbook", self.embeddings, "query", top_k=2)
        self.assertEqual([row["parent_id"] for row in hits], self.parents[:1])

    def test_interrupted_index_resumes_without_publishing_partial_data(self):
        original = index_snapshot(self.client, self.root, self.embeddings)
        seen = []

        class FailSecondBatch(TestEmbedders):
            def encode_documents(self, texts):
                seen.append(texts)
                if len(seen) == 2:
                    raise RuntimeError("Simulated model interruption")
                return super().encode_documents(texts)

        before = {row.name for row in self.client.get_collections().collections}
        with self.assertRaises(RuntimeError):
            index_snapshot(self.client, self.root, FailSecondBatch(), batch_size=1)
        after = {row.name for row in self.client.get_collections().collections}
        staging = (after - before).pop()
        self.assertEqual(read_state(self.client, staging)["indexed_children"], 1)
        self.assertEqual(resolve_collection(self.client, "handbook"), original["collection"])
        result = index_snapshot(self.client, self.root, self.embeddings, batch_size=1, resume=staging)
        self.assertEqual(result["indexed_children"], 2)
        self.assertEqual(resolve_collection(self.client, "handbook"), staging)
        self.write_snapshot(single=True)
        with self.assertRaises(ValueError):
            index_snapshot(self.client, self.root, self.embeddings, resume=staging)

    def test_orphan_parent_rejected(self):
        (self.root / "parents.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing parent"):
            index_snapshot(self.client, self.root, self.embeddings)
        self.assertEqual(len(self.client.get_collections().collections), 0)


if __name__ == "__main__":
    unittest.main()
