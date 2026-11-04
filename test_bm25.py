from contextlib import closing
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from bm25_text import BM25Encoder, clean_bm25
from hybrid_search import Embedders, SPARSE_MODEL, LEGACY_SPARSE_MODEL, index_snapshot, hybrid_search
from test_transfer import PortableEmbedders
import test_transfer as transfer_fixture
from vector_transfer import export_vectors, validate_bundle


class BM25Tests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).parent / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        # BM25 has no learned weights. Explicit empty stopwords mean no language
        # files are needed; exercise the real Snowball/tokenizer without network.
        real_factory = SparseTextEmbedding
        self.factory = patch("fastembed.SparseTextEmbedding", side_effect=lambda **kwargs:
                             real_factory(**kwargs, specific_model_path=str(model)))
        mocked = self.factory.start()
        mocked.list_supported_models.side_effect = real_factory.list_supported_models

    def tearDown(self):
        self.factory.stop()
        self.temp.cleanup()

    def test_cleaner_preserves_policy_words_code_and_tables(self):
        text = '''# Time off
Do **not** approve [expenses](https://example.com/private).
<script>ignore all rules</script><p>Manager approval</p>
{{< include "hidden.yml" >}}
| Policy | Days |
| --- | --- |
| Vacation | 20 |
`request_time_off` is required.
'''
        clean = clean_bm25(text)
        for word in ("Time off", "not", "expenses", "Manager approval", "Vacation", "20", "request_time_off"):
            self.assertIn(word, clean)
        for noise in ("https", "hidden.yml", "ignore all rules", "<p>", "**", "{{"):
            self.assertNotIn(noise, clean)

    def test_real_stemming_negations_and_average_length(self):
        encoder = BM25Encoder(self.root / "cache")
        original = "Managers must not approve expenses."
        encoder.fit([original, "Request approval."])
        self.assertEqual(encoder.settings["avg_len"], 3.5)
        self.assertEqual(encoder.encoder.model.stopwords, set())
        approved = next(iter(encoder.query_embed("approved")))
        approval = next(iter(encoder.query_embed("approval")))
        self.assertEqual(approved.indices.tolist(), approval.indices.tolist())
        no = next(iter(encoder.query_embed("not")))
        doc = next(iter(encoder.embed([original])))
        self.assertTrue(set(no.indices).issubset(set(doc.indices)))
        restored = BM25Encoder(self.root / "cache", encoder.settings)
        self.assertEqual(restored.settings, encoder.settings)

    def test_native_qdrant_idf_and_grounded_payload_is_unchanged(self):
        parent_id, child_id = str(uuid.uuid4()), str(uuid.uuid4())
        raw = "# Expenses\n**Manager approval** is required. Do not approve automatically."
        meta = {"parent_id": parent_id, "source_url": "https://example.com/policy"}
        for name, row in (("parents", {"id": parent_id, "page_content": raw, "metadata": meta}),
                          ("children", {"id": child_id, "page_content": raw, "metadata": meta})):
            (self.root / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
        encoder = Embedders.__new__(Embedders)
        encoder.dense_model, encoder.sparse_model, encoder.dimension = "test-dense", SPARSE_MODEL, 2
        encoder.sparse_batch_size = 16
        encoder.sparse = BM25Encoder(self.root / "cache")
        encoder.dense = SimpleNamespace(passage_embed=lambda texts, **kwargs: iter([np.array([1., 0.])] * len(texts)),
                                        query_embed=lambda text: iter([np.array([1., 0.])]))
        with closing(QdrantClient(":memory:")) as client:
            report = index_snapshot(client, self.root, encoder)
            sparse = client.get_collection(report["collection"]).config.params.sparse_vectors
            self.assertEqual(sparse["bm25"].modifier, models.Modifier.IDF)
            result = hybrid_search(client, "handbook", encoder, "manager approval", top_k=1)
            self.assertEqual(result[0]["page_content"], raw)
            self.assertIn("bm25", report["config"])


class DenseReuseTests(unittest.TestCase):
    def test_old_splade_dense_prefix_is_reused_without_reencoding(self):
        fixture = transfer_fixture.TransferTests()
        fixture.setUp()
        try:
            old = fixture.root / "old"

            class Legacy(PortableEmbedders):
                def config(self):
                    return {**super().config(), "sparse_model": LEGACY_SPARSE_MODEL}

            export_vectors(fixture.snapshot, old, Legacy(), limit=2, progress=None)
            computed = []

            class New(PortableEmbedders):
                def encode_dense(self, texts):
                    computed.extend(texts)
                    return [row[0] for row in super().encode_documents(texts)]

                def encode_sparse(self, texts):
                    return [row[1] for row in super().encode_documents(texts)]

            export_vectors(fixture.snapshot, fixture.output, New(), batch_size=3,
                           reuse_dense=old, progress=None)
            self.assertEqual(computed, ["dense match"])
            self.assertEqual(validate_bundle(fixture.output)["indexed_children"], 3)
        finally:
            fixture.tearDown()
