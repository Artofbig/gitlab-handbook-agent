from contextlib import closing
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

from qdrant_client import QdrantClient

from hybrid_search import DENSE_MODEL, Embedders, SPARSE_MODEL, digest, hybrid_search, read_state
from test_hybrid import TestEmbedders
from vector_transfer import (atomic_json, export_vectors, import_vectors, pack_input,
                             pack_result, validate_bundle)


class PortableEmbedders(TestEmbedders):
    def config(self):
        return {"dense_model": "test-dense", "sparse_model": SPARSE_MODEL, "dimension": 2}


class TransferTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).parent / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.snapshot, self.output = self.root / "snapshot", self.root / "vectors"
        self.snapshot.mkdir()
        self.parent = str(uuid.uuid4())
        metadata = {"parent_id": self.parent, "source_url": "https://example.com/policy"}
        parents = [{"id": self.parent, "page_content": "Full policy", "metadata": metadata}]
        children = [{"id": str(uuid.uuid4()), "page_content": text, "metadata": metadata}
                    for text in ("dense match", "sparse match", "dense match")]
        for name, rows in (("parents", parents), ("children", children)):
            (self.snapshot / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def export(self, **kwargs):
        return export_vectors(self.snapshot, self.output, PortableEmbedders(),
                              progress=None, **kwargs)

    def test_export_import_and_retrieval_without_model_inference(self):
        manifest = self.export(shard_size=2)
        self.assertEqual(manifest["indexed_children"], 3)
        with closing(QdrantClient(":memory:")) as client, patch.object(
                Embedders, "encode_documents", side_effect=AssertionError("Inference during import")):
            result = import_vectors(client, self.output, progress=None)
            self.assertEqual(result["indexed_children"], 3)
            hits = hybrid_search(client, "handbook", PortableEmbedders(), "question", top_k=1)
            self.assertEqual(hits[0]["parent_id"], self.parent)
            self.assertEqual(hits[0]["page_content"], "Full policy")
            # Vectors stored in Qdrant are the exported values, not recomputed vectors.
            point_id = json.loads((self.snapshot / "children.jsonl").read_text().splitlines()[0])["id"]
            point = client.retrieve("handbook", [point_id], with_vectors=True)[0]
            self.assertEqual(point.vector["dense"], [1., 0.])

    def test_export_resume_skips_committed_shards(self):
        calls = []

        class Interrupt(PortableEmbedders):
            def encode_documents(self, texts):
                calls.append(texts)
                if len(calls) == 2:
                    raise RuntimeError("Interrupted GPU session")
                return super().encode_documents(texts)

        with self.assertRaises(RuntimeError):
            export_vectors(self.snapshot, self.output, Interrupt(), batch_size=1,
                           shard_size=1, progress=None)
        manifest = validate_bundle(self.output, complete=False)
        self.assertEqual(manifest["indexed_children"], 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_bundle(self.output)
        manifest = self.export(resume=True, shard_size=1)
        self.assertEqual(manifest["indexed_children"], 3)
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(validate_bundle(self.output)["indexed_children"], 3)

    def test_resume_rejects_changed_source(self):
        self.export()
        with (self.snapshot / "children.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "same corpus"):
            self.export(resume=True)

    def test_missing_dense_export_fails_before_preprocessing_or_output_creation(self):
        class NoWork(PortableEmbedders):
            def fit_bm25(self, texts):
                raise AssertionError("Preprocessing must not start for a missing reuse path")

        with self.assertRaisesRegex(ValueError, "No saved export manifest"):
            export_vectors(self.snapshot, self.output, NoWork(), reuse_dense=self.root / "missing")
        self.assertFalse(self.output.exists())

    def test_corruption_rejected_before_creating_collection(self):
        manifest = self.export()
        (self.output / manifest["shards"][0]["file"]).write_bytes(b"broken")
        with closing(QdrantClient(":memory:")) as client:
            with self.assertRaisesRegex(ValueError, "checksum"):
                import_vectors(client, self.output, progress=None)
            self.assertEqual(client.get_collections().collections, [])

    def test_import_can_resume_after_qdrant_failure(self):
        self.export()
        with closing(QdrantClient(":memory:")) as client:
            original_upsert = client.upsert
            seen = []

            def failing(collection, points, **kwargs):
                if points[0].payload.get("kind") == "child":
                    seen.append(points)
                    if len(seen) == 2:
                        raise RuntimeError("Connection lost")
                return original_upsert(collection, points, **kwargs)

            with patch.object(client, "upsert", side_effect=failing), self.assertRaises(RuntimeError):
                import_vectors(client, self.output, batch_size=1, progress=None)
            collection = client.get_collections().collections[0].name
            self.assertEqual(read_state(client, collection)["indexed_children"], 1)
            result = import_vectors(client, self.output, resume=collection, progress=None)
            self.assertEqual(result["indexed_children"], 3)

    def test_archives_are_self_contained_and_sample_is_separate(self):
        source_zip = self.root / "input.zip"
        pack_input(self.snapshot, source_zip)
        with zipfile.ZipFile(source_zip) as archive:
            self.assertEqual(set(archive.namelist()), {"snapshot/parents.jsonl", "snapshot/children.jsonl",
                             "hybrid_search.py", "vector_transfer.py", "bm25_text.py", "requirements-gpu.txt"})
        self.export(limit=2)
        result_zip = self.root / "result.zip"
        pack_result(self.output, result_zip)
        extracted = self.root / "extracted"
        with zipfile.ZipFile(result_zip) as archive:
            archive.extractall(extracted)
        self.assertEqual(validate_bundle(extracted)["indexed_children"], 2)
        with closing(QdrantClient(":memory:")) as client:
            result = import_vectors(client, extracted, progress=None)
            self.assertEqual(result["alias"], "handbook_sample")

    def test_missing_cuda_fails_before_downloading_models(self):
        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]), \
                patch("onnxruntime.preload_dlls", create=True), \
                patch("fastembed.TextEmbedding") as dense:
            dense.list_supported_models.return_value = [{"model": DENSE_MODEL, "dim": 384}]
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                Embedders(cache=self.root / "models", device="cuda")
            dense.assert_not_called()

    def test_checkpoint_retries_transient_file_lock(self):
        path = self.root / "manifest.json"
        atomic_json(path, {"count": 1})
        import os
        replace = os.replace
        calls = []

        def temporarily_locked(source, destination):
            calls.append(destination)
            if len(calls) < 3:
                raise PermissionError("Simulated temporary file lock")
            replace(source, destination)

        with patch("vector_transfer.os.replace", side_effect=temporarily_locked), patch("vector_transfer.time.sleep"):
            atomic_json(path, {"count": 2})
        self.assertEqual(json.loads(path.read_text()), {"count": 2})
        self.assertEqual(len(calls), 3)

    def test_bad_dimension_is_rejected_even_with_matching_checksums(self):
        manifest = self.export()
        shard = manifest["shards"][0]
        path = self.output / shard["file"]
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        rows[0]["dense"] = [1.]
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        shard["sha256"] = digest(path)
        atomic_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_bundle(self.output)

    def test_cuda_provider_is_passed_to_both_models_and_fallback_rejected(self):
        for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            with self.subTest(provider=provider), \
                    patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), \
                    patch("onnxruntime.preload_dlls", create=True), \
                    patch("fastembed.TextEmbedding") as dense, \
                    patch("fastembed.SparseTextEmbedding") as sparse:
                dense.list_supported_models.return_value = [{"model": DENSE_MODEL, "dim": 384}]
                dense.return_value.model.model.get_providers.return_value = [provider]
                sparse.return_value.model.model.get_providers.return_value = ["CUDAExecutionProvider"]
                if provider == "CPUExecutionProvider":
                    with self.assertRaisesRegex(RuntimeError, "fell back to CPU"):
                        Embedders(cache=self.root / "models", device="cuda")
                else:
                    Embedders(cache=self.root / "models", device="cuda")
                self.assertEqual(dense.call_args.kwargs["providers"], ["CUDAExecutionProvider"])
                # BM25 is lexical CPU work and must never request CUDA/SPLADE.
                self.assertEqual(sparse.call_args.kwargs["model_name"], "Qdrant/bm25")
                self.assertNotIn("providers", sparse.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
