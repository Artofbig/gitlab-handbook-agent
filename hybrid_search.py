"""Persistent parent/child retrieval: dense + native Qdrant BM25/IDF + RRF."""
from __future__ import annotations

import argparse
import gc
import hashlib
from itertools import islice
import json
import logging
import os
from pathlib import Path
import sys
import uuid

from qdrant_client import QdrantClient, models

DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LEGACY_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
SPARSE_MODEL = "Qdrant/bm25"
STATE_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "gitlab-handbook:hybrid-index-state:v1"))


class Embedders:
    def __init__(self, dense_model=DENSE_MODEL, sparse_model=SPARSE_MODEL,
                 cache=Path(".cache/models"), threads=4, sparse_batch_size=128, device="cpu", bm25_config=None):
        from fastembed import SparseTextEmbedding, TextEmbedding

        if sparse_batch_size < 1:
            raise ValueError("sparse-batch-size must be positive")
        self.sparse_batch_size = sparse_batch_size
        self.dense_model, self.sparse_model = dense_model, sparse_model
        descriptions = {row["model"]: row for row in TextEmbedding.list_supported_models()}
        if dense_model not in descriptions:
            raise ValueError(f"Unsupported FastEmbed dense model: {dense_model}")
        if sparse_model not in {SPARSE_MODEL, LEGACY_SPARSE_MODEL}:
            raise ValueError(f"Unsupported sparse model: {sparse_model}")
        self.dimension = descriptions[dense_model]["dim"]
        cache.mkdir(parents=True, exist_ok=True)
        options = dict(cache_dir=str(cache.resolve()), threads=threads)
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device == "cuda":
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA is unavailable. Install fastembed-gpu and a compatible onnxruntime-gpu.")
        options["providers"] = ["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"]
        self.dense = TextEmbedding(model_name=dense_model, **options)
        if sparse_model == SPARSE_MODEL:
            from bm25_text import BM25Encoder
            self.sparse = BM25Encoder(cache.resolve(), bm25_config)
        else:
            self.sparse = SparseTextEmbedding(model_name=sparse_model, **options)
        if device == "cuda":
            encoders = [("dense", self.dense)]
            if sparse_model == LEGACY_SPARSE_MODEL:
                encoders.append(("splade", self.sparse))
            for label, encoder in encoders:
                if "CUDAExecutionProvider" not in encoder.model.model.get_providers():
                    raise RuntimeError(f"{label} fell back to CPU; fix CUDA/cuDNN before exporting")

    @staticmethod
    def sparse_vector(vector):
        return models.SparseVector(indices=vector.indices.tolist(), values=vector.values.tolist())

    def encode_documents(self, texts):
        if not texts:
            return []
        try:
            return self._encode_documents_once(texts)
        except MemoryError as exc:
            if len(texts) == 1:
                raise MemoryError("Insufficient RAM even for one text. Free memory, then resume indexing.") from exc
        # Leave the exception handler before retrying: its traceback can retain
        # the failed inference arrays. Never checkpoint a partially encoded batch.
        gc.collect()
        logging.getLogger(__name__).warning("Embedding MemoryError: reducing batch from %s texts", len(texts))
        middle = len(texts) // 2
        return self.encode_documents(texts[:middle]) + self.encode_documents(texts[middle:])

    def _encode_documents_once(self, texts):
        dense = list(self.dense.passage_embed(texts, batch_size=len(texts)))
        # SPLADE creates [batch, tokens, vocabulary] intermediates. Keep its
        # inference batch independent of the batch uploaded to Qdrant.
        sparse = list(self.sparse.embed(texts, batch_size=self.sparse_batch_size))
        return [(d.tolist(), self.sparse_vector(s)) for d, s in zip(dense, sparse, strict=True)]

    def fit_bm25(self, texts):
        if self.sparse_model == SPARSE_MODEL:
            self.sparse.fit(texts)

    def encode_dense(self, texts):
        return [row.tolist() for row in self.dense.passage_embed(texts, batch_size=len(texts))] if texts else []

    def encode_sparse(self, texts):
        return [self.sparse_vector(row) for row in self.sparse.embed(texts, batch_size=self.sparse_batch_size)]

    def encode_query(self, query, sparse_query=None):
        dense = next(iter(self.dense.query_embed(query))).tolist()
        sparse = next(iter(self.sparse.query_embed(sparse_query or query)))
        return dense, self.sparse_vector(sparse)

    def config(self):
        config = {"dense_model": self.dense_model, "sparse_model": self.sparse_model,
                  "dimension": self.dimension}
        if self.sparse_model == SPARSE_MODEL:
            config["bm25"] = self.sparse.settings
        return config


def sparse_name(config):
    return "bm25" if config["sparse_model"] == SPARSE_MODEL else "splade"


def records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def batches(items, size):
    items = iter(items)
    while batch := list(islice(items, size)):
        yield batch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_state(client, collection):
    rows = client.retrieve(collection, [STATE_ID], with_payload=True, with_vectors=False)
    if not rows or (rows[0].payload or {}).get("kind") != "index_state":
        raise ValueError(f"No Handbook index configuration in {collection}")
    return rows[0].payload["state"]


def save_state(client, collection, state):
    client.upsert(collection, [models.PointStruct(
        id=STATE_ID, vector={}, payload={"kind": "index_state", "state": state}
    )], wait=True)


def child_filter():
    return models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="child"))])


def publish(client, collection, alias):
    existing = {item.alias_name for item in client.get_aliases().aliases}
    changes = []
    if alias in existing:
        changes.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    changes.append(models.CreateAliasOperation(create_alias=models.CreateAlias(
        collection_name=collection, alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=changes)


def index_snapshot(client, snapshot: Path, embedders, alias="handbook", batch_size=8,
                   limit=None, resume=None, progress=None):
    """Build a fresh collection, checkpoint acknowledged batches, publish atomically."""
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch-size and limit must be positive")
    snapshot = snapshot.resolve()
    children_path, parents_path = snapshot / "children.jsonl", snapshot / "parents.jsonl"
    if hasattr(embedders, "fit_bm25"):
        embedders.fit_bm25(row["page_content"] for row in islice(records(children_path), limit))
    config = {**embedders.config(), "schema": 1, "limit": limit,
              "children_sha256": digest(children_path), "parents_sha256": digest(parents_path)}
    # Validate relationships before expensive inference or collection publication.
    parent_ids = {row["id"] for row in records(parents_path)}
    needed_parents = set()
    expected = 0
    for row in islice(records(children_path), limit):
        parent_id = row["metadata"]["parent_id"]
        if parent_id not in parent_ids:
            raise ValueError(f"Missing parent {parent_id} for child {row['id']}")
        needed_parents.add(parent_id)
        expected += 1
    if not expected:
        raise ValueError("No children to index")
    if resume:
        collection = resume
        state = read_state(client, collection)
        if state["config"] != config:
            raise ValueError("Resume requires the same snapshot, models and limit")
        if alias == collection:
            raise ValueError("Alias must differ from the physical collection")
    else:
        collection = "handbook_" + uuid.uuid4().hex
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": models.VectorParams(size=embedders.dimension,
                                                          distance=models.Distance.COSINE)},
            sparse_vectors_config={sparse_name(config): models.SparseVectorParams(
                modifier=models.Modifier.IDF if config["sparse_model"] == SPARSE_MODEL else None)},
        )
        client.create_payload_index(collection, field_name="kind", field_schema=models.PayloadSchemaType.KEYWORD)
        state = {"config": config, "status": "building", "indexed_children": 0,
                 "parents_loaded": False, "expected_children": expected,
                 "expected_parents": len(needed_parents)}
        save_state(client, collection, state)
    if progress:
        progress(f"Collection: {collection}; resume with --resume {collection}")
    if state["status"] == "ready":
        publish(client, collection, alias)
        return {"collection": collection, "alias": alias, **state}
    if not state["parents_loaded"]:
        parents = (row for row in records(parents_path) if row["id"] in needed_parents)
        for batch in batches(parents, 128):
            client.upsert(collection, [models.PointStruct(
                id=row["id"], vector={}, payload={"kind": "parent", **row}
            ) for row in batch], wait=True)
        state["parents_loaded"] = True
        save_state(client, collection, state)
    remaining = islice(records(children_path), state["indexed_children"], limit)
    for batch in batches(remaining, batch_size):
        vectors = embedders.encode_documents([row["page_content"] for row in batch])
        points = [models.PointStruct(
            id=row["id"], vector={"dense": dense, sparse_name(config): sparse},
            payload={"kind": "child", **row},
        ) for row, (dense, sparse) in zip(batch, vectors, strict=True)]
        client.upsert(collection, points, wait=True)
        state["indexed_children"] += len(batch)
        save_state(client, collection, state)
        if progress:
            progress(f"Indexed {state['indexed_children']}/{expected} children")
    actual = client.count(collection, count_filter=child_filter(), exact=True).count
    if actual != expected:
        raise ValueError(f"Incomplete index: expected {expected} children, found {actual}")
    state["status"] = "ready"
    save_state(client, collection, state)
    publish(client, collection, alias)
    return {"collection": collection, "alias": alias, **state}


def resolve_collection(client, name):
    # Pin a physical collection for the whole query, even if its alias moves.
    for item in client.get_aliases().aliases:
        if item.alias_name == name:
            return item.collection_name
    return name


def hybrid_search(client, collection, embedders, query, sparse_query=None, top_k=5, candidates=40):
    if not query.strip() or top_k < 1 or candidates < top_k:
        raise ValueError("Require a nonempty query and candidates >= top-k > 0")
    if sparse_query is not None and not sparse_query.strip():
        raise ValueError("sparse-query cannot be empty")
    collection = resolve_collection(client, collection)
    state = read_state(client, collection)
    if state["status"] != "ready":
        raise ValueError("Index is not ready")
    if any(state["config"][key] != value for key, value in embedders.config().items()):
        raise ValueError("Query models must match the indexed models")
    dense, sparse = embedders.encode_query(query, sparse_query)
    prefetch = [models.Prefetch(query=dense, using="dense", limit=candidates, filter=child_filter())]
    if sparse.indices:
        prefetch.append(models.Prefetch(query=sparse, using=sparse_name(state["config"]), limit=candidates,
                                       filter=child_filter()))
    hits = client.query_points(
        collection_name=collection, prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=candidates,
        with_payload=True, with_vectors=False,
    ).points
    parent_ids = list(dict.fromkeys(hit.payload["metadata"]["parent_id"] for hit in hits))[:top_k]
    if not parent_ids:
        return []
    parents = {str(row.id): row.payload for row in client.retrieve(
        collection, parent_ids, with_payload=True, with_vectors=False)}
    results = []
    for parent_id in parent_ids:
        parent = parents.get(parent_id)
        if parent is None or parent.get("kind") != "parent":
            raise ValueError(f"Index integrity error: parent {parent_id} missing")
        matched = [hit for hit in hits if hit.payload["metadata"]["parent_id"] == parent_id]
        results.append({"parent_id": parent_id, "page_content": parent["page_content"],
                        "metadata": parent["metadata"], "score": matched[0].score,
                        "matched_children": [str(hit.id) for hit in matched]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--local-path", type=Path, help="Embedded Qdrant for small smoke tests")
    parser.add_argument("--cache", type=Path, default=Path(".cache/models"))
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--chunks", type=Path, default=Path("data/parent_child"))
    index.add_argument("--snapshot", type=Path)
    index.add_argument("--alias", default="handbook")
    index.add_argument("--dense-model", default=DENSE_MODEL)
    index.add_argument("--batch-size", type=int, default=8)
    index.add_argument("--sparse-batch-size", type=int, default=128,
                       help="CPU BM25 tokenization batch")
    index.add_argument("--limit", type=int, help="Sample only; alias automatically gains _sample suffix")
    index.add_argument("--resume", help="Physical collection printed by an interrupted index run")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--alias", default="handbook")
    search.add_argument("--sparse-query", help="Optional English reformulation for English lexical search")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--candidates", type=int, default=40)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.command == "index" and (args.batch_size < 1 or args.sparse_batch_size < 1):
        parser.error("batch-size and sparse-batch-size must be positive")
    client = (QdrantClient(path=str(args.local_path)) if args.local_path else
              QdrantClient(url=args.url, api_key=os.getenv("QDRANT_API_KEY"), timeout=120))
    try:
        if args.command == "index":
            snapshot = args.snapshot
            if snapshot is None:
                manifest = json.loads((args.chunks / "CURRENT.json").read_text(encoding="utf-8"))
                snapshot = args.chunks / manifest["snapshot"]
            embeddings = Embedders(args.dense_model, cache=args.cache, threads=args.threads,
                                   sparse_batch_size=args.sparse_batch_size)
            alias = args.alias + ("_sample" if args.limit is not None else "")
            result = index_snapshot(client, snapshot, embeddings, alias, args.batch_size,
                                    args.limit, args.resume, lambda message: print(message, file=sys.stderr, flush=True))
        else:
            collection = resolve_collection(client, args.alias)
            config = read_state(client, collection)["config"]
            embeddings = Embedders(config["dense_model"], config["sparse_model"], args.cache, args.threads,
                                   bm25_config=config.get("bm25"))
            result = hybrid_search(client, collection, embeddings, args.query, args.sparse_query,
                                   args.top_k, args.candidates)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
