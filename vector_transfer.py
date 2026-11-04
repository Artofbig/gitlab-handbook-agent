"""Pack notebook input, export resumable embeddings, and import without inference."""
from __future__ import annotations

import argparse
import gzip
import json
from itertools import islice
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from hybrid_search import (DENSE_MODEL, SPARSE_MODEL, LEGACY_SPARSE_MODEL, Embedders, batches, digest,
                           index_snapshot, read_state, records)
from qdrant_client import QdrantClient, models


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        temporary = stream.name
    # Windows sync/indexing software may briefly hold the destination open.
    # Keep the old manifest intact and retry the same atomic replacement.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def latest_snapshot(root):
    current = json.loads((root / "CURRENT.json").read_text(encoding="utf-8"))
    return root / current["snapshot"]


def pack_input(snapshot, destination):
    """Include only code and public corpus files, never .env or credentials."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).parent
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("hybrid_search.py", "vector_transfer.py", "bm25_text.py", "requirements-gpu.txt"):
            archive.write(code_root / name, name)
        for name in ("parents.jsonl", "children.jsonl"):
            archive.write(Path(snapshot) / name, f"snapshot/{name}")
    return str(destination.resolve())


def checked_vector(dense, sparse, dimension, sparse_model=SPARSE_MODEL):
    if len(dense) != dimension or not all(math.isfinite(x) for x in dense):
        raise ValueError("Invalid dense vector dimension or nonfinite values")
    indices, values = sparse["indices"], sparse["values"]
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        raise ValueError("Invalid sparse vector lengths or duplicate indices")
    maximum = 30522 if sparse_model == LEGACY_SPARSE_MODEL else 2**32
    if any(type(index) is not int or not 0 <= index < maximum for index in indices):
        raise ValueError("Invalid sparse vocabulary index")
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Invalid sparse weight")


def check_reuse_path(root):
    if not (Path(root) / "manifest.json").is_file():
        raise ValueError(f"No saved export manifest at {Path(root) / 'manifest.json'}. "
                         "Use the actual old export directory or omit --reuse-dense; "
                         "add --resume if the new output directory already has a manifest.")


def export_vectors(snapshot, output, encoder, batch_size=16, shard_size=256,
                   limit=None, resume=False, progress=print, reuse_dense=None):
    if min(batch_size, shard_size) < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch/shard size and limit must be positive")
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    reuse_manifest, reused_vectors = None, None
    if reuse_dense:
        check_reuse_path(reuse_dense)
        reuse_manifest = validate_bundle(reuse_dense, complete=False)
    if hasattr(encoder, "fit_bm25"):
        if progress:
            progress("Preparing lexical corpus statistics (BM25 on CPU; no dense inference)")
        encoder.fit_bm25(row["page_content"] for row in islice(records(snapshot / "children.jsonl"), limit))
    config = {**encoder.config(), "limit": limit,
              "children_sha256": digest(snapshot / "children.jsonl"),
              "parents_sha256": digest(snapshot / "parents.jsonl")}
    if config["sparse_model"] not in {SPARSE_MODEL, LEGACY_SPARSE_MODEL}:
        raise ValueError("Unsupported sparse model")
    if reuse_manifest:
        for key in ("dense_model", "dimension", "children_sha256", "parents_sha256"):
            if reuse_manifest["config"][key] != config[key]:
                raise ValueError("Dense reuse requires the same dense model and corpus")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if resume:
        manifest = validate_bundle(output, complete=False)
        if manifest["config"] != config:
            raise ValueError("Resume requires the same corpus, models and limit")
    else:
        if any(output.iterdir()):
            raise ValueError("Output is not empty; use --resume or choose another directory")
        for name in ("parents.jsonl", "children.jsonl"):
            shutil.copyfile(snapshot / name, output / name)
        total = sum(1 for _ in islice(records(output / "children.jsonl"), limit))
        if not total:
            raise ValueError("No children to encode")
        manifest = {"format": 1, "config": config, "status": "building",
                    "expected_children": total, "indexed_children": 0, "shards": []}
        atomic_json(path, manifest)
    if manifest["status"] == "ready":
        return manifest
    if reuse_manifest:
        reused_vectors = islice(vector_rows(reuse_dense, reuse_manifest), manifest["indexed_children"], None)
    position = manifest["indexed_children"]
    remaining = islice(records(output / "children.jsonl"), manifest["indexed_children"], limit)
    for shard in batches(remaining, shard_size):
        name = f"vectors-{len(manifest['shards']):06d}.jsonl.gz"
        temporary = output / (name + ".tmp")
        # A failed shard is recomputed; already committed shards are not encoded again.
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for batch in batches(shard, batch_size):
                texts = [row["page_content"] for row in batch]
                cached_count = min(len(batch), max(0, reuse_manifest["indexed_children"] - position)) if reuse_manifest else 0
                if cached_count:
                    dense = []
                    for child in batch[:cached_count]:
                        cached = next(reused_vectors)
                        if child["id"] != cached["id"]:
                            raise ValueError("Cached dense ID does not match the corpus")
                        dense.append(cached["dense"])
                    dense.extend(encoder.encode_dense(texts[cached_count:]))
                    vectors = list(zip(dense, encoder.encode_sparse(texts), strict=True))
                else:
                    vectors = encoder.encode_documents(texts)
                for row, (dense, sparse) in zip(batch, vectors, strict=True):
                    sparse = sparse.model_dump()
                    checked_vector(dense, sparse, encoder.dimension, config["sparse_model"])
                    stream.write(json.dumps({"id": row["id"], "dense": dense, "sparse": sparse},
                                             allow_nan=False) + "\n")
                position += len(batch)
        os.replace(temporary, output / name)
        manifest["shards"].append({"file": name, "sha256": digest(output / name), "count": len(shard)})
        manifest["indexed_children"] += len(shard)
        atomic_json(path, manifest)
        if progress:
            progress(f"Exported {manifest['indexed_children']}/{manifest['expected_children']} children")
    if manifest["indexed_children"] != manifest["expected_children"]:
        raise ValueError("Incomplete export")
    manifest["status"] = "ready"
    atomic_json(path, manifest)
    return manifest


def vector_rows(root, manifest):
    for shard in manifest["shards"]:
        name = shard["file"]
        if Path(name).name != name or not name.startswith("vectors-") or not name.endswith(".jsonl.gz"):
            raise ValueError("Invalid shard filename")
        with gzip.open(Path(root) / name, "rt", encoding="utf-8") as stream:
            yield from (json.loads(line) for line in stream)


def validate_bundle(root, complete=True):
    """Validate integrity and alignment before touching Qdrant or resuming inference."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != 1 or manifest["status"] not in {"ready", "building"}:
        raise ValueError("Unsupported export format or state")
    if complete and manifest["status"] != "ready":
        raise ValueError("Export is incomplete; resume GPU export first")
    config = manifest["config"]
    if config["sparse_model"] not in {SPARSE_MODEL, LEGACY_SPARSE_MODEL}:
        raise ValueError("Unsupported sparse export")
    for name in ("parents", "children"):
        if digest(root / f"{name}.jsonl") != config[f"{name}_sha256"]:
            raise ValueError(f"{name} file checksum mismatch")
    total = sum(1 for _ in islice(records(root / "children.jsonl"), config["limit"]))
    if total != manifest["expected_children"]:
        raise ValueError("Child count mismatch")
    count = 0
    for shard in manifest["shards"]:
        name = shard["file"]
        if Path(name).name != name or not name.startswith("vectors-") or not name.endswith(".jsonl.gz"):
            raise ValueError("Invalid shard filename")
        if digest(root / name) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {name}")
        with gzip.open(root / name, "rt", encoding="utf-8") as stream:
            count_in_shard = sum(1 for _ in stream)
        if count_in_shard != shard["count"]:
            raise ValueError("Shard count mismatch")
        count += count_in_shard
    if count != manifest["indexed_children"] or count > total or (complete and count != total):
        raise ValueError("Export count mismatch")
    parents = {row["id"] for row in records(root / "parents.jsonl")}
    seen = set()
    children = islice(records(root / "children.jsonl"), count)
    for child, row in zip(children, vector_rows(root, manifest), strict=True):
        if child["id"] != row["id"] or row["id"] in seen:
            raise ValueError("Vector IDs do not match child order, or are duplicated")
        if child["metadata"]["parent_id"] not in parents:
            raise ValueError("Child references a missing parent")
        seen.add(row["id"])
        checked_vector(row["dense"], row["sparse"], config["dimension"], config["sparse_model"])
    return manifest


class StoredEmbedders:
    """Replay validated precomputed vectors; never import or call an embedding model."""
    def __init__(self, root, manifest, offset=0):
        self.root, self.manifest = Path(root), manifest
        self.dimension = manifest["config"]["dimension"]
        self.children = islice(records(self.root / "children.jsonl"), offset, None)
        self.vectors = islice(vector_rows(self.root, manifest), offset, None)

    def config(self):
        config = self.manifest["config"]
        return {name: config[name] for name in ("dense_model", "sparse_model", "dimension", "bm25") if name in config}

    def encode_documents(self, texts):
        vectors = []
        for text in texts:
            child, row = next(self.children), next(self.vectors)
            if text != child["page_content"] or child["id"] != row["id"]:
                raise ValueError("Precomputed vector alignment error")
            vectors.append((row["dense"], models.SparseVector(**row["sparse"])))
        return vectors


def import_vectors(client, root, alias="handbook", batch_size=128, resume=None, progress=print):
    manifest = validate_bundle(root)
    config = manifest["config"]
    offset = read_state(client, resume)["indexed_children"] if resume else 0
    encoder = StoredEmbedders(root, manifest, offset)
    alias += "_sample" if config["limit"] is not None else ""
    return index_snapshot(client, Path(root), encoder, alias, batch_size,
                          config["limit"], resume, progress)


def pack_result(root, destination):
    manifest = validate_bundle(root)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ["manifest.json", "parents.jsonl", "children.jsonl", *[s["file"] for s in manifest["shards"]]]:
            archive.write(Path(root) / name, name)
    return str(destination.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--snapshot", type=Path)
    pack.add_argument("--chunks", type=Path, default=Path("data/parent_child"))
    pack.add_argument("--output", type=Path, default=Path("data/handbook_bm25_input.zip"))
    export = sub.add_parser("export")
    export.add_argument("--snapshot", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    export.add_argument("--dense-model", default=DENSE_MODEL)
    export.add_argument("--batch-size", type=int, default=16)
    export.add_argument("--sparse-batch-size", type=int, default=128)
    export.add_argument("--shard-size", type=int, default=256)
    export.add_argument("--limit", type=int)
    export.add_argument("--resume", action="store_true")
    export.add_argument("--reuse-dense", type=Path, help="Reuse the saved prefix from an old SPLADE/BM25 export")
    load = sub.add_parser("import")
    load.add_argument("--input", type=Path, required=True)
    load.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    load.add_argument("--local-path", type=Path)
    load.add_argument("--alias", default="handbook")
    load.add_argument("--batch-size", type=int, default=128)
    load.add_argument("--resume")
    archive = sub.add_parser("archive")
    archive.add_argument("--input", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        result = pack_input(args.snapshot or latest_snapshot(args.chunks), args.output)
    elif args.command == "export":
        if args.reuse_dense:
            try:
                check_reuse_path(args.reuse_dense)
            except ValueError as exc:
                parser.error(str(exc))
        encoder = Embedders(args.dense_model, sparse_batch_size=args.sparse_batch_size, device=args.device)
        result = export_vectors(args.snapshot, args.output, encoder, args.batch_size,
                                args.shard_size, args.limit, args.resume, reuse_dense=args.reuse_dense)
    elif args.command == "archive":
        result = pack_result(args.input, args.output)
    else:
        client = (QdrantClient(path=str(args.local_path)) if args.local_path else
                  QdrantClient(url=args.url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=120))
        try:
            result = import_vectors(client, args.input, args.alias, args.batch_size, args.resume)
        finally:
            client.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
