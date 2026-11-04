"""Build LangChain parent/child chunks from an exported Handbook snapshot."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class ChunkConfig:
    # Character counts, not tokens. Tune against retrieval quality later.
    parent_size: int = 3000
    parent_overlap: int = 300
    child_size: int = 600
    child_overlap: int = 100

    def validate(self):
        if not 0 <= self.parent_overlap < self.parent_size:
            raise ValueError("Require parent-size > parent-overlap >= 0")
        if not 0 <= self.child_overlap < self.child_size:
            raise ValueError("Require child-size > child-overlap >= 0")
        if self.child_size >= self.parent_size:
            raise ValueError("child-size must be smaller than parent-size")


def make_splitters(config: ChunkConfig):
    config.validate()
    options = dict(add_start_index=True, strip_whitespace=False)
    return (
        RecursiveCharacterTextSplitter(chunk_size=config.parent_size,
                                      chunk_overlap=config.parent_overlap, **options),
        RecursiveCharacterTextSplitter(chunk_size=config.child_size,
                                      chunk_overlap=config.child_overlap, **options),
    )


def split_document(record: dict, config: ChunkConfig, splitters=None):
    """Return LangChain Documents: children carry parent_id for later expansion."""
    parent_splitter, child_splitter = splitters or make_splitters(config)
    metadata = {key: value for key, value in record.items() if key != "text"}
    text = record["text"]
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = json.dumps([record["document_id"], record.get("commit"), fingerprint,
                           asdict(config)], sort_keys=True)
    parents, children = [], []
    for index, parent in enumerate(parent_splitter.split_documents([
            Document(page_content=text, metadata=metadata)])):
        parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{identity}:parent:{index}"))
        parent.id = parent_id
        parent.metadata.update(parent_id=parent_id, parent_index=index,
                               chunk_type="parent")
        parent.metadata["end_index"] = parent.metadata["start_index"] + len(parent.page_content)
        parents.append(parent)
        for child_index, child in enumerate(child_splitter.split_documents([parent])):
            # LangChain's child start_index is relative to its parent.
            offset = child.metadata["start_index"]
            child.id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{parent_id}:child:{child_index}"))
            child.metadata.update(child_index=child_index, chunk_type="child",
                                  start_in_parent=offset,
                                  start_index=parent.metadata["start_index"] + offset)
            child.metadata["end_index"] = child.metadata["start_index"] + len(child.page_content)
            children.append(child)
    return parents, children


def serialize(document: Document):
    return {"id": document.id, "page_content": document.page_content,
            "metadata": document.metadata}


def load_parents(path: Path) -> dict[str, Document]:
    """Simple local parent store; replace with a persistent docstore at larger scale."""
    with path.open(encoding="utf-8") as stream:
        return {row["id"]: Document(**row)
                for row in (json.loads(line) for line in stream)}


def expand_children(hits: list[Document], parents: dict[str, Document]) -> list[Document]:
    """Resolve ranked child hits to unique parents, preserving first-hit order."""
    ids = dict.fromkeys(hit.metadata["parent_id"] for hit in hits)
    return [parents[parent_id] for parent_id in ids]


def build(source: Path, output: Path, config: ChunkConfig):
    source, output = source.resolve(), output.resolve()
    splitters = make_splitters(config)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "snapshots"
    snapshots.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="parent-child-", dir=snapshots))
    count = parent_count = child_count = 0
    with source.open(encoding="utf-8") as stream, \
            (run / "parents.jsonl").open("w", encoding="utf-8") as parent_file, \
            (run / "children.jsonl").open("w", encoding="utf-8") as child_file:
        for line in stream:
            parents, children = split_document(json.loads(line), config, splitters)
            for doc in parents:
                parent_file.write(json.dumps(serialize(doc), ensure_ascii=False) + "\n")
            for doc in children:
                child_file.write(json.dumps(serialize(doc), ensure_ascii=False) + "\n")
            count += 1
            parent_count += len(parents)
            child_count += len(children)
    if not child_count:
        raise ValueError("No child chunks generated; previous CURRENT.json is unchanged")
    manifest = {"source": str(source.resolve()), "documents": count,
                "parents": parent_count, "children": child_count,
                "config": asdict(config), "snapshot": run.relative_to(output).as_posix()}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                     suffix=".tmp", delete=False) as pointer:
        json.dump(manifest, pointer, indent=2)
        temporary = pointer.name
    os.replace(temporary, output / "CURRENT.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="documents.jsonl; default: latest export")
    parser.add_argument("--corpus", type=Path, default=Path("data/handbook"))
    parser.add_argument("--output", type=Path, default=Path("data/parent_child"))
    for name, default in asdict(ChunkConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    config = ChunkConfig(args.parent_size, args.parent_overlap, args.child_size, args.child_overlap)
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    source = args.input
    if source is None:
        current = json.loads((args.corpus / "CURRENT.json").read_text(encoding="utf-8"))
        source = args.corpus / current["snapshot"] / "documents.jsonl"
    print(json.dumps(build(source, args.output, config), indent=2))


if __name__ == "__main__":
    main()
