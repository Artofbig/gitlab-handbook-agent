import json
import os
from pathlib import Path
import tempfile
import unittest

from chunk_handbook import ChunkConfig, build, expand_children, load_parents, split_document


class ParentChildTests(unittest.TestCase):
    def setUp(self):
        self.record = {"document_id": "doc-a", "commit": "abc", "title": "Rules",
                       "source_url": "https://example.com/source",
                       "text": "\n\n".join(f"## Rule {i}\n" + "Ask your manager first. " * 12
                                           for i in range(20))}
        self.config = ChunkConfig(700, 80, 160, 30)

    def test_relationships_offsets_and_limits(self):
        parents, children = split_document(self.record, self.config)
        lookup = {parent.id: parent for parent in parents}
        self.assertGreater(len(parents), 1)
        self.assertGreater(len(children), len(parents))
        for parent in parents:
            self.assertLessEqual(len(parent.page_content), self.config.parent_size)
        for child in children:
            parent = lookup[child.metadata["parent_id"]]
            self.assertLessEqual(len(child.page_content), self.config.child_size)
            offset = child.metadata["start_in_parent"]
            self.assertEqual(parent.page_content[offset:offset + len(child.page_content)], child.page_content)
            self.assertEqual(self.record["text"][child.metadata["start_index"]:child.metadata["end_index"]],
                             child.page_content)
            self.assertEqual(child.metadata["source_url"], self.record["source_url"])
        repeated = [children[0], children[0], children[-1]]
        self.assertEqual([parent.id for parent in expand_children(repeated, lookup)],
                         list(dict.fromkeys(child.metadata["parent_id"] for child in repeated)))
        again = split_document(self.record, self.config)
        self.assertEqual([c.id for c in children], [c.id for c in again[1]])
        changed = split_document({**self.record, "text": self.record["text"] + "updated"}, self.config)
        self.assertNotEqual(children[0].id, changed[1][0].id)

    def test_export_store_and_failed_publication(self):
        cache = Path(__file__).parent / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            root = Path(temp)
            source = root / "documents.jsonl"
            source.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
            output = root / "chunks"
            manifest = build(Path(os.path.relpath(source)), Path(os.path.relpath(output)), self.config)
            run = output / manifest["snapshot"]
            parents = load_parents(run / "parents.jsonl")
            self.assertEqual(len(parents), manifest["parents"])
            with (run / "children.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    self.assertIn(json.loads(line)["metadata"]["parent_id"], parents)
            pointer = (output / "CURRENT.json").read_bytes()
            source.write_text("invalid JSON", encoding="utf-8")
            with self.assertRaises(ValueError):
                build(source, output, self.config)
            self.assertEqual((output / "CURRENT.json").read_bytes(), pointer)

    def test_invalid_configuration(self):
        for config in (ChunkConfig(child_size=3000), ChunkConfig(child_overlap=600),
                       ChunkConfig(parent_size=0)):
            with self.assertRaises(ValueError):
                split_document(self.record, config)


if __name__ == "__main__":
    unittest.main()
