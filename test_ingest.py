import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ingest_handbook import export, markdown_blobs, page_url, parse_markdown


class IngestionTests(unittest.TestCase):
    def test_frontmatter_and_urls(self):
        meta, text = parse_markdown('---\ntitle: "Time off"\n---\n# Rules\nAsk your manager.')
        self.assertEqual(meta["title"], "Time off")
        self.assertTrue(text.startswith("# Rules"))
        self.assertEqual(page_url("content/handbook/people/_index.md", meta),
                         "https://handbook.gitlab.com/handbook/people/")
        self.assertEqual(page_url("content/handbook/people/test.md", {"slug": "new"}),
                         "https://handbook.gitlab.com/handbook/people/new/")
        self.assertEqual(parse_markdown('+++\ntitle = "A"\n+++\nText')[0]["title"], "A")
        with self.assertRaises(ValueError):
            page_url("content/handbook/a.md", {"url": "https://other.example"})

    def test_git_snapshot_and_failed_publication(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / ".cache", ignore_cleanup_errors=True) as temp:
            repo = Path(temp) / "repo"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            page = repo / "content/handbook/_index.md"
            page.parent.mkdir(parents=True)
            page.write_text('---\ntitle: Handbook\n---\n# Rules\nAsk a manager.', encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                            "user.email=test@example.com", "commit", "-qm", "Fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            # Export must read committed objects, not a modified working tree.
            page.write_text("uncommitted replacement", encoding="utf-8")
            self.assertIn("Ask a manager", list(markdown_blobs(repo, commit, "content/handbook/"))[0][1])
            output = Path(temp) / "output"
            first = export(repo, output, commit, "content/handbook/")
            self.assertEqual(first["documents"], 1)
            document = json.loads((output / first["snapshot"] / "documents.jsonl").read_text(encoding="utf-8"))
            self.assertIn(commit, document["source_url"])
            self.assertEqual(document["text"], "# Rules\nAsk a manager.")
            self.assertFalse((output / first["snapshot"] / "chunks.jsonl").exists())
            pointer = (output / "CURRENT.json").read_bytes()
            with patch("ingest_handbook.markdown_blobs", return_value=iter([("content/handbook/bad.md", "---\na: [\n---\nbroken")])):
                with self.assertRaises(ValueError):
                    export(repo, output, commit, "content/handbook/")
            self.assertEqual((output / "CURRENT.json").read_bytes(), pointer)


if __name__ == "__main__":
    unittest.main()
