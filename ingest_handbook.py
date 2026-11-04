"""Export public GitLab Handbook Markdown documents from a Git snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit

import yaml

REPOSITORY = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
PROJECT = REPOSITORY.removesuffix(".git")
SITE = "https://handbook.gitlab.com"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def sync(repo: Path, offline: bool) -> str:
    if not repo.exists():
        if offline:
            raise ValueError("Cache does not exist; run once without --offline")
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", "--depth", "1", "--single-branch",
             REPOSITORY, str(repo)], check=True,
        )
        return git(repo, "rev-parse", "HEAD")
    if git(repo, "rev-parse", "--is-bare-repository") != "true":
        raise ValueError("Use a dedicated bare cache, not a working repository")
    if git(repo, "remote", "get-url", "origin") != REPOSITORY:
        raise ValueError("Unexpected cache origin")
    if offline:
        try:
            return git(repo, "rev-parse", "--verify", "--quiet", "FETCH_HEAD^{commit}")
        except subprocess.CalledProcessError:
            return git(repo, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "--depth", "1", "origin", "HEAD"],
        check=True,
    )
    return git(repo, "rev-parse", "--verify", "FETCH_HEAD^{commit}")


def markdown_blobs(repo: Path, commit: str, prefix: str):
    """Read Git objects directly: no checkout, symlink following, or site code execution."""
    tree = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", commit, "--", prefix]
    )
    entries = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        info, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = info.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if mode in {"100644", "100755"} and kind == "blob" and path.lower().endswith(".md"):
            entries.append((path, oid))
    proc = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    try:
        for path, oid in sorted(entries):
            proc.stdin.write((oid + "\n").encode("ascii"))
            proc.stdin.flush()
            header = proc.stdout.readline().decode("ascii").split()
            if len(header) != 3 or header[1] != "blob":
                raise RuntimeError(f"Cannot read {path}: {header}")
            size = int(header[2])
            body = proc.stdout.read(size)
            if len(body) != size or proc.stdout.read(1) != b"\n":
                raise RuntimeError(f"Incomplete Git object: {path}")
            yield path, body.decode("utf-8-sig")
    finally:
        proc.stdin.close()
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError("git cat-file failed")


def parse_markdown(text: str):
    # Hugo supports YAML, TOML and JSON front matter. Handbook primarily uses YAML.
    text = text.lstrip("\ufeff")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.S)
    if match:
        meta = yaml.safe_load(match[1]) or {}
        body = text[match.end():]
    else:
        match = re.match(r"\A\+\+\+\s*\n(.*?)\n\+\+\+\s*(?:\n|$)", text, re.S)
        if match:
            import tomllib
            meta = tomllib.loads(match[1])
            body = text[match.end():]
        elif text.startswith("{"):
            meta, end = json.JSONDecoder().raw_decode(text)
            body = text[end:]
        else:
            meta, body = {}, text
    if not isinstance(meta, dict):
        raise ValueError("Front matter must be a mapping")
    return meta, body.strip()


def page_url(path: str, meta: dict) -> str:
    explicit = meta.get("url")
    if explicit:
        parts = urlsplit(str(explicit))
        if parts.scheme or parts.netloc or not parts.path.startswith("/"):
            raise ValueError(f"Unexpected page URL: {explicit}")
        return SITE + quote(parts.path, safe="/%")
    relative = PurePosixPath(path).relative_to("content")
    if relative.name in {"index.md", "_index.md"}:
        relative = relative.parent
    else:
        relative = relative.with_suffix("")
        if meta.get("slug"):
            slug = str(meta["slug"])
            if "/" in slug or slug in {".", ".."}:
                raise ValueError(f"Unexpected slug: {slug}")
            relative = relative.parent / slug
    return SITE + "/" + quote(str(relative).strip("./"), safe="/") + "/"


def write_json(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def export(repo: Path, output: Path, commit: str, prefix: str):
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "snapshots"
    snapshots.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=commit[:12] + "-", dir=snapshots))
    count, skipped = 0, 0
    retrieved_at = datetime.now(timezone.utc).isoformat()
    with (run / "documents.jsonl").open("w", encoding="utf-8") as docs:
        for path, raw in markdown_blobs(repo, commit, prefix):
            try:
                meta, body = parse_markdown(raw)
                if meta.get("draft") is True or not body:
                    skipped += 1
                    continue
                url = page_url(path, meta)
            except Exception as exc:
                raise ValueError(f"Invalid document {path}: {exc}") from exc
            heading = re.search(r"^#\s+(.+)$", body, re.M)
            doc_id = hashlib.sha256(path.encode()).hexdigest()
            payload = {
                "document_id": doc_id, "source_path": path,
                "source_url": f"{PROJECT}/-/blob/{commit}/{quote(path, safe='/')}",
                "url": url, "commit": commit, "retrieved_at": retrieved_at,
                "title": str(meta.get("title") or (heading[1] if heading else PurePosixPath(path).stem)),
                "section": path.removeprefix("content/handbook/").split("/")[0],
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
            write_json(docs, {**payload, "metadata": meta, "text": body})
            count += 1
    if not count:
        raise ValueError(f"No nonempty Markdown documents under {prefix}")
    manifest = {"repository": REPOSITORY, "commit": commit, "retrieved_at": retrieved_at,
                "documents": count, "skipped": skipped,
                "prefix": prefix,
                "snapshot": run.relative_to(output).as_posix()}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Publish only a complete snapshot. Failed runs leave the previous pointer intact.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                     suffix=".tmp", delete=False) as pointer:
        json.dump(manifest, pointer, indent=2)
        temporary = pointer.name
    os.replace(temporary, output / "CURRENT.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(".cache/handbook.git"))
    parser.add_argument("--output", type=Path, default=Path("data/handbook"))
    parser.add_argument("--prefix", default="content/handbook/")
    parser.add_argument("--offline", action="store_true", help="Export the cached snapshot without network")
    args = parser.parse_args()
    if not args.prefix.startswith("content/") or ".." in args.prefix.split("/"):
        parser.error("prefix must be inside content/")
    commit = sync(args.repo.resolve(), args.offline)
    print(json.dumps(export(args.repo.resolve(), args.output.resolve(), commit,
                            args.prefix), indent=2))


if __name__ == "__main__":
    main()
