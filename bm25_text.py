"""Versioned lexical-only Markdown cleanup and English BM25 configuration."""
import html
import math
from html.parser import HTMLParser
import re
import unicodedata

from markdown_it import MarkdownIt

VERSION = "handbook-markdown-v1"
PARSER = MarkdownIt("commonmark").enable("table")


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden.append(tag)
        elif not self.hidden:
            self.parts.append(" ")
            if tag == "img":
                self.parts.append(dict(attrs).get("alt") or "")

    def handle_endtag(self, tag):
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()
        elif not self.hidden:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_bm25(text):
    # Render Markdown structurally: link labels, headings, table cells and code
    # content survive. URL destinations, HTML attributes and Hugo directives do not.
    text = re.sub(r"\{\{[<%].*?[>%]\}\}", " ", text, flags=re.S)
    parser = PlainHTML()
    parser.feed(PARSER.render(text))
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"https?://\S+", " ", text)
    return " ".join(unicodedata.normalize("NFKC", text).split())


class BM25Encoder:
    def __init__(self, cache, config=None):
        from fastembed import SparseTextEmbedding
        self.settings = config or {"language": "english", "k": 1.2, "b": 0.75,
                                   "avg_len": 256., "cleaner": VERSION,
                                   "stopwords": "none", "stemming": True}
        if (self.settings.get("cleaner") != VERSION or self.settings.get("language") != "english"
                or self.settings.get("stopwords") != "none" or self.settings.get("stemming") is not True):
            raise ValueError("Unsupported BM25 preprocessing configuration; rebuild the index")
        if (not math.isfinite(self.settings["avg_len"]) or self.settings["avg_len"] <= 0
                or not math.isfinite(self.settings["k"]) or self.settings["k"] <= 0
                or not 0 <= self.settings["b"] <= 1):
            raise ValueError("Invalid BM25 k/b/avg_len")
        self.encoder = SparseTextEmbedding(
            model_name="Qdrant/bm25", cache_dir=str(cache), language="english",
            k=self.settings["k"], b=self.settings["b"], avg_len=self.settings["avg_len"],
            stopwords=set(), disable_stemmer=False,
        )

    def fit(self, texts):
        # Use exactly the same post-stem tokens as FastEmbed 0.9.x raw_embed.
        from fastembed.common.utils import remove_non_alphanumeric
        model = self.encoder.model
        count = total = 0
        for text in texts:
            tokens = model.tokenizer.tokenize(remove_non_alphanumeric(clean_bm25(text)))
            total += len(model._stem(tokens))
            count += 1
        self.settings = {**self.settings, "avg_len": max(total / count, 1.) if count else 1.}
        model.avg_len = self.settings["avg_len"]

    def embed(self, texts, batch_size=256):
        return self.encoder.embed([clean_bm25(text) for text in texts], batch_size=batch_size)

    def query_embed(self, text):
        return self.encoder.query_embed(clean_bm25(text))
