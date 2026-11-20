"""Document-local passage search and arithmetic for streaming knowledge SDG."""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import re
import unicodedata
from collections import Counter

from .client import Chunk


def _terms(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold())


class LocalRetrievalClient:
    """Small BM25 index over ONE source document; no external service or embeddings.

    Only the documents currently being synthesized reside in memory. Chunk ids
    include the sample id so citations remain unique across the output dataset.
    """

    def __init__(self, text: str, sample_id: str, doc_id: str, *,
                 chunk_chars: int = 1800, overlap_chars: int = 200,
                 oversample_factor: int = 2):
        if not 0 <= overlap_chars < chunk_chars:
            raise ValueError("Require 0 <= overlap_chars < chunk_chars")
        self.oversample_factor = max(1, oversample_factor)
        self.chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                if boundary > start + chunk_chars // 2:
                    end = boundary
            passage = text[start:end].strip()
            if passage:
                chunk_id = "h" + hashlib.sha256(f"{sample_id}:{len(self.chunks)}".encode()).hexdigest()[:12]
                self.chunks.append(Chunk(id=chunk_id, text=passage,
                                         doc_id=doc_id))
            if end == len(text):
                break
            start = max(start + 1, end - overlap_chars)
        self.counts = [Counter(_terms(c.text)) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.df = Counter(term for counts in self.counts for term in counts)

    def retrieve(self, query: str, k: int, *, rng) -> list[Chunk]:
        terms = set(_terms(query))
        ranked = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            for term in terms & counts.keys():
                idf = math.log1p((len(self.chunks) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                tf = counts[term]
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.avg_length))
            if score > 0:
                ranked.append((score, i))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        pool = ranked[:k * self.oversample_factor]
        if len(pool) > k:
            pool = [pool[i] for i in sorted(rng.sample(range(len(pool)), k))]
        return [Chunk(id=self.chunks[i].id, text=self.chunks[i].text, doc_id=self.chunks[i].doc_id,
                      score=score) for score, i in pool]


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}


def calculate(expression: str) -> dict:
    """Evaluate bounded numeric arithmetic through AST nodes (never eval/exec)."""
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent exceeds 100")
            value = _OPS[type(node.op)](left, right)
        else:
            raise ValueError("Use numbers, parentheses and + - * / // % ** only")
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Result is outside the supported numeric range")
        return value

    try:
        if len(expression) > 200:
            raise ValueError("Expression exceeds 200 characters")
        tree = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("Expression is too complex")
        return {"expression": expression, "result": evaluate(tree.body)}
    except (ValueError, SyntaxError, ArithmeticError, TypeError) as exc:
        return {"error": str(exc)}
