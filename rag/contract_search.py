"""Real RAG: semantic search over unstructured SLA contract documents.

This is deliberately separate from harness/suppliers.py's get_sla_terms(), which
is an exact-match dict lookup against structured data (mock_suppliers.json) — that
was never RAG, just a database read that the original resolv.md mislabeled as one.
Real RAG only makes sense once there's actually unstructured text to search: the
full prose contract documents in data/contracts/*.txt, where the answer to "what
does clause 3.2 say" isn't a dict key, it's a paragraph you have to find.

Single-stage retrieval (embed + cosine similarity), no reranking step: the
reference two-stage design (bi-encoder top-20 -> cross-encoder rerank top-3) makes
sense over thousands of documents. With 2 contracts and ~12 clauses total, a
second model adds cost and complexity without a measurable accuracy gain at this
corpus size — same "add complexity only when it earns its keep" principle used
everywhere else in this project. Revisit if the contract corpus grows into the
hundreds of documents.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from config import CONTRACTS_DIR

_CLAUSE_PATTERN = re.compile(
    r"Clause (\d+\.\d+) — ([^\n]+)\n((?:(?!Clause \d+\.\d+ —)(?!Master Services Agreement).)*)",
    re.DOTALL,
)


@dataclass
class ClauseChunk:
    supplier_id: str
    clause_id: str
    title: str
    text: str


_model = None
_chunks: list[ClauseChunk] | None = None
_embeddings = None  # numpy array, one row per chunk, L2-normalized


def _load_chunks() -> list[ClauseChunk]:
    chunks = []
    for path in Path(CONTRACTS_DIR).glob("*.txt"):
        supplier_id = path.stem
        text = path.read_text(encoding="utf-8")
        for match in _CLAUSE_PATTERN.finditer(text):
            clause_id, title, body = match.groups()
            chunks.append(
                ClauseChunk(
                    supplier_id=supplier_id,
                    clause_id=clause_id,
                    title=title.strip(),
                    text=f"Clause {clause_id} — {title.strip()}: {' '.join(body.split())}",
                )
            )
    return chunks


def _ensure_index() -> None:
    """Lazy-loaded: importing this module doesn't pay the embedding-model load
    cost unless search() is actually called.
    """
    global _model, _chunks, _embeddings
    if _chunks is not None:
        return

    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _chunks = _load_chunks()
    _embeddings = _model.encode([c.text for c in _chunks], normalize_embeddings=True)


def search(query: str, supplier_id: str | None = None, top_k: int = 3) -> list[dict]:
    """Returns the top_k most semantically relevant contract clauses for `query`,
    optionally restricted to one supplier. Each result: {supplier_id, clause_id,
    title, text, score} — score is cosine similarity, 1.0 = identical meaning.
    """
    _ensure_index()
    query_embedding = _model.encode([query], normalize_embeddings=True)[0]

    scored = [
        (float(_embeddings[i] @ query_embedding), _chunks[i])
        for i in range(len(_chunks))
        if supplier_id is None or _chunks[i].supplier_id == supplier_id
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {
            "supplier_id": chunk.supplier_id,
            "clause_id": chunk.clause_id,
            "title": chunk.title,
            "text": chunk.text,
            "score": score,
        }
        for score, chunk in scored[:top_k]
    ]
