"""search_policy tool: retrieval-augmented FAQ/policy Q&A.

Policy docs (data/policies/*.md) are chunked, embedded, and stored in a
local Chroma collection. The `search_policy` tool embeds a customer's
question and returns the most relevant chunks, so the model can answer
*only* from what was actually retrieved instead of from memory — see
SYSTEM_PROMPT's instruction not to guess when nothing relevant comes back.
This is this project's first hallucination-avoidance mechanism.

Embeddings are behind a swappable backend (EMBEDDING_BACKEND env var):
  - "local" (default): Chroma's bundled MiniLM model. Free, no signup, no
    API key — works immediately. Downloads its small ONNX model on first
    use, then runs fully offline.
  - "voyage": Voyage AI, as PROJECT_PLAN.md's tech stack specifies. Needs
    VOYAGE_API_KEY. Uses Voyage's asymmetric document/query embeddings
    (embed documents with input_type="document", queries with "query"),
    which Voyage recommends for retrieval quality.
Switching is one env var — nothing else in this file or its callers change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import chromadb

POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "policies"
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "policies"

# Cosine distance (0 = identical, ~1 = unrelated) below which a retrieved
# chunk is considered relevant enough to hand to the model. Checked against
# 8 hand-labeled questions on the local MiniLM backend: in-scope questions
# topped out at 0.504, uncovered ones started at 0.572 — a real gap, not a
# guess, though only tested against those 8 and this specific model. If you
# switch to EMBEDDING_BACKEND=voyage, re-check this the same way (see
# scripts in the Phase 3 README section) — a different embedding model
# means a different distance distribution.
RELEVANCE_THRESHOLD = 0.55


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _to_float_list(vector: Any) -> list[float]:
    """Normalize one embedding to a plain list of native Python floats.

    chromadb's embedding functions return numpy arrays (or a list of numpy
    scalars once naively wrapped in list()). Its add() path tolerates numpy
    types, but its query() path is stricter and rejects a list containing
    numpy.float32 scalars — .tolist() converts recursively to native types,
    which satisfies both.
    """
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return [float(x) for x in vector]


class LocalEmbeddingBackend:
    """Free, local, no API key: chromadb's bundled MiniLM ONNX model."""

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_to_float_list(v) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return _to_float_list(self._fn([text])[0])


class VoyageEmbeddingBackend:
    """Voyage AI embeddings — PROJECT_PLAN.md's specified choice. Requires
    VOYAGE_API_KEY. Imports voyageai lazily so the "local" backend never
    needs that package installed at all.
    """

    def __init__(self, model: str | None = None, client: Any = None) -> None:
        import voyageai

        self.model = model or os.getenv("VOYAGE_MODEL", "voyage-3")
        self.client = client or voyageai.Client()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts, model=self.model, input_type="document").embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed([text], model=self.model, input_type="query").embeddings[0]


def get_embedding_backend() -> EmbeddingBackend:
    """EMBEDDING_BACKEND env var: "local" (default) or "voyage"."""
    backend_name = os.getenv("EMBEDDING_BACKEND", "local").lower()
    if backend_name == "voyage":
        return VoyageEmbeddingBackend()
    if backend_name != "local":
        raise ValueError(f"unknown EMBEDDING_BACKEND: {backend_name!r} (expected 'local' or 'voyage')")
    return LocalEmbeddingBackend()


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _load_policy_documents() -> list[tuple[str, str]]:
    """Return (filename, raw text) for every .md file in data/policies/."""
    return [(path.name, path.read_text(encoding="utf-8")) for path in sorted(POLICIES_DIR.glob("*.md"))]


def _chunk_document(filename: str, text: str) -> list[dict[str, str]]:
    """Split one policy doc into paragraph-level chunks.

    The doc's first line is its title (a "# Heading"); it's prefixed onto
    every chunk from that doc so each chunk is self-contained context for
    retrieval — "...must be in original packaging." means nothing on its
    own without knowing it's from the Returns policy.
    """
    lines = text.strip().splitlines()
    title = lines[0].lstrip("#").strip() if lines else filename
    body = "\n".join(lines[1:]).strip()
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    return [
        {
            "id": f"{filename}::{i}",
            "source": filename,
            "title": title,
            "text": f"{title}\n\n{paragraph}",
        }
        for i, paragraph in enumerate(paragraphs)
    ]


def ingest_policies(collection: Any = None, backend: EmbeddingBackend | None = None) -> int:
    """(Re)build the policies collection from data/policies/*.md.

    Wipes and rebuilds from scratch every time — same convention as
    data/mock_db.py's reset_and_seed(): the policy docs are static fixture
    data, not something worth diffing/updating incrementally. Returns the
    number of chunks ingested.
    """
    collection = collection or _get_collection()
    backend = backend or get_embedding_backend()

    chunks: list[dict[str, str]] = []
    for filename, text in _load_policy_documents():
        chunks.extend(_chunk_document(filename, text))

    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    if not chunks:
        return 0

    embeddings = backend.embed_documents([c["text"] for c in chunks])
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
    )
    return len(chunks)


TOOL_SCHEMA: dict[str, Any] = {
    "name": "search_policy",
    "description": (
        "Search this store's official policy documents (returns, refunds, "
        "shipping, warranty, cancellations, etc.) for text relevant to a "
        "customer's question. Always use this before answering any "
        "policy/FAQ question — never answer from memory. If it returns no "
        "results, that means the policy docs don't cover this; say so "
        "honestly rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The customer's question, in their own words.",
            }
        },
        "required": ["query"],
    },
}


def search_policy(
    query: str,
    k: int = 3,
    collection: Any = None,
    backend: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    """Embed `query` and return the most relevant policy chunks, if any clear
    that relevance bar. Never raises; always returns a structured dict.
    """
    collection = collection or _get_collection()
    backend = backend or get_embedding_backend()

    query_embedding = backend.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    hits: list[dict[str, Any]] = []
    ids = results["ids"][0] if results["ids"] else []
    if ids:
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            if distance <= RELEVANCE_THRESHOLD:
                hits.append(
                    {"source": meta["source"], "title": meta["title"], "text": doc, "distance": distance}
                )

    if not hits:
        return {
            "found": False,
            "message": "No policy documents matched this question closely enough to answer from.",
        }
    return {"found": True, "results": hits}


if __name__ == "__main__":
    count = ingest_policies()
    print(f"Ingested {count} chunks from {POLICIES_DIR} into {CHROMA_DIR}")
