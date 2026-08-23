"""Phase 3 checkpoint (retrieval half): search_policy retrieves relevant
chunks for in-scope questions and correctly abstains (found: False) for
ones the docs don't cover — the tool-level safeguard behind "the agent
shouldn't invent an answer."

Most tests use a small synthetic corpus + a deterministic fake embedding
backend (a bag-of-words hash: meaningfully similar for shared vocabulary,
~orthogonal otherwise) and an in-memory Chroma collection — no network, no
API key, no dependency on the real 16-doc corpus' exact wording.

The last test uses the REAL policy corpus and the real local embedding
backend (still no API key needed — "local" is the default) to confirm
retrieval genuinely abstains on a real uncovered question. The full-pipeline
version of this checkpoint (a live Claude call reacting to that abstention)
lives in tests/test_text_cli.py, alongside the project's other full-loop
wiring tests.
"""

from __future__ import annotations

import hashlib

import chromadb
import pytest

from agent.tools import policy_rag

FAKE_DOCS = {
    "returns.md": "# Returns\n\nYou can return an item within thirty days of delivery for a refund.",
    "shipping.md": "# Shipping\n\nStandard delivery takes five business days and ships worldwide.",
}


def _fake_embed(text: str, dims: int = 16) -> list[float]:
    """Deterministic bag-of-words hash embedding for tests: text sharing
    vocabulary lands close together, text with no overlap lands ~orthogonal
    (cosine distance ~1). Not semantically meaningful, but enough to
    exercise the retrieval/threshold plumbing without any real model.
    """
    vec = [0.0] * dims
    for word in text.lower().split():
        idx = int(hashlib.sha256(word.encode()).hexdigest(), 16) % dims
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


class FakeEmbeddingBackend:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_fake_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _fake_embed(text)


@pytest.fixture
def fake_policies(tmp_path, monkeypatch):
    """A small synthetic corpus, isolated from the real data/policies/."""
    for filename, content in FAKE_DOCS.items():
        (tmp_path / filename).write_text(content)
    monkeypatch.setattr(policy_rag, "POLICIES_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def fake_collection():
    """An ephemeral, in-memory Chroma collection — no disk, no shared state
    with the real data/chroma_db/.
    """
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name="test_policies", metadata={"hnsw:space": "cosine"})


def test_chunk_document_prefixes_title_onto_every_paragraph():
    text = "# Returns\n\nFirst paragraph.\n\nSecond paragraph."

    chunks = policy_rag._chunk_document("returns.md", text)

    assert len(chunks) == 2
    assert all(c["title"] == "Returns" for c in chunks)
    assert chunks[0]["text"] == "Returns\n\nFirst paragraph."
    assert chunks[1]["text"] == "Returns\n\nSecond paragraph."
    assert chunks[0]["id"] == "returns.md::0"


def test_load_policy_documents_finds_the_real_fixture_files():
    docs = policy_rag._load_policy_documents()

    assert 10 <= len(docs) <= 20, "PROJECT_PLAN.md calls for 10-20 fake policy documents"
    assert all(text.strip() for _name, text in docs)


def test_ingest_policies_populates_the_collection(fake_policies, fake_collection):
    count = policy_rag.ingest_policies(collection=fake_collection, backend=FakeEmbeddingBackend())

    assert count == 2  # one paragraph each in FAKE_DOCS
    assert fake_collection.count() == 2


def test_ingest_policies_is_idempotent_on_rerun(fake_policies, fake_collection):
    policy_rag.ingest_policies(collection=fake_collection, backend=FakeEmbeddingBackend())
    policy_rag.ingest_policies(collection=fake_collection, backend=FakeEmbeddingBackend())

    assert fake_collection.count() == 2  # rebuilt from scratch, not doubled


def test_search_policy_finds_the_relevant_chunk(fake_policies, fake_collection):
    policy_rag.ingest_policies(collection=fake_collection, backend=FakeEmbeddingBackend())

    result = policy_rag.search_policy(
        "How many days can I return an item?", collection=fake_collection, backend=FakeEmbeddingBackend()
    )

    assert result["found"] is True
    assert result["results"][0]["source"] == "returns.md"


def test_search_policy_abstains_for_a_question_with_no_matching_vocabulary(fake_policies, fake_collection):
    policy_rag.ingest_policies(collection=fake_collection, backend=FakeEmbeddingBackend())

    result = policy_rag.search_policy(
        "xylophone quokka umbrella zephyr", collection=fake_collection, backend=FakeEmbeddingBackend()
    )

    assert result["found"] is False
    assert "message" in result


def test_real_policy_corpus_correctly_abstains_on_an_uncovered_question():
    """Uses the real 16-doc corpus and the real local embedding backend
    (no API key needed — "local" is the default). Confirmed empirically:
    see RELEVANCE_THRESHOLD's comment for the distance data this was tuned
    against. First run downloads the small local embedding model (~80MB);
    cached after that.
    """
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="live_policies", metadata={"hnsw:space": "cosine"})
    backend = policy_rag.get_embedding_backend()
    policy_rag.ingest_policies(collection=collection, backend=backend)

    result = policy_rag.search_policy(
        "Do you offer price matching with other stores?", collection=collection, backend=backend
    )

    assert result["found"] is False
