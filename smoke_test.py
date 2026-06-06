"""Smoke test — exercises Session 1, Session 2, and Session 3 retrieval modules."""

import pathlib
import tempfile

import numpy as np

from omnilex.retrieval import (
    Candidate,
    extract_citation_anchors,
    generate_submission,
    rrf_fuse,
)
from omnilex.retrieval.dense_index import DenseIndex, DenseIndexBuilder
from omnilex.retrieval.models import EmbeddingModel

# ── Session 1 ────────────────────────────────────────────────────────────────

# 1. Anchor extraction
query = "Does Art. 221 Abs. 1 StPO apply? See also BGE 137 IV 122."
anchors = extract_citation_anchors(query)
print("Anchors:", anchors)
assert len(anchors) >= 2, f"Expected >=2 anchors, got {anchors}"

# 2. RRF fusion — Art. 1 ZGB appears in both channels so should be boosted
dense = [
    Candidate("Art. 221 StPO", 0.9, "dense"),
    Candidate("Art. 1 ZGB", 0.7, "dense"),
]
sparse = [
    Candidate("Art. 1 ZGB", 0.8, "sparse"),
    Candidate("BGE 137 IV 122", 0.6, "sparse"),
]
fused = rrf_fuse([dense, sparse])
scores = {c.citation_raw: c.rrf_score for c in fused}
assert scores["Art. 1 ZGB"] > scores["BGE 137 IV 122"], "RRF boost expected for shared candidate"
print("Fused top-3:", [(c.citation_raw, round(c.rrf_score, 4)) for c in fused[:3]])

# 3. Submission CSV
results = {"q1": fused}
path = pathlib.Path(tempfile.mktemp(suffix=".csv"))
generate_submission(results, path, top_k=3)
csv_text = path.read_text()
assert "query_id" in csv_text and "predicted_citations" in csv_text
print("CSV:\n" + csv_text)
path.unlink()

# ── Session 2 ────────────────────────────────────────────────────────────────

class StubEmbedder(EmbeddingModel):
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = np.zeros((len(texts), 4), dtype=np.float32)
        for i in range(len(texts)):
            vecs[i, i % 4] = 1.0
        return vecs

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        vecs = np.zeros((len(queries), 4), dtype=np.float32)
        for i in range(len(queries)):
            vecs[i, i % 4] = 1.0
        return vecs


docs = [
    {"citation": "Art. 221 StPO", "text": "Untersuchungshaft wegen Kollusionsgefahr", "title": "Haft"},
    {"citation": "Art. 1 ZGB",    "text": "Das Gesetz findet auf alle Rechtsfragen Anwendung", "title": "Anwendung"},
    {"citation": "BGE 137 IV 122", "text": "Haftgruende und Verhaeltnismaessigkeit"},
]

with tempfile.TemporaryDirectory() as tmpdir:
    out = pathlib.Path(tmpdir)
    builder = DenseIndexBuilder(StubEmbedder())
    builder.build_from_records(
        records=docs,
        citation_field="citation",
        text_field="text",
        title_field="title",
        output_dir=out,
    )
    assert (out / "faiss.index").exists()
    assert (out / "metadata.jsonl").exists()
    assert (out / "dense.npy").exists()

    index = DenseIndex.load(out)
    assert len(index.metadata) == 3

    # Query vector matching first doc (Art. 221 StPO)
    query_vec = np.zeros((1, 4), dtype=np.float32)
    query_vec[0, 0] = 1.0
    results = index.search(query_vec, top_k=2)
    assert results[0]["citation_raw"] == "Art. 221 StPO", f"Top result: {results[0]['citation_raw']}"
    print("Dense search top-2:", [(r["citation_raw"], round(r["score"], 4)) for r in results])

# ── Session 3 ────────────────────────────────────────────────────────────────

from omnilex.retrieval.dense_retriever import DenseRetriever

with tempfile.TemporaryDirectory() as tmpdir:
    out = pathlib.Path(tmpdir)
    builder = DenseIndexBuilder(StubEmbedder())
    builder.build_from_records(
        records=docs,
        citation_field="citation",
        text_field="text",
        title_field="title",
        output_dir=out,
    )
    index = DenseIndex.load(out)

    # 4. DenseRetriever — query with an explicit anchor in the text
    retriever = DenseRetriever(embedder=StubEmbedder(), laws_index=index, courts_index=None)
    candidates = retriever.retrieve("Art. 221 Abs. 1 StPO applies here", top_k=5)
    assert len(candidates) > 0, "Expected at least one candidate"
    assert all(hasattr(c, "citation_raw") for c in candidates), "All candidates must have citation_raw"
    citations = [c.citation_raw for c in candidates]
    assert any("Art. 221" in c for c in citations), f"Anchor Art. 221 not found in {citations}"
    sources = {c.source for c in candidates}
    assert "anchor" in sources, "Anchor channel must be present when query contains explicit citation"
    print("DenseRetriever candidates:", [(c.citation_raw, c.source, round(c.rrf_score, 4)) for c in candidates])

    # 5. End-to-end: retrieve → submission CSV
    results = {"q1": candidates}
    path = pathlib.Path(tempfile.mktemp(suffix=".csv"))
    generate_submission(results, path, top_k=3)
    csv_text = path.read_text()
    assert "Art. 221" in csv_text, "Expected Art. 221 in submission output"
    print("Session 3 submission CSV:\n" + csv_text)
    path.unlink()

print("\nAll smoke tests passed.")
