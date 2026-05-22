"""Tests for citation-anchor extraction from query text."""

from omnilex.retrieval.anchor_extractor import extract_citation_anchors


class TestExtractCitationAnchors:
    def test_extracts_art_with_book(self):
        query = "Does Art. 221 Abs. 1 StPO apply to pre-trial detention?"
        anchors = extract_citation_anchors(query)
        assert any("Art. 221" in a and "StPO" in a for a in anchors)

    def test_extracts_bge(self):
        query = "According to BGE 137 IV 122, what are the requirements?"
        anchors = extract_citation_anchors(query)
        assert any("BGE 137 IV 122" in a for a in anchors)

    def test_extracts_multiple(self):
        query = (
            "Under Art. 83 SVG and Art. 59 Abs. 1 SVG, "
            "is the driver liable?"
        )
        anchors = extract_citation_anchors(query)
        assert len(anchors) >= 2

    def test_no_citations_returns_empty(self):
        query = "What are the requirements for a valid contract?"
        anchors = extract_citation_anchors(query)
        assert anchors == []

    def test_strips_lit_and_ziff(self):
        query = "Art. 221 Abs. 1 lit. b StPO mentions collusion risk"
        anchors = extract_citation_anchors(query)
        for a in anchors:
            assert "lit." not in a

    def test_deduplicates(self):
        query = "Art. 1 ZGB states that... Art. 1 ZGB also applies to..."
        anchors = extract_citation_anchors(query)
        assert len(anchors) == 1

    def test_returns_canonical_form(self):
        query = "Artikel 41 Abs. 1 OR governs tort liability"
        anchors = extract_citation_anchors(query)
        assert "Art. 41 Abs. 1 OR" in anchors
