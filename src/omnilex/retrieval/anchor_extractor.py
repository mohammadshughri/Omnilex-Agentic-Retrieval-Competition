"""Extract citation anchors from query text."""

from __future__ import annotations

import re

from omnilex.citations.normalizer import CitationNormalizer


_NORMALIZER = None


def _get_normalizer() -> CitationNormalizer:
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = CitationNormalizer()
    return _NORMALIZER


# Match "Art. X [Abs. Y] [lit. z] BOOK" where BOOK is 2-10 mixed-case letters
_ART_PATTERN = re.compile(
    r"(?:Art\.?|Artikel)\s*\d+[a-z]?"
    r"(?:\s+(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*\d+[a-z]?)?"
    r"(?:\s+(?:lit\.?|Ziff\.?|Nr\.?)\s*[a-z0-9]+)?"
    r"\s+([A-Z][A-Za-z]{1,10})",
    re.UNICODE,
)

# Match "BGE VOL SECTION PAGE"
_BGE_PATTERN = re.compile(
    r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+",
    re.IGNORECASE,
)


def extract_citation_anchors(query: str) -> list[str]:
    """Extract and normalize citation anchors from a query string.

    Finds explicit Art./BGE patterns, normalizes via CitationNormalizer,
    and returns deduplicated canonical IDs.
    """
    normalizer = _get_normalizer()
    raw_matches: list[str] = []

    for m in _ART_PATTERN.finditer(query):
        raw_matches.append(m.group(0).strip())

    for m in _BGE_PATTERN.finditer(query):
        raw_matches.append(m.group(0).strip())

    return normalizer.canonicalize_list(raw_matches)
