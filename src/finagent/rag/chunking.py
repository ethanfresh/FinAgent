"""Chunk SEC filing text into retrievable passages.

Filings like 10-Ks and 10-Qs are organized into numbered "Item" sections
(e.g. "Item 1A. Risk Factors"). When those section headers are present in the
extracted text, chunking respects them so a chunk never straddles a section
boundary — a retrieved passage always carries clear, correct section context.
Falls back to a fixed-size sliding window with overlap when no recognizable
section headers are found (e.g. exhibits, or non-filing text).

Pure text logic, no network or model dependencies, so it's testable and
importable without the `rag` extra installed.
"""

import re

_ITEM_HEADER_RE = re.compile(r"(?m)^\s*(item\s+\d+[a-z]?\.?\s*[-—:]?\s*[^\n]{0,120})\s*$", re.IGNORECASE)


def split_into_sections(text: str) -> list[dict]:
    """Split filing text into {"section": title | None, "text": body} pieces
    using "Item N. ..." headers. Falls back to one untitled section when fewer
    than two headers are found — not enough structure to trust as real
    section boundaries."""
    matches = list(_ITEM_HEADER_RE.finditer(text))
    if len(matches) < 2:
        return [{"section": None, "text": text}]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"section": match.group(1).strip(), "text": body})
    return sections


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Fixed-size sliding-window chunking by characters, breaking near a
    whitespace boundary so chunks don't split mid-word."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # the end boundary above is snapped to a space, but next_start (from
        # subtracting overlap) isn't — snap it forward too, or the overlap
        # region can begin mid-word.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def chunk_filing(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[dict]:
    """Chunk a filing's full text into retrievable passages, each tagged with
    its section title when the filing's Item structure could be detected."""
    chunks = []
    for section in split_into_sections(text):
        for piece in chunk_text(section["text"], chunk_size, overlap):
            chunks.append({"section": section["section"], "text": piece})
    return chunks
