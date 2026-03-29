"""
pdf_extractor.py
----------------
Extracts and structurally chunks the Indian Constitution PDF.

Chunk hierarchy:
  PREAMBLE  ─ single chunk
  PART X    ─ label carried forward as metadata
  Article N ─ primary chunk boundary
  Sub-clauses (a), (b), (c)… are kept inside the article chunk
  Footnotes / amendment notes are stripped
  Repeated page headers/footers are stripped
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches page-header noise like:
#   "THE CONSTITUTION OF INDIA"
#   "(Part I.—Union and its territory)"
_PAGE_HEADER_RE = re.compile(
    r"^(THE CONSTITUTION OF INDIA|"
    r"\(Part\s+[^)]+\)|"
    r"\d+\s*$)",  # lone page numbers
    re.IGNORECASE,
)

# Footnote lines: start with a digit followed by a period and "Subs.", "Ins.", "Omitted", etc.
_FOOTNOTE_LINE_RE = re.compile(
    r"^\s*\d+[\.\,]\s+(Subs\.|Ins\.|Omitted|Rep\.|Added|Renumbered)",
    re.IGNORECASE,
)

# PART heading: "PART I" / "PART XIV" / "PART XIVA" etc.
_PART_RE = re.compile(r"^PART\s+([IVXLCDM]+[A-Z]?)\s*$", re.IGNORECASE)

# Article heading: "1. Name and territory…" or "21A. Right to education"
# Must start at beginning of line (after strip)
_ARTICLE_START_RE = re.compile(
    r"^(\d+[A-Z]?)\.\s+[A-Z]",  # e.g. "3. Formation…" or "21A. Right…"
)

# PREAMBLE marker
_PREAMBLE_RE = re.compile(r"^PREAMBLE\s*$", re.IGNORECASE)

# Amendment superscripts embedded inline like: 1[ … ] — kept but harmless
_SUPERSCRIPT_INLINE_RE = re.compile(r"\d+\[")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LegalChunk:
    part: str           # e.g. "PART I", "PREAMBLE"
    article: str        # e.g. "Article 1", "PREAMBLE"
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _extract_raw_text(pdf_path: str | Path) -> str:
    """Extract all text from PDF preserving newlines."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


def _clean_lines(raw_text: str) -> list[str]:
    """
    Remove page-header noise and footnote lines.
    Returns a list of cleaned lines (may still be empty/whitespace).
    """
    cleaned = []
    lines = raw_text.splitlines()
    # State: are we inside a footnote block?
    in_footnote = False

    for line in lines:
        stripped = line.strip()

        # Skip empty
        if not stripped:
            in_footnote = False          # reset footnote block on blank line
            cleaned.append("")
            continue

        # Drop page-header noise
        if _PAGE_HEADER_RE.match(stripped):
            continue

        # Detect footnote start (e.g. "1. Subs. by the Constitution…")
        if _FOOTNOTE_LINE_RE.match(stripped):
            in_footnote = True
            continue

        # Continuation of a footnote block: indented lines after a footnote
        if in_footnote and line.startswith("   "):
            continue

        in_footnote = False
        cleaned.append(stripped)

    return cleaned


def _build_chunks(lines: list[str]) -> Generator[LegalChunk, None, None]:
    """
    Walk cleaned lines and emit LegalChunk objects whenever an article
    or PREAMBLE boundary is crossed.
    """
    current_part = "PREAMBLE"
    current_article = "PREAMBLE"
    buffer: list[str] = []

    def flush() -> LegalChunk | None:
        text = " ".join(t for t in buffer if t).strip()
        if text:
            return LegalChunk(
                part=current_part,
                article=current_article,
                text=text,
            )
        return None

    for line in lines:
        stripped = line.strip()

        # ── PART heading ────────────────────────────────────────────
        part_match = _PART_RE.match(stripped)
        if part_match:
            chunk = flush()
            if chunk:
                yield chunk
            buffer = []
            current_part = f"PART {part_match.group(1).upper()}"
            # do NOT reset article yet; wait for next article heading
            continue

        # ── PREAMBLE marker ─────────────────────────────────────────
        if _PREAMBLE_RE.match(stripped):
            chunk = flush()
            if chunk:
                yield chunk
            buffer = []
            current_part = "PREAMBLE"
            current_article = "PREAMBLE"
            continue

        # ── Article heading ─────────────────────────────────────────
        article_match = _ARTICLE_START_RE.match(stripped)
        if article_match:
            chunk = flush()
            if chunk:
                yield chunk
            buffer = [stripped]
            current_article = f"Article {article_match.group(1)}"
            continue

        # ── Regular content line ─────────────────────────────────────
        buffer.append(stripped)

    # Flush remainder
    chunk = flush()
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_chunks(pdf_path: str | Path) -> list[LegalChunk]:
    """
    Full pipeline: PDF → cleaned lines → structured LegalChunk list.
    Each chunk represents one Article (or PREAMBLE) with its PART metadata.
    """
    raw = _extract_raw_text(pdf_path)
    lines = _clean_lines(raw)
    chunks = list(_build_chunks(lines))
    return chunks


def extract_sample(pdf_path: str | Path, max_words: int = 1000) -> str:
    """Utility: return the first max_words words for debugging."""
    raw = _extract_raw_text(pdf_path)
    words = raw.split()
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Quick sanity-check (run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset_main.pdf"
    chunks = extract_chunks(path)
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks[:5]:
        preview = c.text[:200].replace("\n", " ")
        print(f"[{c.part}] [{c.article}]")
        print(f"  {preview}…")
        print()
