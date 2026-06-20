"""Born-digital text extraction (v1 — no OCR).

Dispatches by file extension. PDFs are read via their text layer only; a PDF
that yields no text is a scanned image and is reported as 'skipped_needs_ocr'
for a future phase-2 OCR pipeline. Pure module: no DB, no network.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".md",
    ".html",
    ".htm",
    ".xml",
)


@dataclass(frozen=True)
class ExtractedPage:
    """One unit of extracted text. page_number is 1-based; 1 for non-paged formats."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of extracting one file.

    status:
        'extracted'           -- pages contains text
        'skipped_unsupported' -- extension not in SUPPORTED_EXTENSIONS
        'skipped_needs_ocr'   -- supported type but no extractable text (a scan)
        'error'               -- extraction raised; see error
    """

    pages: list[ExtractedPage]
    status: str
    error: str | None
    page_count: int


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping <script>/<style> content."""

    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        joined = " ".join(self._chunks)
        return re.sub(r"\n{2,}", "\n", joined).strip()


def _extract_txt(path: Path) -> list[ExtractedPage]:
    return [ExtractedPage(page_number=1, text=path.read_text(errors="replace"))]


def _extract_csv(path: Path) -> list[ExtractedPage]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            rows.append(" ".join(row))
    return [ExtractedPage(page_number=1, text="\n".join(rows))]


def _extract_docx(path: Path) -> list[ExtractedPage]:
    import docx  # python-docx

    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [ExtractedPage(page_number=1, text=text)] if text.strip() else []


def _extract_pdf(path: Path) -> list[ExtractedPage]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(ExtractedPage(page_number=index, text=text))
    return pages


def _extract_html(path: Path) -> list[ExtractedPage]:
    parser = _HTMLTextExtractor()
    parser.feed(path.read_text(errors="replace"))
    text = parser.get_text()
    return [ExtractedPage(page_number=1, text=text)] if text.strip() else []


def _extract_xml(path: Path) -> list[ExtractedPage]:
    root = ElementTree.fromstring(path.read_text(errors="replace"))
    text = " ".join(t.strip() for t in root.itertext() if t.strip()).strip()
    return [ExtractedPage(page_number=1, text=text)] if text.strip() else []


def extract_text(path: Path) -> ExtractionResult:
    """Extract text from a single born-digital file.

    Never raises — failures are reported via ExtractionResult.status == 'error'.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ExtractionResult(pages=[], status="skipped_unsupported", error=None, page_count=0)

    extractors = {
        ".txt": _extract_txt,
        ".md": _extract_txt,
        ".csv": _extract_csv,
        ".docx": _extract_docx,
        ".pdf": _extract_pdf,
        ".html": _extract_html,
        ".htm": _extract_html,
        ".xml": _extract_xml,
    }
    try:
        pages = extractors[ext](path)
    except Exception as exc:  # noqa: BLE001 - extraction must never crash ingest
        return ExtractionResult(pages=[], status="error", error=str(exc), page_count=0)

    if not pages or not any(p.text.strip() for p in pages):
        # Supported type but nothing extractable -> a scan; defer to phase-2 OCR.
        return ExtractionResult(pages=[], status="skipped_needs_ocr", error=None, page_count=0)

    return ExtractionResult(pages=pages, status="extracted", error=None, page_count=len(pages))
