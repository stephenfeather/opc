"""Tests for born-digital text extraction."""

from __future__ import annotations

from pathlib import Path

from scripts.core.documents.extract import ExtractionResult, extract_text


def test_extract_txt(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("hello world\nsecond line")
    result = extract_text(f)
    assert result.status == "extracted"
    assert len(result.pages) == 1
    assert result.pages[0].text == "hello world\nsecond line"
    assert result.pages[0].page_number == 1


def test_extract_csv(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("name,amount\nLisa,100\nCaleb,200")
    result = extract_text(f)
    assert result.status == "extracted"
    assert "Lisa" in result.pages[0].text
    assert "200" in result.pages[0].text


def test_extract_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "image.heic"
    f.write_bytes(b"\x00\x01")
    result = extract_text(f)
    assert result.status == "skipped_unsupported"
    assert result.pages == []


def test_extract_empty_pdf_text_layer_marks_needs_ocr(tmp_path: Path) -> None:
    # A PDF whose pages yield no text is a scan -> phase-2 OCR territory.
    f = tmp_path / "scan.pdf"
    f.write_bytes(_minimal_textless_pdf())
    result = extract_text(f)
    assert result.status == "skipped_needs_ocr"
    assert result.pages == []


def test_extract_md(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Heading\nbody text")
    result = extract_text(f)
    assert result.status == "extracted"
    assert "body text" in result.pages[0].text


def test_extract_html(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text("<html><body><h1>Hi</h1><p>hello world</p></body></html>")
    result = extract_text(f)
    assert result.status == "extracted"
    assert "Hi" in result.pages[0].text
    assert "hello world" in result.pages[0].text
    assert "<p>" not in result.pages[0].text
    assert "<body>" not in result.pages[0].text


def test_extract_html_skips_script(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text("<html><body><script>alert(1)</script><p>visible text</p></body></html>")
    result = extract_text(f)
    assert result.status == "extracted"
    assert "alert" not in result.pages[0].text


def test_extract_xml(tmp_path: Path) -> None:
    f = tmp_path / "data.xml"
    f.write_text("<root><a>alpha</a><b>beta</b></root>")
    result = extract_text(f)
    assert result.status == "extracted"
    assert "alpha" in result.pages[0].text
    assert "beta" in result.pages[0].text
    assert "<a>" not in result.pages[0].text


def test_extract_result_dataclass_shape() -> None:
    result = ExtractionResult(pages=[], status="error", error="boom", page_count=0)
    assert result.status == "error"
    assert result.error == "boom"


def _minimal_textless_pdf() -> bytes:
    """A 1-page PDF with no text content stream."""
    return (
        b"%PDF-1.1\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n170\n%%EOF"
    )
