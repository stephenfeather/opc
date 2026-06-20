"""Tests for the document-collection registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.core.documents.registry import (
    Collection,
    RegistryError,
    append_collection,
    load_registry,
    validate_collection,
)


def test_load_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_registry(tmp_path / "nonexistent.yaml")
    assert result == []


def test_append_then_load_roundtrip(tmp_path: Path) -> None:
    reg = tmp_path / "reg.yaml"
    col = Collection(
        name="caleb-records",
        path="~/Documents/Feather, Caleb",
        scope="restricted",
        extensions=[".pdf", ".docx"],
        ocr=False,
    )
    append_collection(reg, col)
    loaded = load_registry(reg)
    assert loaded == [col]


def test_append_rejects_duplicate_name(tmp_path: Path) -> None:
    reg = tmp_path / "reg.yaml"
    col = Collection(name="x", path="/tmp/x", scope="global", extensions=[".txt"], ocr=False)
    append_collection(reg, col)
    with pytest.raises(RegistryError, match="already exists"):
        append_collection(reg, col)


def test_validate_rejects_bad_scope() -> None:
    with pytest.raises(RegistryError, match="scope"):
        validate_collection(
            Collection(name="x", path="/tmp/x", scope="public", extensions=[".txt"], ocr=False)
        )


def test_validate_rejects_empty_name() -> None:
    with pytest.raises(RegistryError, match="name"):
        validate_collection(
            Collection(name="", path="/tmp/x", scope="global", extensions=[".txt"], ocr=False)
        )


def test_validate_rejects_empty_extensions() -> None:
    with pytest.raises(RegistryError, match="extension"):
        validate_collection(
            Collection(name="x", path="/tmp/x", scope="global", extensions=[], ocr=False)
        )
