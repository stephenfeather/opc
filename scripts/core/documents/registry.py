"""Document-collection registry: the source of truth for tracked folders.

The registry is a YAML file (default ~/.claude/opc/document_collections.yaml,
overridable via OPC_DOC_REGISTRY). Each entry declares a folder, its retrieval
scope, the file extensions to ingest, and whether OCR is requested (OCR is a
phase-2 feature; the flag is stored but born-digital v1 ignores it).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

VALID_SCOPES = ("global", "restricted")
DEFAULT_REGISTRY_PATH = Path.home() / ".claude" / "opc" / "document_collections.yaml"


class RegistryError(Exception):
    """Raised when the registry file or a collection entry is invalid."""


@dataclass(frozen=True)
class Collection:
    """A tracked document folder."""

    name: str
    path: str
    scope: str
    extensions: list[str] = field(default_factory=list)
    ocr: bool = False


def registry_path() -> Path:
    """Resolve the registry file path from OPC_DOC_REGISTRY or the default."""
    override = os.getenv("OPC_DOC_REGISTRY")
    return Path(override).expanduser() if override else DEFAULT_REGISTRY_PATH


def validate_collection(collection: Collection) -> Collection:
    """Validate a collection, returning it unchanged on success.

    Raises:
        RegistryError: if any field is invalid.
    """
    if not collection.name.strip():
        raise RegistryError("collection name must not be empty")
    if collection.scope not in VALID_SCOPES:
        raise RegistryError(f"scope must be one of {VALID_SCOPES}, got {collection.scope!r}")
    if not collection.extensions:
        raise RegistryError("collection must declare at least one extension")
    if not collection.path.strip():
        raise RegistryError("collection path must not be empty")
    return collection


def load_registry(path: Path | None = None) -> list[Collection]:
    """Load all collections from the registry file.

    Returns an empty list if the file does not exist.

    Raises:
        RegistryError: if the file exists but is malformed.
    """
    target = path or registry_path()
    if not target.exists():
        return []
    try:
        raw = yaml.safe_load(target.read_text()) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"registry file is not valid YAML: {exc}") from exc
    entries = raw.get("collections", [])
    collections = []
    for entry in entries:
        # The registry is hand-edited YAML; a missing key or a non-mapping entry
        # must surface as RegistryError (which callers handle), not an unhandled
        # KeyError/TypeError traceback out of the CLI.
        try:
            col = Collection(
                name=entry["name"],
                path=entry["path"],
                scope=entry["scope"],
                extensions=list(entry.get("extensions", [])),
                ocr=bool(entry.get("ocr", False)),
            )
        except (KeyError, TypeError) as exc:
            raise RegistryError(f"malformed registry entry {entry!r}: {exc}") from exc
        collections.append(validate_collection(col))
    return collections


def append_collection(path: Path | None, collection: Collection) -> None:
    """Validate and append a collection to the registry file.

    Creates the file (and parent directory) if it does not exist.

    Raises:
        RegistryError: if the collection is invalid or its name already exists.
    """
    target = path or registry_path()
    validate_collection(collection)
    existing = load_registry(target)
    if any(c.name == collection.name for c in existing):
        raise RegistryError(f"collection {collection.name!r} already exists")
    existing.append(collection)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"collections": [asdict(c) for c in existing]}
    target.write_text(yaml.safe_dump(payload, sort_keys=False))
