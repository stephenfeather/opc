"""Pure configuration construction — no I/O, no global state.

All functions are pure: same input produces same output, no side effects.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

from scripts.core.config.models import (
    ArchivalConfig,
    DatabaseConfig,
    DaemonConfig,
    DedupConfig,
    EmbeddingConfig,
    OPCConfig,
    PatternsConfig,
    QueryExpansionConfig,
    RerankerConfig,
    RecallConfig,
)

logger = logging.getLogger(__name__)

_SECTION_MAP: dict[str, type] = {
    "dedup": DedupConfig,
    "daemon": DaemonConfig,
    "reranker": RerankerConfig,
    "patterns": PatternsConfig,
    "recall": RecallConfig,
    "embedding": EmbeddingConfig,
    "query_expansion": QueryExpansionConfig,
    "archival": ArchivalConfig,
    "database": DatabaseConfig,
}


class ConfigValidationError(ValueError):
    """Raised when config values fail type or range validation."""


def _validate_type(section: str, key: str, value: Any, expected_type: type) -> Any:
    """Validate and coerce a config value to the expected type.

    Returns the validated value, or raises ConfigValidationError.
    """
    # TOML integers are valid for float fields
    if expected_type is float and isinstance(value, int):
        return float(value)
    if isinstance(value, expected_type):
        return value
    raise ConfigValidationError(
        f"[{section}] {key}: expected {expected_type.__name__}, got {type(value).__name__} ({value!r})"
    )


def build_section(cls: type, raw: dict[str, Any], *, section_name: str = "") -> Any:
    """Build a config section dataclass from a raw dict.

    Warns on unknown keys and validates types against the dataclass field types.
    """
    field_map = {f.name: f for f in fields(cls)}
    errors: list[str] = []

    # Warn on unknown keys
    unknown = set(raw.keys()) - set(field_map.keys())
    for key in sorted(unknown):
        logger.warning("Config: unknown key [%s] %s — ignored", section_name or cls.__name__, key)

    # Validate and collect known keys
    validated: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in field_map:
            continue
        expected = field_map[key].type
        # Resolve string type annotations
        type_map = {"float": float, "int": int, "str": str, "bool": bool}
        resolved_type = type_map.get(expected, expected) if isinstance(expected, str) else expected
        try:
            validated[key] = _validate_type(section_name, key, value, resolved_type)
        except ConfigValidationError as e:
            errors.append(str(e))

    if errors:
        raise ConfigValidationError("; ".join(errors))

    return cls(**validated)


def merge_raw(
    file_raw: dict[str, Any],
    env_raw: dict[str, Any],
) -> dict[str, Any]:
    """Merge file config with env overrides. Env wins per-key. Does not mutate inputs."""
    merged: dict[str, Any] = {}

    all_sections = set(file_raw.keys()) | set(env_raw.keys())
    for section in all_sections:
        file_section = file_raw.get(section, {})
        env_section = env_raw.get(section, {})
        merged[section] = {**file_section, **env_section}

    return merged


def build_config(raw: dict[str, Any]) -> OPCConfig:
    """Build a complete OPCConfig from a merged raw dict.

    Warns on unknown top-level sections.
    """
    unknown_sections = set(raw.keys()) - set(_SECTION_MAP.keys())
    for name in sorted(unknown_sections):
        logger.warning("Config: unknown section [%s] — ignored", name)

    sections = {}
    for name, cls in _SECTION_MAP.items():
        section_raw = raw.get(name, {})
        sections[name] = build_section(cls, section_raw, section_name=name)
    return OPCConfig(**sections)
