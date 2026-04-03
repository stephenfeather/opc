"""Pure configuration construction — no I/O, no global state.

All functions are pure: same input produces same output, no side effects.
"""

from __future__ import annotations

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


def build_section(cls: type, raw: dict[str, Any]) -> Any:
    """Build a config section dataclass from a raw dict, ignoring unknown keys."""
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return cls(**filtered)


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
    """Build a complete OPCConfig from a merged raw dict."""
    sections = {}
    for name, cls in _SECTION_MAP.items():
        section_raw = raw.get(name, {})
        sections[name] = build_section(cls, section_raw)
    return OPCConfig(**sections)
