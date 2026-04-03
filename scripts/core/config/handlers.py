"""Configuration I/O handlers — file discovery, TOML loading, env var reads.

All side effects (disk reads, env var access, caching) live here.
Pure construction logic is in core.py.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from scripts.core.config.core import build_config, merge_raw
from scripts.core.config.models import OPCConfig


# ---------------------------------------------------------------------------
# Env var → config mapping
# ---------------------------------------------------------------------------

_ENV_OVERRIDES: list[tuple[str, str, str, type]] = [
    # (env_var, section, key, cast_type)
    ("PATTERN_DETECTION_INTERVAL_HOURS", "daemon", "pattern_detection_interval_hours", int),
    ("OLLAMA_EMBED_MODEL", "embedding", "ollama_model", str),
    ("OLLAMA_HOST", "embedding", "ollama_host", str),
    ("AGENTICA_MAX_POOL_SIZE", "database", "max_pool_size", int),
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_config_paths() -> list[Path]:
    """Return config file paths in precedence order (first wins)."""
    paths: list[Path] = []

    env_path = os.environ.get("OPC_CONFIG")
    if env_path:
        paths.append(Path(env_path))

    # Walk up from this file to find opc.toml at project root
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "opc.toml"
        if candidate.is_file():
            paths.append(candidate)
            break
        current = current.parent

    paths.append(Path.home() / ".config" / "opc" / "config.toml")

    return paths


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------

def load_config_file(paths: list[Path]) -> dict[str, Any]:
    """Load the first existing TOML file from the path list."""
    for path in paths:
        if path.is_file():
            with open(path, "rb") as f:
                return tomllib.load(f)
    return {}


# ---------------------------------------------------------------------------
# Env var reading
# ---------------------------------------------------------------------------

def read_env_overrides() -> dict[str, Any]:
    """Read known env vars and return as a nested dict matching TOML structure.

    Invalid values (e.g. non-numeric for int fields) are logged and skipped
    rather than crashing config loading.
    """
    import logging
    _logger = logging.getLogger(__name__)

    result: dict[str, Any] = {}
    for env_var, section, key, cast in _ENV_OVERRIDES:
        value = os.environ.get(env_var)
        if value is not None:
            try:
                result.setdefault(section, {})[key] = cast(value)
            except (ValueError, TypeError):
                _logger.warning(
                    "Config: env var %s=%r cannot be cast to %s — ignored",
                    env_var, value, cast.__name__,
                )
    return result


# ---------------------------------------------------------------------------
# Cached loader (the one side-effectful piece)
# ---------------------------------------------------------------------------

_cached_config: OPCConfig | None = None


def get_config(*, reload: bool = False) -> OPCConfig:
    """Load and return the OPC configuration (cached after first call).

    Precedence: env vars > first found TOML file > built-in defaults.
    """
    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config

    file_raw = load_config_file(discover_config_paths())
    env_raw = read_env_overrides()
    merged = merge_raw(file_raw, env_raw)
    _cached_config = build_config(merged)
    return _cached_config


def reset_config() -> None:
    """Clear the cached config. Used in tests."""
    global _cached_config
    _cached_config = None
