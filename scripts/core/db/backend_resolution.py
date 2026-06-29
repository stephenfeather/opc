"""Unified backend / connection-URL resolution (issue #71).

Single source of truth for how every memory-pipeline consumer decides:

  1. **Which PostgreSQL connection URL to use** — :func:`resolve_url`.
  2. **Which storage backend to use** (``"sqlite"`` or ``"postgres"``) —
     :func:`resolve_backend`.

Before this module, ``store_learning``, ``recall_learnings``,
``memory_daemon``, ``confidence_calibrator``, and ``postgres_pool`` each
implemented their own resolution with subtly different rules. Two of them
omitted ``OPC_POSTGRES_URL`` and they disagreed on whether
``AGENTICA_MEMORY_BACKEND`` or URL presence took precedence. Mixing those
rules risked split-brain storage (writing to one backend, reading from
another). All consumers now delegate here.

Precedence (documented once, here):

* **URL** (:func:`resolve_url`):
  ``CONTINUOUS_CLAUDE_DB_URL`` (canonical) > ``DATABASE_URL`` (compat) >
  ``OPC_POSTGRES_URL`` (legacy, hooks). Empty strings are ignored. Returns
  ``None`` when none are set.

* **Backend** (:func:`resolve_backend`):
  1. ``AGENTICA_MEMORY_BACKEND`` when it explicitly names a valid backend
     (``"sqlite"``/``"postgres"``, case-insensitive) — an operator override
     always wins. A *non-empty* override that is invalid, or ``postgres`` with
     no connection URL, raises ``ValueError`` (fail-fast, issue #214) rather
     than silently falling through.
  2. Otherwise, the presence of *any* connection URL implies ``"postgres"``.
  3. Otherwise the supplied ``default`` (``"sqlite"`` unless overridden).

Both functions are pure: they take an explicit env mapping and read nothing
from ``os.environ``. The thin :func:`get_connection_url` /
:func:`get_active_backend` wrappers bind them to the live environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# Connection-URL env vars in priority order (canonical, compat, legacy).
URL_VARS: tuple[str, ...] = (
    "CONTINUOUS_CLAUDE_DB_URL",
    "DATABASE_URL",
    "OPC_POSTGRES_URL",
)

# Backends an explicit AGENTICA_MEMORY_BACKEND override may name.
VALID_BACKENDS: frozenset[str] = frozenset({"sqlite", "postgres"})

# Env var that lets an operator pin the backend regardless of URL presence.
BACKEND_VAR = "AGENTICA_MEMORY_BACKEND"


def resolve_url(env: Mapping[str, str]) -> str | None:
    """Return the connection URL by precedence, or ``None`` if unset.

    Order: ``CONTINUOUS_CLAUDE_DB_URL`` > ``DATABASE_URL`` >
    ``OPC_POSTGRES_URL``. Empty or whitespace-only values are treated as unset,
    and the returned URL is stripped of surrounding whitespace so a blank
    DSN never reaches the backend (a connection string never carries meaningful
    leading/trailing whitespace). This keeps the postgres-without-URL fail-fast
    in :func:`resolve_backend` from being bypassed by a templated ``"   "``
    value (issue #214).

    Pure: reads only ``env``.
    """
    for var in URL_VARS:
        value = env.get(var)
        if value and value.strip():
            return value.strip()
    return None


def resolve_backend(env: Mapping[str, str], *, default: str | None = "sqlite") -> str | None:
    """Return the backend name (``"sqlite"``/``"postgres"``) for ``env``.

    Precedence:
      1. An explicit, valid ``AGENTICA_MEMORY_BACKEND`` (case-insensitive).
      2. Presence of any connection URL implies ``"postgres"``.
      3. ``default`` (``"sqlite"`` by default; pass ``None`` to signal
         "undetermined" so the caller can apply its own fallback).

    Fail-fast on misconfiguration (issue #214). An explicit override is an
    operator statement, so a broken one is a hard error rather than a silent
    fall-through — regardless of ``default``:

      * **Finding 1** — a non-empty ``AGENTICA_MEMORY_BACKEND`` that does not
        name a valid backend (e.g. the typo ``"sqllite"``) raises
        :class:`ValueError`. Previously it was silently ignored and resolution
        fell through to URL presence, so a typo plus a leftover URL routed
        storage to postgres against the operator's intent.
      * **Finding 3** — ``AGENTICA_MEMORY_BACKEND=postgres`` with no connection
        URL raises :class:`ValueError` instead of resolving to ``"postgres"``
        (which downstream collapsed into a silent sqlite fall-back in the
        daemon). Blank/whitespace-only values are treated as unset, not invalid.

    Pure: reads only ``env``.

    Raises:
        ValueError: On an invalid override (Finding 1) or explicit ``postgres``
            with no connection URL (Finding 3).
    """
    raw = env.get(BACKEND_VAR)
    explicit = (raw or "").strip().lower()
    if explicit:
        if explicit not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid {BACKEND_VAR}={raw!r}: expected 'sqlite' or 'postgres' "
                "(case-insensitive)."
            )
        if explicit == "postgres" and resolve_url(env) is None:
            raise ValueError(
                f"{BACKEND_VAR}=postgres but no PostgreSQL connection URL is set; "
                f"set one of {', '.join(URL_VARS)}."
            )
        return explicit
    if resolve_url(env) is not None:
        return "postgres"
    return default


def get_connection_url() -> str | None:
    """Resolve the connection URL from the live environment."""
    return resolve_url(os.environ)


def get_active_backend(default: str | None = "sqlite") -> str | None:
    """Resolve the active backend from the live environment."""
    return resolve_backend(os.environ, default=default)
