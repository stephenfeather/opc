"""Sanitize DB-sourced values for inclusion in log messages.

Addresses GitHub issue #104: memory_daemon and its extractors interpolate
DB-sourced strings (session IDs, project paths, transcript paths,
subprocess stderr) into log messages via f-strings. An attacker who has
DB write access can inject newlines (log forgery), ESC sequences (ANSI
terminal manipulation on an admin's TTY), or other control characters.

The ``safe()`` helper is the single black-box boundary where untrusted
bytes become safe display text. One helper, one rule, one test suite.

See ``thoughts/shared/plans/issue-104-log-injection-sanitization.md``.
"""

from __future__ import annotations

import re

__all__ = [
    "safe",
    "redact_db_values",
    "safe_exception",
    "redact_secrets",
    "safe_secret",
]

_DEFAULT_MAX_LEN = 500

# Single-quoted literals delimit bound-parameter VALUES in Postgres error
# text (the SQL statement and ``LINE ...`` context echo them). Double-quoted
# identifiers (column/table names) are intentionally NOT matched — they carry
# no user data and are valuable for diagnosis.
#
# SQL escapes a single quote inside a string literal by DOUBLING it
# (``'O''Brien'`` is the one logical value ``O'Brien``). ``(?:[^']|'')*``
# consumes those doubled quotes as part of the same literal so the whole value
# collapses to one ``'<redacted>'`` instead of splitting into two literals and
# leaking an inner value fragment between them. Over-redaction is the safe
# direction; never leak a fragment (#211 review, Finding 1).
_SINGLE_QUOTED_LITERAL = re.compile(r"'(?:[^']|'')*'")

# Unique-violation DETAIL echo: ``DETAIL: Key (col)=(value) already exists.``
# Redact only the value group — the paren-pair after ``=`` — leaving the
# column-name group intact. The value is matched GREEDILY through the LAST
# ``)`` on the same line (``[^\n]*``): a value containing its own right paren
# (e.g. ``(foo)bar)``) must not leak its suffix. ``[^\n]`` keeps the match on
# one line so multiline tracebacks are not over-collapsed. Over-redacting a
# trailing same-line parenthetical is the accepted safe direction; leaking any
# value char is the bug this guards against (#117 review, Finding 1).
_DETAIL_KEY_VALUE = re.compile(r"\)=\([^\n]*\)")
_NONE_MARKER = "<none>"
_UNREPRESENTABLE = "<unrepresentable>"


def _coerce(value: object) -> str:
    """Best-effort ``str()`` conversion that never raises a non-system Exception.

    Hostile objects may raise from ``__str__`` or return a non-str. We
    do not fall back to ``repr()`` because ``repr()`` can also raise on
    hostile objects; a fixed sentinel is safer.

    ``KeyboardInterrupt`` and ``SystemExit`` (``BaseException`` subclasses
    outside the ``Exception`` hierarchy) are **not** suppressed — a user
    hitting Ctrl-C during a log render should still abort the process.
    """
    if value is None:
        return _NONE_MARKER
    if isinstance(value, str):
        return value
    try:
        result = str(value)
    except Exception:  # noqa: BLE001
        # Deliberately narrow to Exception, NOT BaseException. KeyboardInterrupt
        # and SystemExit MUST propagate so Ctrl-C and sys.exit() still work
        # when a log line is rendering. CodeRabbit cycle-1 suggested
        # BaseException; ARCHITECT overruled — the "never raises" contract
        # covers regular exceptions only. The docstring is updated to
        # "Never raises a non-system Exception" to codify the invariant.
        return _UNREPRESENTABLE
    if not isinstance(result, str):
        return _UNREPRESENTABLE
    return result


def _escape_controls(s: str) -> str:
    """Enforce printable-ASCII-only output; escape everything else.

    The output contract is strict: only ``\\t`` or printable ASCII
    (``0x20``–``0x7e``) is ever passed through unchanged. C0 controls,
    DEL (``0x7f``), C1 controls (``0x80``–``0x9f``), and non-ASCII
    characters (incl. Unicode surrogates and emoji) are all escaped.

    C1 in particular matters for the threat model — a raw ``\\x9b``
    (CSI) is treated as an ANSI escape by UTF-8 terminals, so allowing
    it through would reopen the exact ANSI-injection vector this helper
    exists to close. Allowlisting printable ASCII is simpler and safer
    than chasing denylists across Unicode categories.
    """
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch == "\t" or 0x20 <= o <= 0x7E:
            out.append(ch)
        elif o <= 0xFF:
            out.append(f"\\x{o:02x}")
        elif o <= 0xFFFF:
            out.append(f"\\u{o:04x}")
        else:
            # Non-BMP (> U+FFFF) — emoji and supplementary-plane chars
            # get the fixed-width \UNNNNNNNN form. Using \uNNNNNN here
            # would break log parsers expecting 4-hex \u (cycle-1 Gemini
            # + Copilot).
            out.append(f"\\U{o:08x}")
    return "".join(out)


def safe(value: object, *, max_len: int = _DEFAULT_MAX_LEN) -> str:
    """Render an arbitrary value as a single-line, control-char-free log field.

    **Stability:** the behavioral contract below is covered by
    ``tests/test_log_safety.py`` and is intended to remain stable so call
    sites can rely on the output invariant. The escape format
    (``\\xNN`` / ``\\uNNNN``) and truncation marker text are considered
    part of the contract — log-parsing tools may grep for them.

    Contract (see ``tests/test_log_safety.py``):

    - ``None`` → ``"<none>"`` (preserves field-was-null forensic signal)
    - Empty string → ``""``
    - Non-str values → ``str(value)`` first; if that raises or returns
      non-str, the sentinel ``"<unrepresentable>"`` is used instead of
      falling back to ``repr()`` (which can also raise on hostile
      objects).
    - ``\\n``, ``\\r``, ``\\x00``–``\\x08``, ``\\x0b``–``\\x1f``, ``\\x7f``
      → replaced with ``\\xNN`` markers (lowercase hex, reversible for
      forensics).
    - ``\\x80``–``\\xff`` (includes C1 controls like ``\\x9b`` CSI) →
      ``\\xNN`` markers. These look "innocent" but UTF-8 terminals
      interpret ``\\x9b`` as an ANSI escape, so allowing them through
      would reopen the injection vector.
    - Non-ASCII BMP characters ``\\u0100``–``\\uffff`` (including most
      Latin scripts, CJK, lone surrogates) → ``\\uNNNN`` (fixed 4 hex
      digits).
    - Non-BMP characters ``\\U00010000``+ (emoji, supplementary planes)
      → ``\\UNNNNNNNN`` (fixed 8 hex digits). Using 4-digit ``\\u`` for
      these would produce invalid 5+-digit output.
    - ``\\t`` (``0x09``) preserved — tabs are common in real data and
      harmless for log tailing.
    - Inputs longer than ``max_len`` raw characters are truncated to
      ``max_len`` then suffixed with the ASCII marker
      ``"...[truncated N characters]"``. Truncation happens on the raw
      input length (Python string length = code points), not on the
      escaped output length, to prevent a string of newlines from
      ballooning the log line. The unit is characters (code points),
      not bytes — a truncated 1000-codepoint emoji string would drop
      far more than N bytes, and the suffix reflects the actual unit.

    The return value is always a ``str`` containing only ``\\t`` (``0x09``)
    or printable ASCII (``0x20``–``0x7e``). No other characters can leak
    into the output.

    **Accepted trade-off — sentinel ambiguity:** a DB-stored value whose
    literal text is ``"<none>"`` or ``"<unrepresentable>"`` renders
    identically to the real sentinel. Operators cannot distinguish a
    NULL column from an attacker-written literal on log inspection alone.
    This was accepted during Aegis review (#104) because the alternative
    — suppressing sentinel output entirely — would lose the forensic
    field-was-null signal that motivated choosing a marker over empty
    string in the first place.

    Usage:

        from scripts.core.log_safety import safe

        log(f"Extracting session {safe(session_id)} "
            f"(project={safe(project_dir)})")

    For subprocess stderr, decode with ``errors='replace'`` before wrapping
    so that non-UTF8 bytes become replacement characters rather than a
    ``"b'...'"`` repr::

        stderr_text = result.stderr.decode(errors="replace")
        log(f"zstd failed: {safe(stderr_text)}")
    """
    if max_len < 0:
        # Raise on a developer error at call time — better to surface a
        # misconfigured call site than to silently truncate in ways the
        # caller didn't expect (cycle-1 Copilot; ARCHITECT: raise not clamp).
        # A negative max_len is never meaningful and can only come from a
        # coding mistake, not untrusted input.
        raise ValueError(f"max_len must be >= 0 (got {max_len})")
    coerced = _coerce(value)
    raw_len = len(coerced)
    if raw_len > max_len:
        head = coerced[:max_len]
        dropped = raw_len - max_len
        return _escape_controls(head) + f"...[truncated {dropped} characters]"
    return _escape_controls(coerced)


def redact_db_values(text: object) -> str:
    """Strip DB-sourced VALUES from Postgres/psycopg error text.

    Addresses GitHub issue #117: psycopg2/asyncpg exception messages embed
    the failing SQL statement plus its bound parameter values (in the
    message, the ``LINE ...`` context, and the unique-violation ``DETAIL:``
    echo). Logging that text raw leaks DB content. This helper removes the
    two real leak vectors while preserving the error class and identifiers
    needed to diagnose the failure:

    1. **Single-quoted literals** — every ``'...'`` (regex ``'[^']*'``) is
       replaced with ``'<redacted>'``. Single quotes delimit VALUES in
       Postgres. Double-quoted identifiers (``"column"``/``"table"``) are
       deliberately LEFT INTACT — they carry no user data and are useful
       for diagnosis.
    2. **Unique-violation DETAIL echo** — psycopg emits
       ``DETAIL: Key (col)=(value) already exists.``; the value group
       ``)=(...)`` (regex ``\\)=\\([^\\n]*\\)``, greedy to the LAST ``)`` on
       the line so a value containing its own paren cannot leak) is replaced
       with ``)=(<redacted>)``, leaving the column-name group intact.

    Both substitutions are global (all occurrences) and operate uniformly on
    the whole string, so the same call works for a single exception message
    or a full ``traceback.format_exc()``.

    Control characters are **not** escaped here — that is ``safe()``'s job.
    Callers that log the result should wrap it with ``safe()`` (or use
    ``safe_exception()``, which composes the two).

    Never raises a non-system Exception: non-``str`` input is coerced via the
    same never-raises path ``safe()`` uses (``_coerce``).
    """
    coerced = _coerce(text)
    coerced = _SINGLE_QUOTED_LITERAL.sub("'<redacted>'", coerced)
    coerced = _DETAIL_KEY_VALUE.sub(")=(<redacted>)", coerced)
    return coerced


_SECRET_MARKER = "<redacted-secret>"

# Credential-shaped tokens that can appear bare in arbitrary subprocess stderr
# (issue #209). Each entry is ``(compiled pattern, replacement)`` and the rules
# are applied in order. Structured forms (Bearer / connection-string password /
# sensitive ``VAR=value`` assignment) keep their surrounding context via ``\1``/
# ``\2`` backrefs and mask only the secret; bare prefixed tokens are replaced
# whole. The replacement marker contains no regex-replacement metacharacters
# (``\`` / ``\g<>``), so it is substituted literally.
#
# Matching is conservative and prefix-anchored (the audit's explicit
# recommendation): only known credential prefixes are masked, so a leaked
# OPENAI/VOYAGE/GitHub/AWS token is caught while UUIDs, hashes, and ordinary
# hyphenated words pass through unchanged. ``\b`` guards keep ``sk-`` from
# matching inside words like ``task-``/``disk-``. This complements ``safe()``
# (which escapes control chars) — it does NOT replace it; use ``safe_secret``
# to get both.
_SECRET_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Bearer / Authorization tokens — keep the scheme, mask the credential.
    (re.compile(r"(Bearer\s+)\S+", re.IGNORECASE), rf"\1{_SECRET_MARKER}"),
    # Connection-string passwords: ``://user:PASSWORD@host`` — keep user+host.
    (re.compile(r"(://[^:/@\s]+:)[^@\s]+(@)"), rf"\1{_SECRET_MARKER}\2"),
    # Sensitive ``VAR=value`` / ``export VAR=value`` env assignments — keep the
    # variable name (which names the leak), mask the value.
    (
        re.compile(
            r"((?:export\s+)?[A-Za-z0-9_]*"
            r"(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)[A-Za-z0-9_]*=)\S+",
            re.IGNORECASE,
        ),
        rf"\1{_SECRET_MARKER}",
    ),
    # Bare ``sk-*`` tokens (OpenAI, Anthropic ``sk-ant-``, Stripe, …).
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), _SECRET_MARKER),
    # Bare GitHub tokens: classic ``ghp_/gho_/ghu_/ghs_/ghr_`` + fine-grained.
    (re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}"), _SECRET_MARKER),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), _SECRET_MARKER),
    # AWS access key IDs (long-term ``AKIA`` / temporary ``ASIA``).
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), _SECRET_MARKER),
    # Bare Voyage AI keys (``pa-`` prefix).
    (re.compile(r"\bpa-[A-Za-z0-9_-]{16,}"), _SECRET_MARKER),
)


def redact_secrets(text: object) -> str:
    """Mask credential-shaped tokens in free text (e.g. subprocess stderr).

    Addresses GitHub issue #209: the ``claude -p`` extraction child runs with
    the full daemon environment (VOYAGE/OPENAI/GH tokens). On a crash a token
    can appear in its stderr, which is captured and persisted to
    ``sessions.last_error``. ``safe()`` escapes control chars but does NOT
    redact secret-shaped tokens, so a leaked credential would otherwise be
    stored at rest. This helper masks known credential shapes with
    ``"<redacted-secret>"`` so the secret never reaches the log or the DB.

    Matching is conservative and prefix-anchored (see ``_SECRET_RULES``):
    bare ``sk-*``/GitHub/AWS/Voyage tokens, ``Bearer`` credentials,
    connection-string passwords, and sensitive ``VAR=value`` env assignments.
    Structured forms keep their context (scheme / host / variable name) and
    mask only the value. UUIDs, hashes, and ordinary hyphenated words are not
    matched, so over-redaction is avoided.

    Control characters are **not** escaped here — that is ``safe()``'s job.
    Callers that log or store the result should wrap it with ``safe()`` (or
    use ``safe_secret()``, which composes the two). The substitution is global
    and idempotent (the marker matches none of the rules), so it is safe to
    run on already-redacted text.

    Never raises a non-system Exception: non-``str`` input is coerced via the
    same never-raises path ``safe()`` uses (``_coerce``).
    """
    coerced = _coerce(text)
    for pattern, replacement in _SECRET_RULES:
        coerced = pattern.sub(replacement, coerced)
    return coerced


def safe_secret(value: object, *, max_len: int = _DEFAULT_MAX_LEN) -> str:
    """Redact credential-shaped tokens, then render as a safe log field.

    Composes :func:`redact_secrets` (mask secrets) with :func:`safe` (escape
    control chars + bound length), in that order: redaction runs on the full
    raw string BEFORE ``safe()`` escapes and truncates, so a secret in the
    truncated tail cannot survive and a control char smuggled alongside a token
    is still escaped. The output therefore satisfies the full ``safe()``
    contract — only ``\\t`` or printable ASCII (``0x20``–``0x7e``) — with any
    known credential shape replaced by ``"<redacted-secret>"``.

    Use this at the boundary where untrusted child-process stderr is logged or
    persisted to ``last_error`` (issue #209). Never raises a non-system
    Exception.
    """
    return safe(redact_secrets(value), max_len=max_len)


def _safe_getattr(obj: object, name: str) -> object:
    """``getattr(obj, name, None)`` that never raises a non-system Exception.

    Hostile exception objects may expose ``pgcode``/``diag``/identifier
    fields as properties that raise. Narrow to ``Exception`` (NOT
    ``BaseException``) so ``KeyboardInterrupt``/``SystemExit`` still
    propagate, mirroring ``_coerce``.
    """
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - hostile property must not break logging
        return None


_SENTINEL_NAME = "Exception"

# Structured IDENTIFIER fields (schema metadata, NOT row data — SAFE to log),
# rendered in this fixed order. Each tuple is ``(label, attr_names)`` where the
# label is the short ``key=`` shown in the output and ``attr_names`` is the
# tuple of source attribute names tried in order (first non-empty wins). The
# rendered LABEL is fixed regardless of which source attribute supplied the
# value.
#
# Most fields share a name between psycopg2 ``.diag`` and asyncpg direct attrs.
# The datatype field is the exception: psycopg2 ``.diag`` exposes
# ``datatype_name`` while asyncpg's direct attribute is ``data_type_name``
# (with underscores). Trying both keeps asyncpg datatype diagnostics from
# being silently dropped (#117 review, Finding 2).
_IDENTIFIER_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("schema", ("schema_name",)),
    ("table", ("table_name",)),
    ("column", ("column_name",)),
    ("datatype", ("datatype_name", "data_type_name")),
    ("constraint", ("constraint_name",)),
)


def safe_exception(e: object, *, max_len: int = _DEFAULT_MAX_LEN) -> str:
    """Render an exception for logging via a structured-diagnostics ALLOWLIST.

    Composes the structured fields with :func:`safe` (printable-ASCII-only +
    truncation) so the full output contract of ``safe()`` still holds — the
    return value contains only ``\\t`` or printable ASCII. As with ``safe()``,
    over-length input is truncated to ``max_len`` raw characters and then the
    marker ``"...[truncated N characters]"`` is appended, so the returned
    string may exceed ``max_len`` by the length of that marker.

    **DB exceptions drop the free-text message.** psycopg/asyncpg messages
    embed DB VALUES in forms that free-text regex redaction cannot reliably
    catch — double-quoted values
    (``invalid input syntax for type uuid: "secret"``),
    ``DETAIL: Failing row contains (1, secret, ...)``, and
    ``CONTEXT: COPY ...: "secret"``. Postgres uses double quotes for BOTH
    identifiers and values, so regex on free text cannot be leak-tight
    (#117 review, HIGH design finding). For DB exceptions we therefore render
    ONLY an allowlist of safe structured fields and discard the message.

    A DB exception is one where a SQLSTATE code is present OR at least one
    structured identifier was found:

    - ``code = e.pgcode or e.sqlstate`` — psycopg2 exposes SQLSTATE as
      ``.pgcode``, asyncpg as ``.sqlstate``.
    - IDENTIFIER fields (``schema_name``, ``table_name``, ``column_name``,
      ``datatype_name``, ``constraint_name``) are read from the psycopg2
      ``.diag`` namespace, falling back to direct attributes for asyncpg.
      These are schema metadata, not row data, and are SAFE to log. The
      value-bearing diag fields (``message_*``, ``context``, ``detail``,
      ``hint``, ``internal_query``, ``*query``, ``message``) are NEVER read.

    Render rules:

    - DB exception → ``ClassName`` + ``[CODE]`` (if a code is present) +
      space-joined ``label=value`` identifier pairs in the fixed order
      schema, table, column, datatype, constraint. Free-text message dropped.
      Example: ``UniqueViolation[23505] table=sessions constraint=sessions_pkey``.
    - Non-DB exception (no code, no identifiers) →
      ``f"{name}: {redact_db_values(_coerce(e))}"`` (the best-effort regex is
      still applied to ordinary Python exceptions whose message may
      incidentally contain a quoted value). An empty coerced message renders
      ``name`` alone.

    The whole assembled string is passed through ``safe()`` so a control char
    smuggled into an identifier is escaped and the output is truncated.

    Never raises a non-system Exception. Any unexpected error during assembly
    falls back to ``safe(name, ...)`` (or a fixed sentinel if even the type
    name is unreadable).
    """
    try:
        name = type(e).__name__
    except Exception:  # noqa: BLE001 - hostile metaclass must not break logging
        name = _SENTINEL_NAME

    try:
        # SQLSTATE: psycopg2 uses .pgcode, asyncpg uses .sqlstate.
        raw_code = _safe_getattr(e, "pgcode") or _safe_getattr(e, "sqlstate")
        code = _coerce(raw_code) if raw_code else ""

        # Structured identifiers: prefer the psycopg2 .diag value, else the
        # asyncpg direct attr. Keep only non-empty values.
        diag = _safe_getattr(e, "diag")
        identifiers: list[str] = []
        for label, attr_names in _IDENTIFIER_FIELDS:
            value = None
            for attr in attr_names:
                if diag is not None:
                    value = _safe_getattr(diag, attr)
                if not value:
                    value = _safe_getattr(e, attr)
                if value:
                    break
            if value:
                identifiers.append(f"{label}={_coerce(value)}")

        is_db_exception = bool(code) or bool(identifiers)

        if is_db_exception:
            assembled = f"{name}[{code}]" if code else name
            if identifiers:
                assembled = assembled + " " + " ".join(identifiers)
            return safe(assembled, max_len=max_len)

        # Non-DB exception: keep the best-effort regex-redacted message.
        msg = redact_db_values(_coerce(e))
        if msg:
            return safe(f"{name}: {msg}", max_len=max_len)
        return safe(name, max_len=max_len)
    except Exception:  # noqa: BLE001 - assembly must never break logging
        # Defensive worst-case fallback: render just the class name (or the
        # sentinel if even that is unavailable).
        return safe(name, max_len=max_len)
