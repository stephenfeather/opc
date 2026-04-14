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

__all__ = ["safe"]

_DEFAULT_MAX_LEN = 500
_NONE_MARKER = "<none>"
_UNREPRESENTABLE = "<unrepresentable>"


def _coerce(value: object) -> str:
    """Best-effort ``str()`` conversion that never raises.

    Hostile objects may raise from ``__str__`` or return a non-str. We
    do not fall back to ``repr()`` because ``repr()`` can also raise on
    hostile objects; a fixed sentinel is safer.
    """
    if value is None:
        return _NONE_MARKER
    if isinstance(value, str):
        return value
    try:
        result = str(value)
    except Exception:
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
        else:
            out.append(f"\\u{o:04x}")
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
    - Non-ASCII characters ``\\u0100``+ (including emoji and Unicode
      surrogates) → ``\\uNNNN`` markers.
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
    coerced = _coerce(value)
    raw_len = len(coerced)
    if raw_len > max_len:
        head = coerced[:max_len]
        dropped = raw_len - max_len
        return _escape_controls(head) + f"...[truncated {dropped} characters]"
    return _escape_controls(coerced)
