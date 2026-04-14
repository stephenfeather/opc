"""Tests for scripts.core.log_safety.safe().

Locks the sanitizer contract used to render DB-sourced fields into log
messages. See thoughts/shared/plans/issue-104-log-injection-sanitization.md
and GitHub issue #104.
"""

from __future__ import annotations

import pytest

from scripts.core.log_safety import safe

# ---------------------------------------------------------------------------
# None / empty
# ---------------------------------------------------------------------------


def test_none_renders_as_none_marker():
    # Per ARCHITECT refinement: preserve field-was-null forensic signal.
    assert safe(None) == "<none>"


def test_empty_string_passes_through():
    assert safe("") == ""


# ---------------------------------------------------------------------------
# Happy-path ASCII
# ---------------------------------------------------------------------------


def test_plain_ascii_unchanged():
    assert safe("hello world") == "hello world"


def test_printable_punctuation_unchanged():
    s = "session-abc_123:/Users/x/project (foo)"
    assert safe(s) == s


def test_tab_preserved():
    assert safe("a\tb") == "a\tb"


# ---------------------------------------------------------------------------
# Control-character sanitization
# ---------------------------------------------------------------------------


def test_newline_replaced():
    assert safe("a\nb") == "a\\x0ab"


def test_carriage_return_replaced():
    assert safe("a\rb") == "a\\x0db"


def test_crlf_both_replaced():
    assert safe("a\r\nb") == "a\\x0d\\x0ab"


def test_esc_replaced():
    assert safe("\x1b[31mRED\x1b[0m") == "\\x1b[31mRED\\x1b[0m"


def test_null_byte_replaced():
    assert safe("a\x00b") == "a\\x00b"


def test_all_c0_controls_replaced_except_tab():
    # 0x00-0x1f minus 0x09 (tab); plus 0x7f (DEL).
    raw = "".join(chr(c) for c in range(0x00, 0x20) if c != 0x09) + "\x7f"
    out = safe(raw)
    # No raw control chars should remain.
    for c in raw:
        assert c not in out, f"raw {hex(ord(c))} leaked into output"
    # Every replaced char should appear as \xNN.
    for c in raw:
        assert f"\\x{ord(c):02x}" in out


def test_del_byte_replaced():
    assert safe("a\x7fb") == "a\\x7fb"


def test_bell_replaced():
    assert safe("beep\x07here") == "beep\\x07here"


# ---------------------------------------------------------------------------
# C1 control sanitization (Gemini Round 1 HIGH)
# ---------------------------------------------------------------------------


def test_c1_csi_replaced():
    # 0x9b is CSI — UTF-8 terminals interpret it as ESC [, reopening the
    # ANSI-injection vector if allowed through.
    assert safe("\x9b[31mRED") == "\\x9b[31mRED"


def test_all_c1_controls_replaced():
    # 0x80-0x9f must all be escaped.
    for c in range(0x80, 0xA0):
        raw = chr(c)
        out = safe(raw)
        assert raw not in out
        assert f"\\x{c:02x}" in out


def test_high_byte_latin1_replaced():
    # 0xa0-0xff: e.g. \xa0 NBSP — not a control but not printable ASCII.
    assert safe("a\xa0b") == "a\\xa0b"
    assert safe("\xff") == "\\xff"


def test_non_ascii_unicode_replaced():
    # Emoji / non-Latin-1 chars must be escaped as \uNNNN.
    out = safe("caf\u00e9")  # é
    assert "\u00e9" not in out
    assert "\\xe9" in out  # within 0xff, gets \xNN
    out = safe("hi \u4e2d")  # Chinese char > 0xff
    assert "\u4e2d" not in out
    assert "\\u4e2d" in out


def test_emoji_replaced():
    # Emoji are typically > 0xffff (use surrogate pair when in BMP), but
    # a BMP emoji like U+2600 ☀ is single code point > 0xff.
    out = safe("\u2600 sunny")
    assert "\u2600" not in out
    assert "\\u2600" in out


def test_unicode_surrogate_replaced():
    # Lone surrogate (invalid Unicode but Python can hold it in a str).
    out = safe("\ud83d")
    assert "\ud83d" not in out
    assert "\\ud83d" in out


# ---------------------------------------------------------------------------
# Non-string coercion
# ---------------------------------------------------------------------------


def test_int_coerced():
    assert safe(42) == "42"


def test_float_coerced():
    assert safe(3.14) == "3.14"


def test_list_coerced():
    assert safe(["a", "b"]) == "['a', 'b']"


def test_bytes_coerced_and_sanitized():
    # bytes → str(bytes) yields "b'abc'" then sanitized (no control chars here).
    assert safe(b"abc") == "b'abc'"


def test_bytes_with_escape_sanitized():
    # str(b"\x1b") -> "b'\\x1b'" — the ESC is already escaped by bytes' repr.
    # But if a hostile bytes object contained a real ESC in its repr
    # (impossible with stdlib bytes, but a subclass could), we still scrub.
    out = safe(b"\x1b")
    assert "\x1b" not in out


# ---------------------------------------------------------------------------
# Hostile __str__
# ---------------------------------------------------------------------------


class _RaisingStr:
    def __str__(self) -> str:
        raise RuntimeError("hostile __str__")

    def __repr__(self) -> str:  # pragma: no cover - also hostile
        raise RuntimeError("hostile __repr__")


class _StrReturnsNonStr:
    def __str__(self):  # type: ignore[override]
        return 12345  # type: ignore[return-value]


def test_hostile_str_falls_back_to_sentinel():
    # Per ARCHITECT refinement: hardcode sentinel, do NOT fall back to repr()
    # since repr() can also raise on hostile objects.
    assert safe(_RaisingStr()) == "<unrepresentable>"


def test_str_returning_nonstr_falls_back_to_sentinel():
    # TypeError from str() on broken __str__ is also treated as unrepresentable.
    assert safe(_StrReturnsNonStr()) == "<unrepresentable>"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_short_input_not_truncated():
    s = "x" * 500
    assert safe(s) == s


def test_long_input_truncated_with_ascii_marker():
    s = "x" * 1000
    out = safe(s)
    assert out.startswith("x" * 500)
    assert out.endswith("...[truncated 500 characters]")
    assert "\u2026" not in out  # ASCII-only per ARCHITECT


def test_truncation_marker_reports_remaining_characters():
    s = "a" * 750
    out = safe(s)
    # Default max_len=500 → 250 characters dropped.
    assert out.endswith("...[truncated 250 characters]")


def test_custom_max_len():
    out = safe("abcdefghij", max_len=4)
    assert out.startswith("abcd")
    assert out.endswith("...[truncated 6 characters]")


def test_truncation_happens_before_escaping_length():
    # 1000 newlines. Each expands to "\x0a" (4 chars) if escaped first,
    # which would balloon the log. We must cap at the raw code-point
    # count first.
    s = "\n" * 1000
    out = safe(s)
    # The truncation count refers to raw input code points dropped,
    # not escaped chars.
    assert "[truncated 500 characters]" in out
    # And no raw newline leaked.
    assert "\n" not in out


def test_truncation_unit_is_codepoints_not_bytes():
    # 1000 emoji code points — each is 4 bytes in UTF-8. We truncate at
    # 500 code points. The marker must say "characters" because that's
    # the actual unit — saying "bytes" would be off by a factor of 4.
    s = "\U0001f600" * 1000  # each codepoint is 1 len-unit in Python str
    out = safe(s)
    assert "[truncated 500 characters]" in out
    assert "[truncated 2000 bytes]" not in out


# ---------------------------------------------------------------------------
# Output invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "plain",
        "a\nb\rc\x1bd\x00e",
        42,
        ["a", "\n", "b"],
        b"\x00\x01\x02",
        "x" * 2000,
        _RaisingStr(),
        # Gemini Round 1 HIGH: extend coverage across Unicode space so
        # the ASCII-only invariant is genuinely enforced, not coincidence.
        "\x9b[31m",  # C1 CSI
        "\x80\x81\x82\x9f",  # C1 block
        "\xa0\xff",  # Latin-1 supplement (non-control but non-ASCII)
        "caf\u00e9",  # accented Latin
        "\u4e2d\u6587",  # CJK
        "\u2600\u2b50",  # BMP symbols
        "\U0001f600",  # supplementary-plane emoji
        "\ud83d",  # lone surrogate
    ],
)
def test_output_has_no_raw_control_chars(value):
    out = safe(value)
    assert isinstance(out, str)
    for c in out:
        o = ord(c)
        # Allow tab (0x09) and printable 0x20-0x7e. Nothing else.
        assert c == "\t" or 0x20 <= o <= 0x7E, f"leaked control char {hex(o)} in {out!r}"


@pytest.mark.parametrize("value", [None, "x", 1, ["a"], b"b"])
def test_output_is_always_str(value):
    assert isinstance(safe(value), str)
