"""Tests for scripts.core.log_safety.redact_db_values() and safe_exception().

Locks the DB-value-scrubbing contract used to render psycopg/asyncpg
exception text into log messages. See GitHub issue #117.

The two leak vectors covered:

1. Single-quoted literals (bound parameter VALUES) in the SQL statement /
   ``LINE ...`` context.
2. Unique-violation ``DETAIL: Key (col)=(value) already exists.`` echo.

Double-quoted identifiers (column/table names) are intentionally PRESERVED
for diagnosability.
"""

from __future__ import annotations

from scripts.core.log_safety import redact_db_values, safe_exception

# ---------------------------------------------------------------------------
# redact_db_values — single-quoted literals
# ---------------------------------------------------------------------------


def test_single_quoted_literal_redacted():
    assert redact_db_values("WHERE id = 'secret'") == "WHERE id = '<redacted>'"


def test_multiple_single_quoted_literals_redacted():
    text = "VALUES ('a', 'b', 'c')"
    assert redact_db_values(text) == "VALUES ('<redacted>', '<redacted>', '<redacted>')"


def test_empty_single_quoted_literal_redacted():
    assert redact_db_values("x = ''") == "x = '<redacted>'"


# ---------------------------------------------------------------------------
# redact_db_values — unique-violation DETAIL echo
# ---------------------------------------------------------------------------


def test_detail_key_value_redacted():
    text = "DETAIL:  Key (email)=(alice@example.com) already exists."
    assert redact_db_values(text) == "DETAIL:  Key (email)=(<redacted>) already exists."


def test_detail_composite_key_value_redacted():
    # The value group (right paren-pair) is redacted; the column group is left.
    text = "Key (a, b)=(1, 2) already exists."
    assert redact_db_values(text) == "Key (a, b)=(<redacted>) already exists."


# ---------------------------------------------------------------------------
# redact_db_values — identifiers preserved
# ---------------------------------------------------------------------------


def test_double_quoted_identifiers_preserved():
    text = 'column "evil" does not exist'
    assert redact_db_values(text) == 'column "evil" does not exist'


def test_identifier_preserved_alongside_redacted_literal():
    text = 'column "evil" = \'secret\''
    assert redact_db_values(text) == 'column "evil" = \'<redacted>\''


# ---------------------------------------------------------------------------
# redact_db_values — plain / edge inputs
# ---------------------------------------------------------------------------


def test_plain_text_unchanged():
    assert redact_db_values("no quotes here") == "no quotes here"


def test_empty_string_unchanged():
    assert redact_db_values("") == ""


def test_non_str_input_coerced_not_raised():
    # Must coerce via the never-raises path, not raise.
    assert redact_db_values(12345) == "12345"


def test_none_input_coerced_to_none_marker():
    assert redact_db_values(None) == "<none>"


def test_hostile_str_does_not_raise():
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("nope")

    # Coerced to the sentinel rather than raising.
    assert redact_db_values(Hostile()) == "<unrepresentable>"


# ---------------------------------------------------------------------------
# redact_db_values — full traceback uniformity
# ---------------------------------------------------------------------------


def test_redaction_applies_across_multiline_traceback():
    text = (
        "Traceback (most recent call last):\n"
        "  psycopg2.errors.UndefinedColumn: column \"evil\" does not exist\n"
        "LINE 1: ... WHERE id = 'sk-secret-value'\n"
    )
    out = redact_db_values(text)
    assert "sk-secret-value" not in out
    assert "'<redacted>'" in out
    assert 'column "evil"' in out


# ---------------------------------------------------------------------------
# safe_exception — pgcode bracketing
# ---------------------------------------------------------------------------


class _FakePgError(Exception):
    """psycopg2-style exception exposing SQLSTATE via .pgcode."""

    def __init__(self, msg: str, pgcode: str) -> None:
        super().__init__(msg)
        self.pgcode = pgcode


def test_safe_exception_includes_pgcode_bracket_and_redacts():
    e = _FakePgError("duplicate key value violates unique constraint", "23505")
    out = safe_exception(e)
    assert out.startswith("_FakePgError[23505]: ")
    assert "duplicate key value" in out


def test_safe_exception_redacts_single_quoted_value_with_pgcode():
    e = _FakePgError("INSERT failed for id = 'secret'", "23505")
    out = safe_exception(e)
    assert "secret" not in out
    assert "'<redacted>'" in out
    assert out.startswith("_FakePgError[23505]: ")


def test_safe_exception_plain_exception_has_no_bracket():
    out = safe_exception(ValueError("boom 'secret'"))
    assert out == "ValueError: boom '<redacted>'"


def test_safe_exception_none_pgcode_has_no_bracket():
    e = _FakePgError("msg", None)  # type: ignore[arg-type]
    out = safe_exception(e)
    assert out == "_FakePgError: msg"


# ---------------------------------------------------------------------------
# safe_exception — composes with safe() (control chars + truncation)
# ---------------------------------------------------------------------------


def test_safe_exception_escapes_control_chars():
    out = safe_exception(ValueError("line1\nline2"))
    assert "\n" not in out
    assert "\\x0a" in out


def test_safe_exception_truncates_long_message():
    long_msg = "x" * 1000
    out = safe_exception(ValueError(long_msg), max_len=50)
    assert "truncated" in out
    # The escaped+truncated form is far shorter than the raw 1000 chars.
    assert len(out) < 200


def test_safe_exception_hostile_str_does_not_raise():
    class HostileError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("nope")

    out = safe_exception(HostileError())
    assert out.startswith("HostileError")
    assert "unrepresentable" in out


def test_safe_exception_empty_message_renders_class_only():
    out = safe_exception(ValueError(""))
    assert out == "ValueError"


# ---------------------------------------------------------------------------
# Regression: realistic psycopg error string
# ---------------------------------------------------------------------------


def test_regression_realistic_psycopg_error_redacts_secret_keeps_identifier():
    msg = "column \"evil\" does not exist\nLINE 1: ... WHERE id = 'sk-secret-value'\n"
    out = safe_exception(ValueError(msg))
    assert "sk-secret-value" not in out
    assert '"evil"' in out  # identifier survives
    assert out.startswith("ValueError: ")
    assert "'<redacted>'" in out
