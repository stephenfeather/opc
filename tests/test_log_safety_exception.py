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
# redact_db_values — SQL doubled-quote escaping inside a single literal
#
# Postgres escapes a single quote inside a string literal by doubling it
# (``'O''Brien'`` is the one logical value ``O'Brien``). The literal regex must
# treat the doubled ``''`` as part of ONE literal so the whole value collapses
# to a single ``'<redacted>'`` and no inner value fragment survives between two
# separately-matched literals (#211 Copilot review, Finding 1).
# ---------------------------------------------------------------------------


def test_doubled_quote_escaped_literal_fully_redacted():
    # 'O''Brien' is one SQL value (O'Brien); none of its chars may leak.
    out = redact_db_values("name = 'O''Brien'")
    assert out == "name = '<redacted>'"
    assert "Brien" not in out


def test_doubled_quote_escaped_value_no_fragment_leaks():
    out = redact_db_values("value = 'it''s a secret'")
    assert out == "value = '<redacted>'"
    assert "secret" not in out


def test_multiple_doubled_quotes_in_one_literal_redacted():
    # The literal 'a''b''c' (value a'b'c) collapses to a single redaction; no
    # value fragment ('b') survives between separately-matched literals. (The
    # marker text "redacted" itself contains a/c, so assert the exact output.)
    out = redact_db_values("x = 'a''b''c'")
    assert out == "x = '<redacted>'"
    assert "'b'" not in out


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


def test_detail_value_with_embedded_paren_does_not_leak_suffix():
    # The value contains a right paren; nothing after )=( may survive on the
    # value side. Over-redaction to the final paren is the safe direction;
    # leaking any value char is the bug (Finding 1).
    text = "DETAIL: Key (path)=(foo)bar) already exists."
    out = redact_db_values(text)
    assert "foo" not in out
    assert "bar" not in out
    assert ")=(<redacted>)" in out
    assert out.endswith(" already exists.")


def test_detail_value_is_parenthetical_secret_fully_redacted():
    text = "DETAIL: Key (k)=((secret)) already exists."
    out = redact_db_values(text)
    assert "secret" not in out
    assert ")=(<redacted>)" in out


def test_detail_filesystem_path_value_with_parens_redacted():
    text = "DETAIL: Key (file)=(/var/data (old)/backup.db) already exists."
    out = redact_db_values(text)
    assert "/var/data" not in out
    assert "backup.db" not in out
    assert ")=(<redacted>)" in out


def test_detail_composite_value_with_embedded_paren_keeps_column_group():
    text = "Key (a, b)=(1, 2)x) already exists."
    out = redact_db_values(text)
    assert "Key (a, b)=(<redacted>)" in out
    assert out.startswith("Key (a, b)=(<redacted>)")
    assert "1, 2" not in out


def test_detail_redaction_does_not_cross_newline():
    # The value match is newline-bounded so a multiline traceback is not
    # over-collapsed across lines.
    text = "DETAIL: Key (k)=(v) already exists.\nLINE 2: keep (this) paren\n"
    out = redact_db_values(text)
    assert "(this)" in out
    assert ")=(<redacted>)" in out
    assert "\n" in out


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
# safe_exception — structured-diagnostics ALLOWLIST (DB exceptions)
#
# For DB exceptions the free-text message is DROPPED entirely: psycopg/asyncpg
# messages leak DB VALUES in forms the regex misses (double-quoted values,
# DETAIL row contents, CONTEXT echoes). Only SAFE structured IDENTIFIER fields
# (schema/table/column/datatype/constraint) plus the SQLSTATE code are rendered.
# ---------------------------------------------------------------------------


class _FakeDiag:
    """psycopg2-style .diag namespace exposing structured identifier fields."""

    def __init__(self, **fields: object) -> None:
        # Default every known field to None; override with provided values.
        for name in (
            "schema_name",
            "table_name",
            "column_name",
            "datatype_name",
            "constraint_name",
        ):
            setattr(self, name, None)
        for name, value in fields.items():
            setattr(self, name, value)


class _FakePgError(Exception):
    """psycopg2-style exception: SQLSTATE via .pgcode, identifiers via .diag."""

    def __init__(self, msg: str, pgcode: object, **diag_fields: object) -> None:
        super().__init__(msg)
        self.pgcode = pgcode
        self.diag = _FakeDiag(**diag_fields)


class _FakeAsyncpgError(Exception):
    """asyncpg-style exception: SQLSTATE via .sqlstate, identifiers as attrs."""

    def __init__(self, msg: str, sqlstate: object, **fields: object) -> None:
        super().__init__(msg)
        self.sqlstate = sqlstate
        for name, value in fields.items():
            setattr(self, name, value)


def test_safe_exception_psycopg_renders_code_and_identifiers_drops_message():
    e = _FakePgError(
        "duplicate key value violates unique constraint",
        "23505",
        table_name="sessions",
        constraint_name="sessions_pkey",
    )
    out = safe_exception(e)
    assert out == "_FakePgError[23505] table=sessions constraint=sessions_pkey"
    # Free-text message is dropped.
    assert "duplicate key value" not in out


def test_safe_exception_psycopg_identifier_order_schema_to_constraint():
    e = _FakePgError(
        "msg",
        "23505",
        constraint_name="c",
        datatype_name="d",
        column_name="col",
        table_name="t",
        schema_name="public",
    )
    out = safe_exception(e)
    assert out == "_FakePgError[23505] schema=public table=t column=col datatype=d constraint=c"


def test_safe_exception_asyncpg_uses_sqlstate_and_direct_attrs():
    e = _FakeAsyncpgError(
        "duplicate key",
        "23505",
        table_name="users",
        column_name="email",
    )
    out = safe_exception(e)
    assert out == "_FakeAsyncpgError[23505] table=users column=email"
    assert "duplicate key" not in out


def test_safe_exception_db_with_code_but_no_identifiers_drops_message():
    # UndefinedTable: code present, every diag field None → code alone.
    e = _FakePgError("relation \"secret_tbl\" does not exist", "42P01")
    out = safe_exception(e)
    assert out == "_FakePgError[42P01]"
    assert "secret_tbl" not in out


def test_safe_exception_db_with_identifiers_but_no_code():
    # Identifiers present but pgcode None → still a DB exception (no bracket).
    e = _FakePgError("msg", None, table_name="t")
    out = safe_exception(e)
    assert out == "_FakePgError table=t"
    assert "msg" not in out


def test_safe_exception_asyncpg_data_type_name_alias_rendered():
    # asyncpg's direct datatype attribute is ``data_type_name`` (underscores),
    # while psycopg2 ``.diag`` uses ``datatype_name``. The datatype field must
    # fall back to the asyncpg attribute name so asyncpg datatype diagnostics
    # are not silently dropped (#117 review, Finding 2). The OUTPUT label stays
    # ``datatype=`` regardless of which source attribute supplied the value.
    e = _FakeAsyncpgError(
        "invalid input",
        "22P02",
        table_name="users",
        data_type_name="uuid",
    )
    out = safe_exception(e)
    assert out == "_FakeAsyncpgError[22P02] table=users datatype=uuid"
    assert "invalid input" not in out


# ---------------------------------------------------------------------------
# safe_exception — reviewer leak cases (message MUST be dropped for DB errors)
# ---------------------------------------------------------------------------


class _LeakyPgError(Exception):
    """DB exception whose __str__ returns a value-bearing free-text message."""

    def __init__(self, leaky_text: str) -> None:
        self._text = leaky_text
        self.pgcode = "22P02"
        self.diag = _FakeDiag(table_name="t")

    def __str__(self) -> str:  # noqa: D401
        return self._text


def test_safe_exception_drops_double_quoted_value_leak():
    e = _LeakyPgError('invalid input syntax for type uuid: "sk-secret"')
    out = safe_exception(e)
    assert "sk-secret" not in out
    assert out == "_LeakyPgError[22P02] table=t"


def test_safe_exception_drops_detail_failing_row_leak():
    e = _LeakyPgError("DETAIL:  Failing row contains (1, sk-secret, ...).")
    out = safe_exception(e)
    assert "sk-secret" not in out


def test_safe_exception_drops_context_copy_leak():
    e = _LeakyPgError('CONTEXT:  COPY t, line 1, column c: "sk-secret"')
    out = safe_exception(e)
    assert "sk-secret" not in out


# ---------------------------------------------------------------------------
# safe_exception — non-DB exceptions keep redacted free-text message
# ---------------------------------------------------------------------------


def test_safe_exception_non_db_keeps_redacted_message():
    out = safe_exception(ValueError("boom 'secret'"))
    assert out == "ValueError: boom '<redacted>'"


def test_safe_exception_non_db_plain_message_unchanged():
    out = safe_exception(ValueError("plain"))
    assert out == "ValueError: plain"


def test_safe_exception_non_db_empty_message_renders_class_only():
    out = safe_exception(ValueError(""))
    assert out == "ValueError"


# ---------------------------------------------------------------------------
# safe_exception — hostile / defensive cases (never raises)
# ---------------------------------------------------------------------------


def test_safe_exception_hostile_pgcode_property_does_not_raise():
    # A hostile .pgcode property that raises must not propagate; with no
    # identifiers it is treated as a non-DB exception (no [code] bracket).
    class HostilePgcodeError(Exception):
        @property
        def pgcode(self):
            raise RuntimeError("pgcode boom")

    out = safe_exception(HostilePgcodeError("msg"))
    assert isinstance(out, str)
    assert "[" not in out
    assert out == "HostilePgcodeError: msg"


def test_safe_exception_hostile_diag_property_does_not_raise():
    # A hostile .diag property that raises must not propagate.
    class HostileDiagError(Exception):
        pgcode = "23505"

        @property
        def diag(self):
            raise RuntimeError("diag boom")

    out = safe_exception(HostileDiagError("msg"))
    assert isinstance(out, str)
    # Code still rendered; no identifiers; message dropped (DB exception).
    assert out == "HostileDiagError[23505]"


def test_safe_exception_hostile_str_does_not_raise():
    # Non-DB exception whose __str__ raises → sentinel, no raise.
    class HostileError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("nope")

    out = safe_exception(HostileError())
    assert out.startswith("HostileError")
    assert "unrepresentable" in out


# ---------------------------------------------------------------------------
# safe_exception — composes with safe() (control chars + truncation)
# ---------------------------------------------------------------------------


def test_safe_exception_escapes_control_chars_in_non_db_message():
    out = safe_exception(ValueError("line1\nline2"))
    assert "\n" not in out
    assert "\\x0a" in out


def test_safe_exception_escapes_control_char_in_identifier():
    # A control char smuggled into a structured identifier is escaped by safe().
    e = _FakePgError("msg", "23505", table_name="bad\nname")
    out = safe_exception(e)
    assert "\n" not in out
    assert "\\x0a" in out
    assert out.startswith("_FakePgError[23505] table=bad")


def test_safe_exception_truncates_long_non_db_message():
    long_msg = "x" * 1000
    out = safe_exception(ValueError(long_msg), max_len=50)
    assert "truncated" in out
    assert len(out) < 200
