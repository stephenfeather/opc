"""Tests for scripts.core.log_safety.redact_secrets() and safe_secret().

Locks the secret-redaction contract that masks credential-shaped tokens
before captured subprocess stderr is logged or stored at rest in
``sessions.last_error``. See GitHub issue #209 (follow-up from the #98
aegis audit, ``thoughts/shared/agents/aegis/issue-98-security-audit.md``).
"""

from __future__ import annotations

from scripts.core.log_safety import redact_secrets, safe_secret

_MARKER = "<redacted-secret>"


# ---------------------------------------------------------------------------
# redact_secrets — bare prefixed tokens
# ---------------------------------------------------------------------------


def test_openai_sk_token_redacted():
    assert redact_secrets("error: key sk-abcdEFGH1234 rejected") == (
        "error: key <redacted-secret> rejected"
    )


def test_anthropic_sk_ant_token_redacted():
    # sk-ant-... is matched by the same sk- rule.
    out = redact_secrets("ANTHROPIC_API_KEY leaked: sk-ant-api03-AbCdEf123456")
    assert "sk-ant" not in out
    assert _MARKER in out


def test_github_classic_token_redacted():
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0" + "Klmno"
    assert redact_secrets(f"remote: {token}") == "remote: <redacted-secret>"


def test_github_fine_grained_token_redacted():
    token = "github_pat_" + "11ABCDE0Y" + "a" * 30
    assert redact_secrets(token) == _MARKER


def test_aws_access_key_id_redacted():
    # AKIA + exactly 16 uppercase-alnum chars.
    assert redact_secrets("aws: AKIAIOSFODNN7EXAMPLE done") == "aws: <redacted-secret> done"


def test_aws_temporary_access_key_redacted():
    assert redact_secrets("ASIAIOSFODNN7EXAMPLE") == _MARKER


def test_voyage_pa_token_redacted():
    out = redact_secrets("voyage rejected pa-AbCdEf0123456789abcdef")
    assert _MARKER in out
    assert "pa-AbCdEf" not in out


# ---------------------------------------------------------------------------
# redact_secrets — structured forms (keep context, mask value)
# ---------------------------------------------------------------------------


def test_bearer_token_redacted_keeps_scheme():
    assert redact_secrets("Authorization: Bearer abc123XYZ.def") == (
        "Authorization: Bearer <redacted-secret>"
    )


def test_connection_string_password_redacted_keeps_user_and_host():
    assert redact_secrets("postgresql://claude:s3cretPw@localhost:5432/db") == (
        "postgresql://claude:<redacted-secret>@localhost:5432/db"
    )


def test_sensitive_env_assignment_redacted_keeps_name():
    assert redact_secrets("VOYAGE_API_KEY=pa-abc123def456") == ("VOYAGE_API_KEY=<redacted-secret>")


def test_export_prefixed_secret_assignment_redacted():
    assert redact_secrets("export OPENAI_API_KEY=sk-zzzzzzzzzzzz") == (
        "export OPENAI_API_KEY=<redacted-secret>"
    )


def test_bare_credential_name_assignment_redacted():
    # The credential word may be the whole name: TOKEN=, SECRET=, PASSWORD=.
    assert redact_secrets("GH_TOKEN=abcdef123456") == "GH_TOKEN=<redacted-secret>"
    assert redact_secrets("DB_PASSWORD=hunter2value") == "DB_PASSWORD=<redacted-secret>"


# ---------------------------------------------------------------------------
# redact_secrets — env-assignment must NOT over-redact (#209 R2)
# ---------------------------------------------------------------------------


def test_word_merely_containing_key_not_redacted():
    # "monkey", "TURKEY" embed "key" but are not credential vars.
    assert redact_secrets("monkey=banana") == "monkey=banana"
    assert redact_secrets("TURKEY=roasted") == "TURKEY=roasted"


def test_compound_name_with_embedded_credential_word_not_redacted():
    # "KEYBOARD_LAYOUT" contains "KEY" and "TOKENIZER" contains "TOKEN", but
    # neither ENDS in a credential word, so neither is a secret assignment.
    assert redact_secrets("KEYBOARD_LAYOUT=us") == "KEYBOARD_LAYOUT=us"
    assert redact_secrets("TOKENIZER=bpe") == "TOKENIZER=bpe"


def test_lowercase_keyish_assignment_not_redacted():
    # Lowercase env-style names are not the uppercase env-token threat model.
    assert redact_secrets("api_key=notmatched") == "api_key=notmatched"


# ---------------------------------------------------------------------------
# redact_secrets — no false positives / passthrough
# ---------------------------------------------------------------------------


def test_plain_text_unchanged():
    s = "Traceback: ImportError in module foo at line 42"
    assert redact_secrets(s) == s


def test_uuid_not_redacted():
    s = "session 550e8400-e29b-41d4-a716-446655440000 failed"
    assert redact_secrets(s) == s


def test_hyphenated_words_not_redacted():
    # "task-master", "disk-usage" embed "sk-" but not at a word boundary.
    s = "task-master and disk-usage are fine /path/to/sk-thing"
    assert redact_secrets(s) == s


def test_empty_string_unchanged():
    assert redact_secrets("") == ""


def test_none_coerced_to_marker():
    # _coerce path: None renders as <none>, never raises.
    assert redact_secrets(None) == "<none>"


# ---------------------------------------------------------------------------
# redact_secrets — multiplicity and idempotency
# ---------------------------------------------------------------------------


def test_multiple_secrets_all_redacted():
    out = redact_secrets("sk-abcdEFGH1234 then AKIAIOSFODNN7EXAMPLE end")
    assert out == "<redacted-secret> then <redacted-secret> end"


def test_redaction_is_idempotent():
    once = redact_secrets("key sk-abcdEFGH1234 here")
    twice = redact_secrets(once)
    assert once == twice == "key <redacted-secret> here"


# ---------------------------------------------------------------------------
# safe_secret — composition of redact_secrets + safe
# ---------------------------------------------------------------------------


def test_safe_secret_redacts_and_escapes_controls():
    # A newline-injection payload riding alongside a real token: the token is
    # masked AND the control char is escaped to \x0a.
    assert safe_secret("sk-abcdEFGH1234\nFAKE LOG LINE") == ("<redacted-secret>\\x0aFAKE LOG LINE")


def test_safe_secret_none_renders_marker():
    assert safe_secret(None) == "<none>"


def test_safe_secret_empty_string():
    assert safe_secret("") == ""


def test_safe_secret_output_is_printable_ascii_only():
    out = safe_secret("tok sk-abcdEFGH1234 \x1b[31m\x9b ctrl")
    assert _MARKER in out
    # No raw ESC / C1 control survives.
    assert "\x1b" not in out
    assert "\x9b" not in out


def test_safe_secret_truncates_after_redaction():
    # Redaction runs on the full string before safe() truncates, so a secret
    # in the tail cannot survive truncation.
    payload = "x" * 600 + " sk-abcdEFGH1234"
    out = safe_secret(payload)
    assert "sk-abcdEFGH1234" not in out
    assert "truncated" in out
