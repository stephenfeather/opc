"""Tests for recall_backends — tsquery sanitization and search helpers."""

from __future__ import annotations


# ==================== tsquery Sanitization ====================


class TestSanitizeTsqueryWords:
    """Ensure tsquery metacharacters are stripped before building OR queries."""

    def test_plain_words_unchanged(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["session", "affinity"]) == ["session", "affinity"]

    def test_strips_exclamation(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["hello!", "world"]) == ["hello", "world"]

    def test_strips_ampersand(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["test&exploit"]) == ["testexploit"]

    def test_strips_parentheses(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["(inject)", "normal"]) == ["inject", "normal"]

    def test_strips_pipe_operator(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["a|b"]) == []  # "ab" is len 2, filtered

    def test_strips_angle_brackets(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["<->proximity"]) == ["proximity"]

    def test_strips_colon(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["weight:A"]) == ["weightA"]

    def test_filters_short_words_after_strip(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        # "!!" becomes "" -> filtered out
        assert sanitize_tsquery_words(["!!", "valid"]) == ["valid"]

    def test_empty_input(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words([]) == []

    def test_all_metacharacters(self):
        """All tsquery operators: ! & | ( ) < > : * are stripped."""
        from scripts.core.recall_backends import sanitize_tsquery_words

        import re

        result = sanitize_tsquery_words(
            ["!not", "&and", "|or", "(group)", "<prox>", ":weight", "*prefix"]
        )
        # Each should be stripped to just alphanumeric
        for word in result:
            assert re.match(r"^[a-zA-Z0-9]+$", word), f"Unclean word: {word!r}"

    def test_preserves_digits(self):
        from scripts.core.recall_backends import sanitize_tsquery_words

        assert sanitize_tsquery_words(["error404", "http500"]) == ["error404", "http500"]


class TestBuildOrQuery:
    """Test the full or_query building pipeline including sanitization."""

    def test_injection_via_not_operator(self):
        """Query with ! should not produce tsquery NOT operator."""
        from scripts.core.recall_backends import sanitize_tsquery_words

        words = ["!secret", "data"]
        sanitized = sanitize_tsquery_words(words)
        or_query = " | ".join(sanitized)
        assert "!" not in or_query

    def test_injection_via_followed_by(self):
        """Query with <-> should not produce tsquery FOLLOWED BY operator."""
        from scripts.core.recall_backends import sanitize_tsquery_words

        words = ["a<->b", "test"]
        sanitized = sanitize_tsquery_words(words)
        or_query = " | ".join(sanitized)
        assert "<" not in or_query
        assert ">" not in or_query
