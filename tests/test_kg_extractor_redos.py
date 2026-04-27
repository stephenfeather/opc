"""ReDoS regression tests for ``scripts/core/kg_extractor`` (issue #120).

Per the aegis audit at ``thoughts/shared/agents/aegis/kg_extractor_redos_audit.md``
the only pattern with non-linear worst-case behavior is ``_RE_FILE_PATH`` branch
B (``[\\w.-]+\\.(ext|...)``). After tightening branch B to remove ``.`` from the
prefix character class the backtracking is reduced to linear, but we keep these
tests as a permanent budget check to catch any future regressions.

The 16 parametrized payloads come directly from the audit's "Test inputs to add"
section (audit groups 1, 7, and 10 each contribute two payloads, so the visible
count is 16). Each one exercises an adversarial dotty/slashy/underscore pattern
at or near the ``_KG_QUERY_EXTRACTION_MAX_CHARS = 4096`` cap and must complete
inside the per-call wall-clock budget.

The module also includes a focused dotfile-with-known-extension regression test
to guard against the issue caught by AI review on PR #127: an earlier hardening
of branch B accidentally rejected inputs like ``.bashrc.log`` because the prefix
character class no longer permitted a leading dot.
"""

from __future__ import annotations

import time

import pytest

from scripts.core.kg_extractor import extract_entities, extract_relations

# Wall-clock budget per call. The fixed regex runs each payload in well under
# 50ms on developer hardware; 0.5s is a generous CI ceiling that still trips
# convincingly if backtracking explodes.
REDOS_BUDGET_SECONDS = 0.5


@pytest.mark.parametrize(
    "payload",
    [
        # 1. _RE_FILE_PATH branch B worst case: many dots, no valid extension.
        "a" + ".a" * 1500 + ".xyz",
        "x" + ".x" * 1500 + ".notanext",
        # 2. _RE_FILE_PATH branch A: many slashes (sanity check, was already linear).
        ("a/" * 2000) + "b",
        # 3. Mixed dots and slashes.
        ("a." * 1000) + "/" + ("b." * 1000) + "c",
        # 4. _RE_ENV_VAR: many underscores (sanity check, already linear).
        "A" + "_A" * 2000,
        # 5. _RE_ERROR_TYPE: long word with no Error/Exception/Warning suffix.
        "A" * 4000,
        # 6. _RE_ERROR_TYPE: long word ending almost-but-not-quite in suffix.
        "A" * 3990 + "Erro",
        # 7. _RE_QUOTED: unbalanced backticks.
        "`" + "a" * 4000,
        "`" + "a" * 4000 + "`",
        # 8. _RE_PYTHON_IMPORT: long dotted path.
        "import " + ".".join(["a"] * 1000),
        # 9. Pathological mix of quote-like boundaries.
        "(" * 1000 + "a.b" + ")" * 1000,
        # 10. Whitespace-only and zero-width-space inputs.
        " " * 4000,
        "\u200B" * 4000,  # U+200B ZERO WIDTH SPACE — explicit escape, not a literal
        # 11. Long line with no whitespace (stress MULTILINE alternation).
        "x" * 4000,
        # 12. Many short would-be sentences (stress _RE_SENTENCE).
        "A. " * 1000,
        # 13. Deeply nested punctuation triggering boundary alternation.
        "[" * 1000 + "a.py" + "]" * 1000,
    ],
)
def test_no_redos(payload: str) -> None:
    """Each adversarial payload must complete within the wall-clock budget."""
    start = time.perf_counter()
    entities = extract_entities(payload)
    extract_relations(payload, entities)
    elapsed = time.perf_counter() - start
    assert elapsed < REDOS_BUDGET_SECONDS, (
        f"extraction took {elapsed:.3f}s on {len(payload)}-char input"
    )


def test_query_extraction_cap_constant() -> None:
    """The 4096-char input cap referenced by the audit must remain in place.

    The cap is the defense-in-depth that keeps even the polynomial worst-case
    bounded. Removing or raising it would invalidate the audit and require a
    re-evaluation of the regex hardening.
    """
    from scripts.core.recall_learnings import _KG_QUERY_EXTRACTION_MAX_CHARS

    assert _KG_QUERY_EXTRACTION_MAX_CHARS == 4096, (
        "kg query extraction cap must stay at 4096 chars (see audit)."
    )


@pytest.mark.parametrize(
    "dotfile",
    [
        ".bashrc.log",   # leading-dot config + known extension
        ".envrc.json",
        ".npmrc.yaml",
        ".config.toml",
    ],
)
def test_dotfile_with_known_extension_matches(dotfile: str) -> None:
    """Regression for PR #127: dotfiles with a known extension must still match.

    The previous tightening of ``_RE_FILE_PATH`` branch B replaced the prefix
    class ``[\\w.-]+`` with ``[\\w-]+``, which rejected inputs starting with
    ``.`` (e.g. ``.bashrc.log``). The fix restores leading-dot support via an
    optional ``\\.?`` anchor while preserving the deterministic-splitter
    property the audit depends on.
    """
    payload = f"see {dotfile} for details"
    entities = extract_entities(payload)
    # ``display_name`` preserves the original surface form (including the
    # leading dot). ``name`` is canonicalized via ``lstrip("./")`` for file
    # entities, so it intentionally drops the leading dot.
    display_names = {e.display_name for e in entities if e.entity_type == "file"}
    assert dotfile in display_names, (
        f"expected dotfile {dotfile!r} to be extracted as a file entity; "
        f"got file display_names={sorted(display_names)!r}"
    )
