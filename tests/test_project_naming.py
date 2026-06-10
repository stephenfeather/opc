"""Tests for scripts/core/project_naming.py (issue #130 audit fix 3).

The audit found 40 fragmented project values in archival_memory: case
variants (Pharmacokinetics-Grapher / pharmacokinetics-grapher), flattened
path artifacts (-Users-stephenfeather-Operations-DigitalOcean), and alias
pairs (calebs-hospital / 2026-calebs-hospital). canonicalize_project is
the single source of truth applied at store time, recall-context time,
and by the one-time migration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.core.project_naming import (  # noqa: E402
    PROJECT_ALIASES,
    canonicalize_project,
)


class TestCanonicalizeProject:
    """Pure function: raw project value -> canonical form (or None)."""

    def test_simple_name_lowercased(self):
        assert canonicalize_project("Pharmacokinetics-Grapher") == (
            "pharmacokinetics-grapher"
        )

    def test_already_canonical_unchanged(self):
        assert canonicalize_project("opc") == "opc"

    def test_none_passes_through(self):
        assert canonicalize_project(None) is None

    def test_empty_and_whitespace_become_none(self):
        assert canonicalize_project("") is None
        assert canonicalize_project("   ") is None

    def test_surrounding_whitespace_stripped(self):
        assert canonicalize_project("  opc  ") == "opc"

    def test_flattened_home_path_artifact_stripped(self):
        # '-Users-<user>-Operations-DigitalOcean' is a flattened absolute
        # path; the home prefix is dropped, the remainder canonicalized.
        assert canonicalize_project(
            "-Users-stephenfeather-Operations-DigitalOcean"
        ) == "digitalocean"

    def test_known_alias_collapsed(self):
        assert canonicalize_project("operations-digitalocean") == "digitalocean"
        assert canonicalize_project("calebs-hospital") == "2026-calebs-hospital"

    def test_alias_lookup_happens_after_lowercasing(self):
        assert canonicalize_project("Operations-DigitalOcean") == "digitalocean"

    def test_unresolved_sentinel_preserved(self):
        # '_unresolved' is the daemon's explicit unknown marker, not a name.
        assert canonicalize_project("_unresolved") == "_unresolved"

    def test_idempotent(self):
        for raw in (
            "Pharmacokinetics-Grapher",
            "-Users-stephenfeather-Operations-DigitalOcean",
            "calebs-hospital",
            "opc",
        ):
            once = canonicalize_project(raw)
            assert canonicalize_project(once) == once

    def test_alias_map_values_are_canonical(self):
        # Every alias target must itself be a fixed point, or the map
        # would produce different results on repeated application.
        for target in PROJECT_ALIASES.values():
            assert canonicalize_project(target) == target


class TestProjectMatchCaseInsensitive:
    """Reranker project_match must not miss on pure case variants —
    stored values predate canonicalization."""

    def test_exact_match_after_case_fold(self):
        from scripts.core.reranker import RecallContext, project_match

        ctx = RecallContext(project="Pharmacokinetics-Grapher")
        result = {"metadata": {"project": "pharmacokinetics-grapher"}}
        assert project_match(result, ctx) == 1.0

    def test_substring_match_after_case_fold(self):
        from scripts.core.reranker import RecallContext, project_match

        ctx = RecallContext(project="BinBrain")
        result = {"metadata": {"project": "binbrain-ios"}}
        assert project_match(result, ctx) == 0.5

    def test_no_match_unchanged(self):
        from scripts.core.reranker import RecallContext, project_match

        ctx = RecallContext(project="opc")
        result = {"metadata": {"project": "binbrain"}}
        assert project_match(result, ctx) == 0.0


class TestStoreTimeCanonicalization:
    """resolve_project_for_store applies canonicalization to CLI/env input."""

    def test_explicit_arg_canonicalized(self):
        from scripts.core.project_naming import resolve_project_for_store

        assert resolve_project_for_store(
            "Operations-DigitalOcean", env_project_dir="",
        ) == "digitalocean"

    def test_env_basename_used_when_no_arg(self):
        from scripts.core.project_naming import resolve_project_for_store

        assert resolve_project_for_store(
            None, env_project_dir="/Users/stephenfeather/opc",
        ) == "opc"

    def test_env_basename_canonicalized(self):
        from scripts.core.project_naming import resolve_project_for_store

        assert resolve_project_for_store(
            None, env_project_dir="/Users/x/Development/2026-Calebs-Hospital",
        ) == "2026-calebs-hospital"

    def test_nothing_available_returns_none(self):
        from scripts.core.project_naming import resolve_project_for_store

        assert resolve_project_for_store(None, env_project_dir="") is None


class TestBuildNormalizationPlan:
    """Migration planning is pure: stored values in, (old, new) pairs out."""

    def test_only_changed_values_in_plan(self):
        from scripts.migrations.normalize_project_values import (
            build_normalization_plan,
        )

        plan = build_normalization_plan(
            ["opc", "Pharmacokinetics-Grapher", "binbrain"]
        )
        assert plan == [("Pharmacokinetics-Grapher", "pharmacokinetics-grapher")]

    def test_audit_fragments_collapse(self):
        from scripts.migrations.normalize_project_values import (
            build_normalization_plan,
        )

        plan = dict(
            build_normalization_plan(
                [
                    "DigitalOcean",
                    "Operations-DigitalOcean",
                    "operations-digitalocean",
                    "-Users-stephenfeather-Operations-DigitalOcean",
                    "calebs-hospital",
                    "_unresolved",
                ]
            )
        )
        assert plan["DigitalOcean"] == "digitalocean"
        assert plan["Operations-DigitalOcean"] == "digitalocean"
        assert plan["operations-digitalocean"] == "digitalocean"
        assert plan["-Users-stephenfeather-Operations-DigitalOcean"] == (
            "digitalocean"
        )
        assert plan["calebs-hospital"] == "2026-calebs-hospital"
        assert "_unresolved" not in plan

    def test_empty_input_empty_plan(self):
        from scripts.migrations.normalize_project_values import (
            build_normalization_plan,
        )

        assert build_normalization_plan([]) == []

    def test_metadata_sync_phase_is_guarded_and_idempotent(self):
        """Review rounds 1-2: the migration must align metadata.project
        (the field the reranker historically reads) with the column —
        rewriting stale keys AND creating missing ones (backfilled rows)."""
        from scripts.migrations.normalize_project_values import (
            METADATA_SYNC_COUNT_SQL,
            METADATA_SYNC_SQL,
        )

        assert "jsonb_set" in METADATA_SYNC_SQL
        assert "COALESCE(metadata, '{}'::jsonb)" in METADATA_SYNC_SQL
        for sql in (METADATA_SYNC_SQL, METADATA_SYNC_COUNT_SQL):
            # Only touch rows that disagree — re-runs match zero rows.
            assert "IS DISTINCT FROM" in sql


class TestWritePathsCanonicalize:
    """Review round 1: every writer and exact-match reader must share the
    canonical form, or the migration re-fragments immediately."""

    def test_daemon_normalize_project_canonicalizes(self):
        from scripts.core.memory_daemon_core import _normalize_project

        assert _normalize_project(
            "/Users/x/Operations/DigitalOcean"
        ) == "digitalocean"

    def test_daemon_normalize_project_worktree_canonicalizes(self):
        from scripts.core.memory_daemon_core import _normalize_project

        assert _normalize_project(
            "/Users/x/Operations/DigitalOcean/.worktrees/fix-123"
        ) == "digitalocean"

    def test_push_learnings_args_canonicalized(self):
        from scripts.core.push_learnings import parse_args

        parsed = parse_args(["--project", "Operations-DigitalOcean"])
        assert parsed["project"] == "digitalocean"


class TestProjectFromPath:
    """Review round 2: store-time path resolution must be worktree-aware,
    or worktree sessions re-fragment project values immediately."""

    def test_plain_repo_path(self):
        from scripts.core.project_naming import project_from_path

        assert project_from_path("/Users/x/opc") == "opc"

    def test_dot_worktrees_resolves_to_parent_repo(self):
        from scripts.core.project_naming import project_from_path

        assert project_from_path(
            "/Users/x/opc/.worktrees/issue-130-normalize"
        ) == "opc"

    def test_claude_worktrees_resolves_to_parent_repo(self):
        from scripts.core.project_naming import project_from_path

        assert project_from_path(
            "/Users/x/opc/.claude/worktrees/issue-131-circuit-breaker"
        ) == "opc"

    def test_worktree_of_aliased_project_canonicalizes(self):
        from scripts.core.project_naming import project_from_path

        assert project_from_path(
            "/Users/x/Operations/DigitalOcean/.worktrees/fix-1"
        ) == "digitalocean"

    def test_none_and_empty(self):
        from scripts.core.project_naming import project_from_path

        assert project_from_path(None) is None
        assert project_from_path("") is None

    def test_resolve_for_store_is_worktree_aware(self):
        from scripts.core.project_naming import resolve_project_for_store

        assert resolve_project_for_store(
            None, env_project_dir="/Users/x/opc/.worktrees/agent-130-normalize",
        ) == "opc"


class TestBackfillProducesCanonicalNames:
    """Review round 2: backfill_project_column reruns (e.g. on remaining
    NULL rows) must not reintroduce deprecated aliases."""

    def test_session_prefix_inference_is_canonical(self):
        from scripts.core.project_naming import canonicalize_project
        from scripts.migrations.backfill_project_column import (
            SESSION_PREFIX_TO_PROJECT,
            infer_from_session_id,
        )

        for prefix in SESSION_PREFIX_TO_PROJECT:
            inferred = infer_from_session_id(f"{prefix}-some-session")
            assert inferred == canonicalize_project(inferred), (
                f"prefix {prefix!r} infers non-canonical {inferred!r}"
            )

    def test_deprecated_alias_prefix_maps_to_canonical(self):
        from scripts.migrations.backfill_project_column import (
            infer_from_session_id,
        )

        assert infer_from_session_id("calebs-hospital-abc") == (
            "2026-calebs-hospital"
        )
        assert infer_from_session_id("2026-calebs-hospital-abc") == (
            "2026-calebs-hospital"
        )

    def test_tag_inference_is_canonical(self):
        from scripts.core.project_naming import canonicalize_project
        from scripts.migrations.backfill_project_column import (
            TAG_TO_PROJECT,
            infer_from_tags,
        )

        for required_tags, _project in TAG_TO_PROJECT:
            inferred = infer_from_tags(sorted(required_tags))
            assert inferred == canonicalize_project(inferred), (
                f"tags {required_tags!r} infer non-canonical {inferred!r}"
            )
