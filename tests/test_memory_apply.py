"""Tests for scripts/core/memory_apply.py — promotion apply (issue #63 Phase 2a).

Dry-run is the default; writes happen only under execute=True, and only after a DB
backup. The read-only detector (memory_review.py) is untouched; all write logic lives
here. Pure planning/rendering is unit-tested; I/O handlers use tmp dirs + mocked pools.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.core.memory_apply import (
    ApplyAction,
    ApplyPlan,
    append_claude_md,
    apply_memory_file,
    backup_database,
    build_plan,
    claude_md_block,
    default_memory_dir,
    fetch_candidates_by_ids,
    fetch_promoted_ids,
    memory_entry,
    parse_ids,
    render_plan,
    route_apply_target,
    run_apply,
    slugify,
    write_provenance,
)
from scripts.core.memory_apply import _parse_args as parse_apply_args
from scripts.core.memory_review import PromotionCandidate


def _cand(id="a1", lt="CODEBASE_PATTERN", dest="MEMORY.md", recall=12, content="A useful pattern"):
    return PromotionCandidate(
        id=id, content=content, recall_count=recall, learning_type=lt, destination=dest
    )


# --- route_apply_target (pure) ---


class TestRouteApplyTarget:
    def test_codebase_pattern_to_memory_md(self):
        assert route_apply_target("CODEBASE_PATTERN") == "MEMORY.md"

    def test_architectural_decision_to_claude_md(self):
        assert route_apply_target("ARCHITECTURAL_DECISION") == "CLAUDE.md"

    def test_user_preference_deferred(self):
        # rules/ is a separate repo — not a supported target in Phase 2a.
        assert route_apply_target("USER_PREFERENCE") is None

    def test_stay_on_demand_unsupported(self):
        assert route_apply_target("WORKING_SOLUTION") is None


# --- slugify (pure) ---


class TestSlugify:
    def test_basic(self):
        assert slugify("Never use git commit; use github-agent-commit") == (
            "never-use-git-commit-use-github-agent-commit"
        )

    def test_collapses_and_trims(self):
        assert slugify("  Multiple   spaces!! ") == "multiple-spaces"

    def test_truncates_long(self):
        out = slugify("word " * 40)
        assert len(out) <= 60
        assert not out.endswith("-")

    def test_empty_falls_back(self):
        assert slugify("") == "promoted-learning"
        assert slugify("!!!") == "promoted-learning"


# --- build_plan (pure) ---


class TestBuildPlan:
    def test_routes_each_candidate(self):
        cands = [
            _cand(id="a", lt="CODEBASE_PATTERN"),
            _cand(id="b", lt="ARCHITECTURAL_DECISION", dest="CLAUDE.md"),
        ]
        plan = build_plan(cands, already_promoted=set(), dry_run=True)
        targets = {a.candidate.id: a.target for a in plan.actions}
        assert targets == {"a": "MEMORY.md", "b": "CLAUDE.md"}
        assert all(not a.skipped for a in plan.actions)

    def test_skips_already_promoted(self):
        cands = [_cand(id="a")]
        plan = build_plan(cands, already_promoted={"a"}, dry_run=True)
        assert plan.actions[0].skipped
        assert "already promoted" in plan.actions[0].skip_reason.lower()

    def test_skips_unsupported_target(self):
        cands = [_cand(id="p", lt="USER_PREFERENCE", dest="rules/")]
        plan = build_plan(cands, already_promoted=set(), dry_run=True)
        assert plan.actions[0].skipped
        assert "rules/" in plan.actions[0].skip_reason or "deferred" in plan.actions[0].skip_reason

    def test_dry_run_flag_carried(self):
        plan = build_plan([_cand()], already_promoted=set(), dry_run=True)
        assert plan.dry_run is True

    def test_applicable_actions_helper(self):
        cands = [_cand(id="a"), _cand(id="b", lt="USER_PREFERENCE", dest="rules/")]
        plan = build_plan(cands, already_promoted=set(), dry_run=True)
        applicable = [a for a in plan.actions if not a.skipped]
        assert len(applicable) == 1
        assert applicable[0].candidate.id == "a"


# --- render_plan (pure) ---


class TestRenderPlan:
    def test_shows_targets_and_dry_run_banner(self):
        plan = build_plan([_cand(id="a")], already_promoted=set(), dry_run=True)
        out = render_plan(plan)
        assert "DRY RUN" in out
        assert "MEMORY.md" in out

    def test_execute_mode_no_dry_run_banner(self):
        plan = build_plan([_cand(id="a")], already_promoted=set(), dry_run=False)
        out = render_plan(plan)
        assert "DRY RUN" not in out

    def test_shows_skipped(self):
        plan = build_plan([_cand(id="a", lt="USER_PREFERENCE", dest="rules/")], set(), True)
        out = render_plan(plan)
        assert "skip" in out.lower()


# --- memory_entry / claude_md_block (pure) ---


class TestMemoryEntry:
    def test_has_frontmatter_and_body(self):
        c = _cand(content="Recall scoping is a soft reranker boost")
        filename, body = memory_entry(c)
        assert filename.endswith(".md")
        assert body.startswith("---")
        assert "name:" in body
        assert "Recall scoping is a soft reranker boost" in body

    def test_filename_is_slug(self):
        c = _cand(content="Hello World pattern")
        filename, _ = memory_entry(c)
        assert filename == "promoted-hello-world-pattern.md"


class TestClaudeMdBlock:
    def test_renders_decision(self):
        c = _cand(lt="ARCHITECTURAL_DECISION", dest="CLAUDE.md", content="Chose X over Y")
        block = claude_md_block(c)
        assert "Chose X over Y" in block
        assert c.id[:8] in block  # provenance id for traceability


# --- async I/O handlers (mocked) ---


def _pool(rows=None):
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows or [])
    conn.execute = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=ctx)
    return pool, conn


class TestFetchPromotedIds:
    async def test_returns_id_set(self):
        pool, conn = _pool(rows=[{"id": "a"}, {"id": "b"}])
        out = await fetch_promoted_ids(pool, "opc")
        assert out == {"a", "b"}

    async def test_query_filters_promoted_to_tag(self):
        pool, conn = _pool(rows=[])
        await fetch_promoted_ids(pool, "opc")
        sql = conn.fetch.call_args.args[0]
        assert "promoted_to" in sql
        assert "LOWER(project)" in sql


class TestWriteProvenance:
    async def test_stores_structured_provenance(self):
        pool, conn = _pool()
        await write_provenance(
            pool, "abc", tier="MEMORY.md", target="/m/promoted-x.md", at="20260101-000000"
        )
        args = conn.execute.call_args.args
        sql = args[0]
        assert "UPDATE archival_memory" in sql
        assert "promoted_to" in sql
        # structured object: tier + exact target path + timestamp all bound (not interpolated)
        assert "abc" in args
        assert "MEMORY.md" in args
        assert "/m/promoted-x.md" in args
        assert "20260101-000000" in args


def test_apply_action_dataclass_shape():
    a = ApplyAction(candidate=_cand(), target="MEMORY.md", skipped=False, skip_reason=None)
    assert a.target == "MEMORY.md"
    assert isinstance(ApplyPlan(actions=[a], dry_run=True).actions, list)


class TestCliParsing:
    def test_execute_defaults_to_false(self):
        ns = parse_apply_args(["opc", "--ids", "a,b"])
        assert ns.execute is False  # dry-run is the default

    def test_execute_flag(self):
        ns = parse_apply_args(["opc", "--ids", "a", "--execute"])
        assert ns.execute is True

    def test_parse_ids_comma_list_dedups_and_trims(self):
        assert parse_ids(" a , b ,a, ", None) == ["a", "b"]

    def test_parse_ids_from_manifest(self, tmp_path):
        m = tmp_path / "ids.txt"
        m.write_text("# approved\nid1\n\nid2\n")
        assert parse_ids(None, str(m)) == ["id1", "id2"]

    def test_parse_ids_merges_sources(self, tmp_path):
        m = tmp_path / "ids.txt"
        m.write_text("id2\nid3\n")
        assert parse_ids("id1,id2", str(m)) == ["id1", "id2", "id3"]

    def test_default_memory_dir_flattens_path(self):
        d = default_memory_dir("/Users/x/opc")
        assert d.name == "memory"
        assert "-Users-x-opc" in str(d)


class TestCliMain:
    async def test_no_ids_returns_error(self, monkeypatch):
        import scripts.core.memory_apply as ma

        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        rc = await ma.main(["opc"])  # no --ids/--manifest
        assert rc == 2

    async def test_unresolved_project_returns_error(self, monkeypatch):
        import scripts.core.memory_apply as ma

        monkeypatch.setattr(ma, "canonicalize_project", lambda _p: None)
        rc = await ma.main(["   ", "--ids", "a"])
        assert rc == 2

    async def test_dry_run_through_main_writes_nothing(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        rows = [
            {"id": "a", "content": "p", "recall_count": 11, "learning_type": "CODEBASE_PATTERN"}
        ]
        pool, conn = _pool()
        conn.fetch.side_effect = [rows, []]

        async def _get_pool():
            return pool

        memory_dir = tmp_path / "mem"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("# Index\n")
        monkeypatch.setattr(ma, "get_pool", _get_pool)
        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        rc = await ma.main(
            [
                "opc",
                "--ids",
                "a",
                "--memory-dir",
                str(memory_dir),
                "--claude-md",
                str(tmp_path / "C.md"),
            ]
        )
        assert rc == 0
        assert (memory_dir / "MEMORY.md").read_text() == "# Index\n"  # untouched
        conn.execute.assert_not_called()  # no provenance write in dry-run


class TestFetchCandidatesByIds:
    async def test_builds_candidates_with_destination(self):
        rows = [
            {"id": "a", "content": "x", "recall_count": 11, "learning_type": "CODEBASE_PATTERN"}
        ]
        pool, conn = _pool(rows=rows)
        out = await fetch_candidates_by_ids(pool, "opc", ["a"])
        assert len(out) == 1
        assert out[0].destination == "MEMORY.md"

    async def test_empty_ids_returns_empty_without_query(self):
        pool, conn = _pool()
        out = await fetch_candidates_by_ids(pool, "opc", [])
        assert out == []
        conn.fetch.assert_not_called()


class TestBackupDatabase:
    def test_runs_pg_dump_and_returns_path(self, tmp_path):
        calls = {}

        def _run(cmd, **kw):
            calls["cmd"] = cmd
            # simulate pg_dump writing the file
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        dest = tmp_path / "backup.sql"
        out = backup_database(dest, run=_run)
        assert out == dest
        assert dest.exists()
        assert any("pg_dump" in str(c) for c in calls["cmd"])

    def test_raises_when_pg_dump_fails(self, tmp_path):
        def _run(cmd, **kw):
            return MagicMock(returncode=1)

        with pytest.raises(RuntimeError, match="backup"):
            backup_database(tmp_path / "b.sql", run=_run)


class TestApplyMemoryFile:
    def test_creates_file_and_appends_index(self, tmp_path):
        memory_dir = tmp_path
        (memory_dir / "MEMORY.md").write_text("# Memory Index\n")
        c = _cand(content="Recall scoping is a soft boost")
        ok = apply_memory_file(memory_dir, c)
        assert ok
        created = list(memory_dir.glob("promoted-*.md"))
        assert len(created) == 1
        assert "Recall scoping" in created[0].read_text()
        assert "promoted-recall" in (memory_dir / "MEMORY.md").read_text()

    def test_idempotent_no_duplicate_when_reapplied(self, tmp_path):
        memory_dir = tmp_path
        (memory_dir / "MEMORY.md").write_text("# Memory Index\n")
        c = _cand(content="Same content here")
        assert apply_memory_file(memory_dir, c)
        assert apply_memory_file(memory_dir, c)  # present → no error
        assert (memory_dir / "MEMORY.md").read_text().count("promoted-same-content-here") == 1
        assert len(list(memory_dir.glob("promoted-*.md"))) == 1

    def test_reconciles_missing_index_pointer(self, tmp_path):
        # Regression (round 1): a prior partial apply left the file but no index pointer.
        memory_dir = tmp_path
        (memory_dir / "MEMORY.md").write_text("# Memory Index\n")
        c = _cand(content="Partial apply learning")
        # simulate: file exists (for this candidate) but pointer was never written
        fname, body = memory_entry(c)
        (memory_dir / fname).write_text(body)
        assert "promoted-partial" not in (memory_dir / "MEMORY.md").read_text()
        ok = apply_memory_file(memory_dir, c)
        assert ok
        assert "promoted-partial-apply-learning" in (memory_dir / "MEMORY.md").read_text()

    def test_slug_collision_uses_distinct_filename(self, tmp_path):
        # Regression (round 1): two different learnings with the same slug must not shadow.
        memory_dir = tmp_path
        (memory_dir / "MEMORY.md").write_text("# Memory Index\n")
        c1 = _cand(id="11111111-aaaa", content="Same Slug Text")
        c2 = _cand(id="22222222-bbbb", content="Same Slug Text")
        assert apply_memory_file(memory_dir, c1) is not None
        assert apply_memory_file(memory_dir, c2) is not None
        files = sorted(f.name for f in memory_dir.glob("promoted-*.md"))
        assert len(files) == 2  # neither candidate shadowed the other
        # both source ids preserved across the two files
        blob = "".join((memory_dir / f).read_text() for f in files)
        assert "11111111-aaaa" in blob and "22222222-bbbb" in blob


class TestAppendClaudeMd:
    def test_appends_under_section(self, tmp_path):
        path = tmp_path / "CLAUDE.md"
        path.write_text("# Project\n\nSome content.\n")
        c = _cand(
            id="dec12345-0000", lt="ARCHITECTURAL_DECISION", dest="CLAUDE.md", content="Chose X"
        )
        assert append_claude_md(path, c)
        text = path.read_text()
        assert "Promoted Decisions" in text
        assert "Chose X" in text

    def test_idempotent_on_exact_marker(self, tmp_path):
        path = tmp_path / "CLAUDE.md"
        path.write_text("# Project\n")
        c = _cand(
            id="dec12345-0000", lt="ARCHITECTURAL_DECISION", dest="CLAUDE.md", content="Chose X"
        )
        assert append_claude_md(path, c)
        assert append_claude_md(path, c)  # present (exact marker) → no duplicate
        assert path.read_text().count("Chose X") == 1

    def test_substring_id_does_not_false_suppress(self, tmp_path):
        # Regression (round 1): an 8-char prefix appearing in unrelated text must NOT be
        # treated as "already promoted" — only the exact full-id marker counts.
        path = tmp_path / "CLAUDE.md"
        c = _cand(
            id="dec12345-9999",
            lt="ARCHITECTURAL_DECISION",
            dest="CLAUDE.md",
            content="Real decision",
        )
        path.write_text(f"# Project\n\nUnrelated mention of {c.id[:8]} in prose.\n")
        wrote = append_claude_md(path, c)
        assert wrote
        assert "Real decision" in path.read_text()


class TestRunApply:
    def _setup(self, tmp_path):
        memory_dir = tmp_path / "mem"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("# Memory Index\n")
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Project\n")
        backup_dir = tmp_path / "backups"
        return memory_dir, claude_md, backup_dir

    async def test_dry_run_writes_nothing_and_no_backup(self, tmp_path, monkeypatch):
        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        rows = [
            {"id": "a", "content": "p", "recall_count": 11, "learning_type": "CODEBASE_PATTERN"}
        ]
        pool, conn = _pool()
        # fetch #1 = candidates by id, fetch #2 = already-promoted ids (none).
        conn.fetch.side_effect = [rows, []]

        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        result = await run_apply(
            pool,
            "opc",
            ["a"],
            execute=False,
            memory_dir=memory_dir,
            claude_md_path=claude_md,
            backup_dir=backup_dir,
            timestamp="20260101-000000",
            run=_run,
        )
        assert result.plan.dry_run is True
        assert result.applied == []
        assert result.backup_path is None
        assert backup_called is False
        # no MEMORY.md mutation, no provenance write
        assert (memory_dir / "MEMORY.md").read_text() == "# Memory Index\n"
        conn.execute.assert_not_called()

    async def test_execute_backs_up_then_writes_and_tags(self, tmp_path):
        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        rows = [
            {
                "id": "a",
                "content": "Good pattern",
                "recall_count": 11,
                "learning_type": "CODEBASE_PATTERN",
            }
        ]
        pool, conn = _pool()
        conn.fetch.side_effect = [rows, []]  # candidates, then no already-promoted
        order = []

        def _run(cmd, **kw):
            order.append("backup")
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        orig_execute = conn.execute

        async def _exec(*a, **k):
            order.append("provenance")
            return await orig_execute(*a, **k)

        conn.execute = _exec

        result = await run_apply(
            pool,
            "opc",
            ["a"],
            execute=True,
            memory_dir=memory_dir,
            claude_md_path=claude_md,
            backup_dir=backup_dir,
            timestamp="20260101-000000",
            run=_run,
        )
        assert result.backup_path is not None and result.backup_path.exists()
        assert len(result.applied) == 1
        assert "Good pattern" in (memory_dir / "MEMORY.md").read_text() or any(
            f.name.startswith("promoted-") for f in memory_dir.glob("promoted-*.md")
        )
        # backup happens before the provenance write
        assert order[0] == "backup"
        assert "provenance" in order

    async def test_provenance_not_written_when_writer_reports_failure(self, tmp_path, monkeypatch):
        # Regression (round 1): tag the row only after the artifact is confirmed present.
        import scripts.core.memory_apply as ma

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        rows = [
            {"id": "a", "content": "P", "recall_count": 11, "learning_type": "CODEBASE_PATTERN"}
        ]
        pool, conn = _pool()
        conn.fetch.side_effect = [rows, []]

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        monkeypatch.setattr(ma, "apply_memory_file", lambda *a, **k: None)
        result = await run_apply(
            pool,
            "opc",
            ["a"],
            execute=True,
            memory_dir=memory_dir,
            claude_md_path=claude_md,
            backup_dir=backup_dir,
            timestamp="20260101-000000",
            run=_run,
        )
        assert result.applied == []
        conn.execute.assert_not_called()  # no provenance tag without a confirmed write

    async def test_execute_snapshots_files_before_writing(self, tmp_path):
        # Round 2: the file side must be recoverable too — MEMORY.md/CLAUDE.md are copied
        # to the backup dir before any mutation.
        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        rows = [
            {"id": "a", "content": "Pat", "recall_count": 11, "learning_type": "CODEBASE_PATTERN"}
        ]
        pool, conn = _pool()
        conn.fetch.side_effect = [rows, []]

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        await run_apply(
            pool,
            "opc",
            ["a"],
            execute=True,
            memory_dir=memory_dir,
            claude_md_path=claude_md,
            backup_dir=backup_dir,
            timestamp="20260101-000000",
            run=_run,
        )
        backups = {p.name for p in backup_dir.glob("*.bak")}
        assert any("MEMORY.md" in n for n in backups)
        assert any("CLAUDE.md" in n for n in backups)


class TestBackupFiles:
    def test_copies_existing_skips_missing(self, tmp_path):
        from scripts.core.memory_apply import backup_files

        present = tmp_path / "MEMORY.md"
        present.write_text("hello")
        missing = tmp_path / "CLAUDE.md"  # not created
        out = backup_files(tmp_path / "bk", "20260101-000000", [present, missing])
        assert len(out) == 1
        assert out[0].read_text() == "hello"


class TestExecutePathGuard:
    async def test_refuses_execute_on_project_path_mismatch(self, monkeypatch):
        # Round 2: requested project differs from the working tree -> fail closed unless
        # explicit write paths are given.
        import scripts.core.memory_apply as ma

        monkeypatch.setattr(ma, "canonicalize_project", lambda p: "other-project")
        monkeypatch.setattr(ma, "project_from_path", lambda p: "opc")
        called = False

        async def _get_pool():
            nonlocal called
            called = True
            return MagicMock()

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        rc = await ma.main(["other-project", "--ids", "a", "--execute"])
        assert rc == 2
        assert called is False  # bailed before touching the DB

    async def test_mismatch_allowed_with_explicit_paths(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        monkeypatch.setattr(ma, "canonicalize_project", lambda p: "other-project")
        monkeypatch.setattr(ma, "project_from_path", lambda p: "opc")
        memory_dir = tmp_path / "mem"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("# Index\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [[], []]  # no candidates -> dry plan, no writes

        async def _get_pool():
            return pool

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        # explicit paths provided -> guard passes; dry-run (no --execute on the write side)
        rc = await ma.main(
            [
                "other-project",
                "--ids",
                "a",
                "--memory-dir",
                str(memory_dir),
                "--claude-md",
                str(tmp_path / "C.md"),
            ]
        )
        assert rc == 0
