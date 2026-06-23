"""Tests for scripts/core/memory_apply.py — promotion apply (issue #63 Phase 2a).

Dry-run is the default; writes happen only under execute=True, and only after a DB
backup. The read-only detector (memory_review.py) is untouched; all write logic lives
here. Pure planning/rendering is unit-tested; I/O handlers use tmp dirs + mocked pools.
"""

import datetime as _dt
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_DT_A = _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)
_DT_B = _dt.datetime(2026, 2, 1, tzinfo=_dt.UTC)

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
    validate_ids,
    write_provenance,
)
from scripts.core.memory_apply import _parse_args as parse_apply_args
from scripts.core.memory_review import PromotionCandidate

_UUID = "11111111-1111-1111-1111-111111111111"
_UUID2 = "22222222-2222-2222-2222-222222222222"


def _cand(id="a1", lt="CODEBASE_PATTERN", dest="MEMORY.md", recall=12, content="A useful pattern"):
    return PromotionCandidate(
        id=id, content=content, recall_count=recall, learning_type=lt, destination=dest
    )


# --- route_apply_target (pure) ---


def _mrow(id="a", recall=0, created="2026-01-01", superseded_by=None):
    import datetime as _dt

    from scripts.core.memory_review import MergeRow

    y, m, d = (int(x) for x in created.split("-"))
    return MergeRow(
        id=id,
        recall_count=recall,
        created_at=_dt.datetime(y, m, d, tzinfo=_dt.UTC),
        superseded_by=superseded_by,
    )


class TestSelectMergeKeeper:
    """Pure keeper selection: higher recall, then older created_at, then smaller id."""

    def test_higher_recall_wins(self):
        from scripts.core.memory_apply import select_merge_keeper

        a = _mrow(id="aaaa", recall=3)
        b = _mrow(id="bbbb", recall=9)
        keeper, loser = select_merge_keeper(a, b)
        assert keeper.id == "bbbb"
        assert loser.id == "aaaa"

    def test_recall_tie_older_created_at_wins(self):
        from scripts.core.memory_apply import select_merge_keeper

        older = _mrow(id="zzzz", recall=5, created="2026-01-01")
        newer = _mrow(id="aaaa", recall=5, created="2026-06-01")
        keeper, loser = select_merge_keeper(newer, older)
        assert keeper.id == "zzzz"  # older keeper despite larger id
        assert loser.id == "aaaa"

    def test_full_tie_smaller_id_wins(self):
        from scripts.core.memory_apply import select_merge_keeper

        a = _mrow(id="aaaa", recall=5, created="2026-01-01")
        b = _mrow(id="bbbb", recall=5, created="2026-01-01")
        keeper, loser = select_merge_keeper(b, a)  # order-independent
        assert keeper.id == "aaaa"
        assert loser.id == "bbbb"

    def test_pure_does_not_mutate_inputs(self):
        from scripts.core.memory_apply import select_merge_keeper

        a = _mrow(id="aaaa", recall=3)
        b = _mrow(id="bbbb", recall=9)
        select_merge_keeper(a, b)
        assert a.recall_count == 3 and b.recall_count == 9  # frozen, untouched

    def test_raises_when_keeper_side_already_superseded(self):
        from scripts.core.memory_apply import MergeKeeperError, select_merge_keeper

        a = _mrow(id="aaaa", recall=9, superseded_by="cccc")  # would-be keeper, dead
        b = _mrow(id="bbbb", recall=3)
        with pytest.raises(MergeKeeperError):
            select_merge_keeper(a, b)

    def test_raises_when_loser_side_already_superseded(self):
        from scripts.core.memory_apply import MergeKeeperError, select_merge_keeper

        a = _mrow(id="aaaa", recall=9)
        b = _mrow(id="bbbb", recall=3, superseded_by="cccc")
        with pytest.raises(MergeKeeperError):
            select_merge_keeper(a, b)


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
    conn.execute = AsyncMock(return_value="UPDATE 1")  # asyncpg status string
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
            pool,
            "abc",
            project="opc",
            tier="MEMORY.md",
            target="/m/promoted-x.md",
            at="20260101-000000",
        )
        args = conn.execute.call_args.args
        sql = args[0]
        assert "UPDATE archival_memory" in sql
        assert "promoted_to" in sql
        assert "superseded_by IS NULL" in sql  # only tags a current row
        # structured object: tier + exact target path + timestamp + project all bound
        assert "abc" in args
        assert "MEMORY.md" in args
        assert "/m/promoted-x.md" in args
        assert "20260101-000000" in args
        assert "opc" in args

    async def test_raises_when_no_row_updated(self):
        # Round 3: a superseded/removed row -> UPDATE 0 must raise, not silently succeed.
        pool, conn = _pool()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        with pytest.raises(RuntimeError, match="exactly one row"):
            await write_provenance(
                pool, "gone", project="opc", tier="MEMORY.md", target="/m/x.md", at="t"
            )


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


class TestMergeCliParsing:
    def test_merge_flag_defaults_false(self):
        ns = parse_apply_args(["opc", "--ids", "a"])
        assert ns.merge is False

    def test_merge_flag_set(self):
        ns = parse_apply_args(["opc", "--merge", "--pair", f"{_UUID}:{_UUID2}"])
        assert ns.merge is True

    def test_pair_is_repeatable(self):
        ns = parse_apply_args(
            ["opc", "--merge", "--pair", f"{_UUID}:{_UUID2}", "--pair", f"{_UUID2}:{_UUID}"]
        )
        assert ns.pair == [f"{_UUID}:{_UUID2}", f"{_UUID2}:{_UUID}"]

    def test_parse_pairs_splits_on_colon(self):
        from scripts.core.memory_apply import parse_pairs

        assert parse_pairs([f"{_UUID}:{_UUID2}"]) == [(_UUID, _UUID2)]

    def test_parse_pairs_rejects_malformed(self):
        from scripts.core.memory_apply import parse_pairs

        with pytest.raises(ValueError, match="pair"):
            parse_pairs(["only-one-id"])

    def test_parse_pairs_validates_uuids(self):
        from scripts.core.memory_apply import parse_pairs

        with pytest.raises(ValueError):
            parse_pairs(["not-a-uuid:also-bad"])


class TestMergeCliMain:
    async def test_merge_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        rows = [
            {"id": _UUID, "recall_count": 3, "created_at": _DT_A, "superseded_by": None},
            {"id": _UUID2, "recall_count": 9, "created_at": _DT_B, "superseded_by": None},
        ]
        pool, conn = _pool()
        conn.fetch.side_effect = [rows]
        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        async def _get_pool():
            return pool

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        rc = await ma.main(
            ["opc", "--merge", "--pair", f"{_UUID}:{_UUID2}", "--backup-dir", str(tmp_path)]
        )
        assert rc == 0
        assert backup_called is False
        conn.execute.assert_not_called()  # no supersede, no backup in dry-run

    async def test_merge_execute_backs_up_then_supersedes(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        rows = [
            {"id": _UUID, "recall_count": 3, "created_at": _DT_A, "superseded_by": None},
            {"id": _UUID2, "recall_count": 9, "created_at": _DT_B, "superseded_by": None},
        ]
        pool, conn = _pool()
        conn.fetch.side_effect = [rows]
        order = []

        def _backup(dest, **kw):
            order.append("backup")
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("-- dump")
            return Path(dest)

        async def _exec(*a, **k):
            order.append("supersede")
            return "UPDATE 1"

        conn.execute = _exec

        async def _get_pool():
            return pool

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        monkeypatch.setattr(ma, "backup_database", _backup)
        rc = await ma.main(
            [
                "opc",
                "--merge",
                "--pair",
                f"{_UUID}:{_UUID2}",
                "--execute",
                "--backup-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        assert order and order[0] == "backup"
        assert "supersede" in order


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
            {"id": _UUID, "content": "p", "recall_count": 11, "learning_type": "CODEBASE_PATTERN"}
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
                _UUID,
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
            return MagicMock(returncode=1, stderr=b"could not connect")

        dest = tmp_path / "b.sql"
        with pytest.raises(RuntimeError, match="backup"):
            backup_database(dest, run=_run)
        assert not dest.exists()  # no misleading partial backup left behind

    def test_spawn_failure_leaves_no_backup_file(self, tmp_path):
        # Round 3: docker missing -> subprocess raises before returncode; dest must not exist.
        def _run(cmd, **kw):
            raise FileNotFoundError("docker not found")

        dest = tmp_path / "b.sql"
        with pytest.raises(FileNotFoundError):
            backup_database(dest, run=_run)
        assert not dest.exists()
        assert list(tmp_path.glob("*.tmp")) == []  # temp file cleaned up


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

    def test_inserts_under_section_when_other_sections_follow(self, tmp_path):
        # Review finding: with a section after Promoted Decisions, the block must land
        # under the right header, not at EOF.
        path = tmp_path / "CLAUDE.md"
        path.write_text("# P\n\n## Promoted Decisions\n\n## Later Section\nstuff\n")
        c = _cand(
            id="cccccccc-0000",
            lt="ARCHITECTURAL_DECISION",
            dest="CLAUDE.md",
            content="New decision",
        )
        append_claude_md(path, c)
        text = path.read_text()
        # the new block appears before the later section, i.e. under Promoted Decisions
        assert text.index("New decision") < text.index("## Later Section")

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

    async def test_provenance_update_zero_goes_to_failed_not_applied(self, tmp_path):
        # Round 3: file written but UPDATE 0 (row superseded) -> item is reported failed,
        # not applied, and the run returns the reason.
        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        rows = [
            {"id": "a", "content": "Pat", "recall_count": 11, "learning_type": "CODEBASE_PATTERN"}
        ]
        pool, conn = _pool()
        conn.fetch.side_effect = [rows, []]
        conn.execute = AsyncMock(return_value="UPDATE 0")  # provenance tags nothing

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

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
        assert len(result.failed) == 1
        assert result.failed[0][0] == "a"

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


class TestRunMergeApply:
    """Merge-supersede apply path: same safety envelope as promotion, reason='merge'."""

    _A = "11111111-1111-1111-1111-111111111111"
    _B = "22222222-2222-2222-2222-222222222222"

    def _merge_rows(self, recall_a=3, recall_b=9, sup_a=None, sup_b=None):
        # row_b wins keeper (higher recall) by default.
        return [
            {
                "id": self._A,
                "recall_count": recall_a,
                "created_at": _DT_A,
                "superseded_by": sup_a,
            },
            {
                "id": self._B,
                "recall_count": recall_b,
                "created_at": _DT_B,
                "superseded_by": sup_b,
            },
        ]

    async def test_dry_run_writes_nothing_and_no_backup(self, tmp_path):
        from scripts.core.memory_apply import run_merge_apply

        pool, conn = _pool(rows=self._merge_rows())
        backup_dir = tmp_path / "backups"
        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        result = await run_merge_apply(
            pool,
            "opc",
            [(self._A, self._B)],
            execute=False,
            backup_dir=backup_dir,
            timestamp="20260101-000000",
            run=_run,
        )
        assert result.plan.dry_run is True
        assert result.applied == []
        assert result.backup_path is None
        assert backup_called is False
        conn.execute.assert_not_called()  # no supersede UPDATE in dry-run

    async def test_execute_backs_up_then_supersedes_loser_with_merge_reason(self, tmp_path):
        from scripts.core.memory_apply import run_merge_apply

        pool, conn = _pool()
        conn.fetch.side_effect = [self._merge_rows()]  # one pair -> one detail fetch
        order = []

        def _run(cmd, **kw):
            order.append("backup")
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        async def _exec(*a, **k):
            order.append("supersede")
            return "UPDATE 1"

        conn.execute = _exec
        backup_dir = tmp_path / "backups"

        result = await run_merge_apply(
            pool,
            "opc",
            [(self._A, self._B)],
            execute=True,
            backup_dir=backup_dir,
            timestamp="20260101-000000",
            run=_run,
        )
        assert result.backup_path is not None and result.backup_path.exists()
        assert order[0] == "backup"  # backup precedes any write
        assert "supersede" in order
        # row_b (recall 9) is keeper, row_a (recall 3) is loser
        assert result.applied == [self._A]

    async def test_supersede_called_with_keeper_loser_and_merge_reason(self, tmp_path):
        from scripts.core.memory_apply import run_merge_apply

        pool, conn = _pool()
        conn.fetch.side_effect = [self._merge_rows()]
        captured = {}

        async def _exec(sql, *params, **kw):
            captured["sql"] = sql
            captured["params"] = params
            return "UPDATE 1"

        conn.execute = _exec

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        await run_merge_apply(
            pool,
            "opc",
            [(self._A, self._B)],
            execute=True,
            backup_dir=tmp_path / "b",
            timestamp="t",
            run=_run,
        )
        # keeper=_B, loser=_A, reason="merge" all reach supersede_row's UPDATE
        assert self._A in captured["params"]  # loser
        assert self._B in captured["params"]  # keeper
        assert "merge" in captured["params"]
        assert "superseded_by IS NULL" in captured["sql"]

    async def test_idempotent_zero_row_update_skips_without_raising(self, tmp_path):
        from scripts.core.memory_apply import run_merge_apply

        pool, conn = _pool()
        conn.fetch.side_effect = [self._merge_rows()]
        conn.execute = AsyncMock(return_value="UPDATE 0")  # already superseded concurrently

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        result = await run_merge_apply(
            pool,
            "opc",
            [(self._A, self._B)],
            execute=True,
            backup_dir=tmp_path / "b",
            timestamp="t",
            run=_run,
        )
        assert result.applied == []  # nothing applied
        assert len(result.skipped) == 1  # reported as a skip, not a failure
        # no exception raised -> the assertion below proves we returned normally

    async def test_already_superseded_pair_skips_no_raise(self, tmp_path):
        from scripts.core.memory_apply import run_merge_apply

        pool, conn = _pool()
        # row_a already superseded -> select_merge_keeper refuses -> skip, no write
        conn.fetch.side_effect = [self._merge_rows(sup_a=self._B)]

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        result = await run_merge_apply(
            pool,
            "opc",
            [(self._A, self._B)],
            execute=True,
            backup_dir=tmp_path / "b",
            timestamp="t",
            run=_run,
        )
        assert result.applied == []
        assert len(result.skipped) == 1
        conn.execute.assert_not_called()  # no supersede attempted

    async def test_missing_side_skips(self, tmp_path):
        from scripts.core.memory_apply import run_merge_apply

        pool, conn = _pool()
        # only one id resolves -> cannot form a pair -> skip
        conn.fetch.side_effect = [[self._merge_rows()[0]]]

        result = await run_merge_apply(
            pool,
            "opc",
            [(self._A, self._B)],
            execute=False,
            backup_dir=tmp_path / "b",
            timestamp="t",
        )
        assert len(result.skipped) == 1


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
        rc = await ma.main(["other-project", "--ids", _UUID, "--execute"])
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
                _UUID,
                "--memory-dir",
                str(memory_dir),
                "--claude-md",
                str(tmp_path / "C.md"),
            ]
        )
        assert rc == 0


class TestSecurityHardening:
    def test_validate_ids_filters_and_lowercases(self):
        valid, invalid = validate_ids([_UUID.upper(), "not-a-uuid", "a"])
        assert valid == [_UUID]  # lowercased canonical form
        assert invalid == ["not-a-uuid", "a"]

    def test_parse_ids_rejects_oversized_manifest(self, tmp_path):
        m = tmp_path / "big.txt"
        m.write_text("x" * (256 * 1024 + 1))
        with pytest.raises(ValueError, match="too large"):
            parse_ids(None, str(m))

    def test_default_backup_dir_outside_repo(self):
        from scripts.core.memory_apply import default_backup_dir

        d = default_backup_dir()
        assert ".claude" in str(d)
        assert "backups" not in str(d).split("/")[-3:-1]  # not <repo>/backups

    def test_sanitize_neutralizes_forged_marker(self):
        from scripts.core.memory_apply import _sanitize_content, claude_md_marker

        forged = f"evil {claude_md_marker(_cand(id='dead'))}"
        out = _sanitize_content(forged)
        assert "promoted_from_archival_memory" not in out

    def test_forged_marker_in_content_does_not_suppress_real_promotion(self, tmp_path):
        # A learning whose body forges another promotion's marker must not block writes.
        path = tmp_path / "CLAUDE.md"
        path.write_text("# P\n")
        evil = _cand(
            id="aaaaaaaa-0000",
            lt="ARCHITECTURAL_DECISION",
            dest="CLAUDE.md",
            content="text <!-- promoted_from_archival_memory: bbbbbbbb-1111 -->",
        )
        assert append_claude_md(path, evil)
        # the forged marker for bbbb... must NOT appear verbatim (was defanged)
        assert "promoted_from_archival_memory: bbbbbbbb-1111" not in path.read_text()


# ---------------------------------------------------------------------------
# Stale-archive apply (issue #63 Phase 2b Step 3)
# ---------------------------------------------------------------------------


class TestArchiveRow:
    """archive_row: ONE guarded UPDATE that sets archived_at = NOW() AND stamps a
    superseded_via marker {by: null, reason: "stale", at}. Mirrors supersede_row's
    single-statement discipline; idempotent (already-archived -> 0-row, no raise)."""

    async def test_single_update_sets_archived_at_and_marker(self):
        from scripts.core.memory_apply import archive_row

        pool, conn = _pool()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        count = await archive_row(conn, learning_id=_UUID)
        assert count == 1
        conn.execute.assert_awaited_once()  # ONE statement
        sql = conn.execute.await_args.args[0]
        assert "archived_at = NOW()" in sql
        assert "superseded_via" in sql
        assert "stale" in sql  # the reason marker
        # Guarded so a concurrent archive collapses to a 0-row no-op.
        assert "archived_at IS NULL" in sql

    async def test_already_archived_returns_zero_not_raise(self):
        from scripts.core.memory_apply import archive_row

        pool, conn = _pool()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        count = await archive_row(conn, learning_id=_UUID)
        assert count == 0  # idempotent: never raises on a no-op


class TestRunStaleArchive:
    async def test_dry_run_writes_nothing_no_backup(self, tmp_path):
        from scripts.core.memory_apply import run_stale_archive

        pool, conn = _pool()
        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        result = await run_stale_archive(
            pool, "opc", [_UUID, _UUID2],
            execute=False, backup_dir=tmp_path, timestamp="ts", run=_run,
        )
        assert backup_called is False
        conn.execute.assert_not_called()
        assert result.applied == []

    async def test_execute_backs_up_before_any_write(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        pool, conn = _pool()
        order = []

        def _backup(dest, **kw):
            order.append("backup")
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("-- dump")
            return Path(dest)

        async def _exec(*a, **k):
            order.append("archive")
            return "UPDATE 1"

        conn.execute = _exec
        monkeypatch.setattr(ma, "backup_database", _backup)
        result = await ma.run_stale_archive(
            pool, "opc", [_UUID, _UUID2],
            execute=True, backup_dir=tmp_path, timestamp="ts", lock_dir=tmp_path,
        )
        # Backup happens FIRST, before any archive write.
        assert order and order[0] == "backup"
        assert "archive" in order
        assert result.applied == [_UUID, _UUID2]
        assert result.backup_path is not None

    async def test_idempotent_already_archived_skipped(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        pool, conn = _pool()

        def _backup(dest, **kw):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("-- dump")
            return Path(dest)

        async def _exec(*a, **k):
            return "UPDATE 0"  # every row already archived

        conn.execute = _exec
        monkeypatch.setattr(ma, "backup_database", _backup)
        result = await ma.run_stale_archive(
            pool, "opc", [_UUID],
            execute=True, backup_dir=tmp_path, timestamp="ts", lock_dir=tmp_path,
        )
        assert result.applied == []  # nothing newly archived
        assert _UUID in [s[0] for s in result.skipped]  # reported as skipped, not raised


class TestArchiveCliMain:
    async def test_archive_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        pool, conn = _pool()
        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        async def _get_pool():
            return pool

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        rc = await ma.main(
            ["opc", "--archive", "--ids", _UUID, "--backup-dir", str(tmp_path)]
        )
        assert rc == 0
        assert backup_called is False
        conn.execute.assert_not_called()

    async def test_archive_execute_backs_up_first(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        pool, conn = _pool()
        order = []

        def _backup(dest, **kw):
            order.append("backup")
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("-- dump")
            return Path(dest)

        async def _exec(*a, **k):
            order.append("archive")
            return "UPDATE 1"

        conn.execute = _exec

        async def _get_pool():
            return pool

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        monkeypatch.setattr(ma, "backup_database", _backup)
        rc = await ma.main(
            ["opc", "--archive", "--ids", _UUID, "--execute", "--backup-dir", str(tmp_path)]
        )
        assert rc == 0
        assert order and order[0] == "backup"
        assert "archive" in order

    def test_archive_flag_parses(self):
        args = parse_apply_args(["opc", "--archive", "--ids", _UUID])
        assert args.archive is True


# --- Unpromote / repair (issue #63 Phase 2b Step 4) ---


def _prow(id="a", *, tier="MEMORY.md", target="/m/promoted-x.md", lt="CODEBASE_PATTERN",
          content="A useful pattern", recall=12):
    from scripts.core.memory_review import PromotedRow

    return PromotedRow(
        id=id, content=content, recall_count=recall, learning_type=lt, tier=tier, target=target
    )


class TestBuildUnpromotePlan:
    """build_unpromote_plan (pure): map promoted rows -> unpromote actions, distinguishing the
    single-artifact CLAUDE.md block from the two-artifact MEMORY.md (file + index line), and
    skipping a row whose promoted_to is already absent (tier/target None)."""

    def test_memory_tier_is_two_artifact(self):
        from scripts.core.memory_apply import build_unpromote_plan

        plan = build_unpromote_plan([_prow("a", tier="MEMORY.md")], dry_run=True)
        assert len(plan.actions) == 1
        act = plan.actions[0]
        assert act.skipped is False
        assert act.two_artifact is True
        assert act.tier == "MEMORY.md"

    def test_claude_tier_is_single_artifact(self):
        from scripts.core.memory_apply import build_unpromote_plan

        plan = build_unpromote_plan(
            [_prow("a", tier="CLAUDE.md", target="/c/CLAUDE.md", lt="ARCHITECTURAL_DECISION")],
            dry_run=True,
        )
        act = plan.actions[0]
        assert act.skipped is False
        assert act.two_artifact is False
        assert act.tier == "CLAUDE.md"

    def test_skips_row_with_no_promoted_marker(self):
        from scripts.core.memory_apply import build_unpromote_plan

        plan = build_unpromote_plan([_prow("a", tier=None, target=None)], dry_run=True)
        act = plan.actions[0]
        assert act.skipped is True
        assert "already" in (act.skip_reason or "").lower()

    def test_unknown_tier_skipped(self):
        from scripts.core.memory_apply import build_unpromote_plan

        plan = build_unpromote_plan([_prow("a", tier="rules/", target="/x")], dry_run=True)
        assert plan.actions[0].skipped is True

    def test_dry_run_flag_carried(self):
        from scripts.core.memory_apply import build_unpromote_plan

        assert build_unpromote_plan([_prow()], dry_run=True).dry_run is True
        assert build_unpromote_plan([_prow()], dry_run=False).dry_run is False

    def test_pure_no_io(self):
        from scripts.core.memory_apply import build_unpromote_plan

        rows = [_prow("a"), _prow("b", tier="CLAUDE.md", target="/c/CLAUDE.md")]
        plan = build_unpromote_plan(rows, dry_run=True)
        assert len(plan.applicable) == 2  # both routable


class TestClearPromotedMarker:
    """clear_promoted_marker: ONE guarded UPDATE that removes ONLY the promoted_to key
    (metadata - 'promoted_to'), guarded so it matches only a row that still has it.
    Idempotent: a 0-row clear is a no-op, never a raise. Never touches superseded_via."""

    async def test_single_update_removes_only_promoted_to(self):
        from scripts.core.memory_apply import clear_promoted_marker

        pool, conn = _pool()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        count = await clear_promoted_marker(conn, learning_id=_UUID, project="opc")
        assert count == 1
        conn.execute.assert_awaited_once()
        sql = conn.execute.await_args.args[0]
        assert "promoted_to" in sql
        assert "- 'promoted_to'" in sql or "- $" in sql  # subtracts the key
        assert "superseded_via" not in sql  # must not touch other keys
        assert "metadata ? 'promoted_to'" in sql  # guarded to rows still carrying it

    async def test_already_cleared_returns_zero_not_raise(self):
        from scripts.core.memory_apply import clear_promoted_marker

        pool, conn = _pool()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        count = await clear_promoted_marker(conn, learning_id=_UUID, project="opc")
        assert count == 0  # idempotent: no raise


class TestRemoveLineByMarker:
    """remove_line_by_marker: splice a single line out of a file by a substring marker,
    atomically. Absent marker -> no-op (already removed); returns whether a line was removed."""

    def test_removes_matching_line(self, tmp_path):
        from scripts.core.memory_apply import remove_line_by_marker

        f = tmp_path / "MEMORY.md"
        f.write_text("# Index\n- [x](promoted-x.md) — promoted\n- [y](other.md)\n")
        removed = remove_line_by_marker(f, "promoted-x.md")
        assert removed is True
        text = f.read_text()
        assert "promoted-x.md" not in text
        assert "other.md" in text  # other lines untouched

    def test_absent_marker_is_noop(self, tmp_path):
        from scripts.core.memory_apply import remove_line_by_marker

        f = tmp_path / "MEMORY.md"
        f.write_text("# Index\n- [y](other.md)\n")
        removed = remove_line_by_marker(f, "promoted-x.md")
        assert removed is False
        assert f.read_text() == "# Index\n- [y](other.md)\n"

    def test_missing_file_is_noop(self, tmp_path):
        from scripts.core.memory_apply import remove_line_by_marker

        removed = remove_line_by_marker(tmp_path / "nope.md", "marker")
        assert removed is False


class TestRunUnpromote:
    """The W-2 ordering invariant. Two-artifact: delete promoted-<slug>.md FIRST, THEN splice
    the MEMORY.md index line, THEN clear promoted_to. Single-artifact: remove the CLAUDE.md
    block FIRST, then clear. A clear is a guarded UPDATE; 0-row -> skip, never raise."""

    def _setup(self, tmp_path):
        memory_dir = tmp_path / "mem"
        memory_dir.mkdir()
        claude_md = tmp_path / "CLAUDE.md"
        backup_dir = tmp_path / "backups"
        return memory_dir, claude_md, backup_dir

    async def test_dry_run_writes_nothing_no_backup(self, tmp_path):
        from scripts.core.memory_apply import run_unpromote

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        slug_file = memory_dir / "promoted-x.md"
        slug_file.write_text("body")
        (memory_dir / "MEMORY.md").write_text("- [x](promoted-x.md) — promoted\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [
            [{"id": "a", "content": "c", "recall_count": 12,
              "learning_type": "CODEBASE_PATTERN",
              "promoted_tier": "MEMORY.md", "promoted_target": str(slug_file)}]
        ]
        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        result = await run_unpromote(
            pool, "opc", ["a"], execute=False, memory_dir=memory_dir,
            claude_md_path=claude_md, backup_dir=backup_dir, timestamp="ts", run=_run,
        )
        assert result.plan.dry_run is True
        assert backup_called is False
        assert slug_file.exists()  # nothing deleted in dry-run
        conn.execute.assert_not_called()

    async def test_two_artifact_order_file_then_index_then_clear(self, tmp_path):
        from scripts.core.memory_apply import run_unpromote

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        slug_file = memory_dir / "promoted-x.md"
        slug_file.write_text("body")
        memory_md = memory_dir / "MEMORY.md"
        memory_md.write_text("# Index\n- [x](promoted-x.md) — promoted\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [
            [{"id": "a", "content": "c", "recall_count": 12,
              "learning_type": "CODEBASE_PATTERN",
              "promoted_tier": "MEMORY.md", "promoted_target": str(slug_file)}]
        ]
        order = []

        def _run(cmd, **kw):
            order.append("backup")
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        async def _exec(*a, **k):
            # the DB clear must come AFTER the file is gone and the index line spliced
            order.append("clear")
            assert not slug_file.exists(), "file must be deleted before DB clear"
            assert "promoted-x.md" not in memory_md.read_text(), "index spliced before clear"
            return "UPDATE 1"

        conn.execute = _exec
        result = await run_unpromote(
            pool, "opc", ["a"], execute=True, memory_dir=memory_dir,
            claude_md_path=claude_md, backup_dir=backup_dir, timestamp="ts", run=_run,
        )
        assert order[0] == "backup"
        assert "clear" in order
        assert not slug_file.exists()
        assert "promoted-x.md" not in memory_md.read_text()
        assert result.applied == ["a"]

    async def test_two_artifact_partial_failure_does_not_clear(self, tmp_path, monkeypatch):
        # W-2: index-line removal fails AFTER the file delete -> promoted_to is NOT cleared,
        # so the action is re-runnable (the file is gone but the DB still points at it).
        import scripts.core.memory_apply as ma
        from scripts.core.memory_apply import run_unpromote

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        slug_file = memory_dir / "promoted-x.md"
        slug_file.write_text("body")
        memory_md = memory_dir / "MEMORY.md"
        memory_md.write_text("# Index\n- [x](promoted-x.md) — promoted\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [
            [{"id": "a", "content": "c", "recall_count": 12,
              "learning_type": "CODEBASE_PATTERN",
              "promoted_tier": "MEMORY.md", "promoted_target": str(slug_file)}]
        ]
        cleared = False

        async def _exec(*a, **k):
            nonlocal cleared
            cleared = True
            return "UPDATE 1"

        conn.execute = _exec

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        def _boom(*a, **k):
            raise OSError("disk full splicing index")

        monkeypatch.setattr(ma, "remove_line_by_marker", _boom)
        result = await run_unpromote(
            pool, "opc", ["a"], execute=True, memory_dir=memory_dir,
            claude_md_path=claude_md, backup_dir=backup_dir, timestamp="ts", run=_run,
        )
        # The file delete happened (stage 1), but the index splice failed (stage 2),
        # so the DB clear (stage 3) must NOT have run.
        assert cleared is False, "promoted_to must not be cleared when index splice fails"
        assert "a" not in result.applied
        assert result.failed and result.failed[0][0] == "a"

    async def test_single_artifact_removes_block_then_clears(self, tmp_path):
        from scripts.core.memory_apply import claude_md_block, claude_md_marker, run_unpromote
        from scripts.core.memory_review import PromotionCandidate

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        cand = PromotionCandidate(
            id="a", content="Decide X", recall_count=12,
            learning_type="ARCHITECTURAL_DECISION", destination="CLAUDE.md",
        )
        marker = claude_md_marker(cand)
        claude_md.write_text("# Project\n## Promoted Decisions\n" + claude_md_block(cand) + "\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [
            [{"id": "a", "content": "Decide X", "recall_count": 12,
              "learning_type": "ARCHITECTURAL_DECISION",
              "promoted_tier": "CLAUDE.md", "promoted_target": str(claude_md)}]
        ]
        order = []

        def _run(cmd, **kw):
            order.append("backup")
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        async def _exec(*a, **k):
            order.append("clear")
            assert marker not in claude_md.read_text(), "block spliced before DB clear"
            return "UPDATE 1"

        conn.execute = _exec
        result = await run_unpromote(
            pool, "opc", ["a"], execute=True, memory_dir=memory_dir,
            claude_md_path=claude_md, backup_dir=backup_dir, timestamp="ts", run=_run,
        )
        assert order[0] == "backup"
        assert marker not in claude_md.read_text()
        assert result.applied == ["a"]

    async def test_idempotent_already_unpromoted_no_raise_no_write(self, tmp_path):
        # No promoted_to row resolves -> plan has nothing applicable; files already absent.
        from scripts.core.memory_apply import run_unpromote

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        (memory_dir / "MEMORY.md").write_text("# Index\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [[]]  # fetch_promoted_rows returns nothing
        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        result = await run_unpromote(
            pool, "opc", ["a"], execute=True, memory_dir=memory_dir,
            claude_md_path=claude_md, backup_dir=backup_dir, timestamp="ts", run=_run,
        )
        assert result.applied == []
        assert backup_called is False  # nothing applicable -> no backup, no write
        conn.execute.assert_not_called()

    async def test_clear_zero_row_reports_skip_not_raise(self, tmp_path):
        # Stages 1-2 succeed but the guarded clear matches 0 rows (concurrent clear) ->
        # reported as skipped, never raised.
        from scripts.core.memory_apply import run_unpromote

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        slug_file = memory_dir / "promoted-x.md"
        slug_file.write_text("body")
        (memory_dir / "MEMORY.md").write_text("- [x](promoted-x.md) — promoted\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [
            [{"id": "a", "content": "c", "recall_count": 12,
              "learning_type": "CODEBASE_PATTERN",
              "promoted_tier": "MEMORY.md", "promoted_target": str(slug_file)}]
        ]
        conn.execute = AsyncMock(return_value="UPDATE 0")

        def _run(cmd, **kw):
            Path(kw["stdout"].name).write_text("-- dump")
            return MagicMock(returncode=0)

        result = await run_unpromote(
            pool, "opc", ["a"], execute=True, memory_dir=memory_dir,
            claude_md_path=claude_md, backup_dir=backup_dir, timestamp="ts", run=_run,
        )
        assert result.applied == []
        assert "a" in [s[0] for s in result.skipped]
        assert not slug_file.exists()  # stages 1-2 still ran

    async def test_backup_taken_before_any_removal(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma
        from scripts.core.memory_apply import run_unpromote

        memory_dir, claude_md, backup_dir = self._setup(tmp_path)
        slug_file = memory_dir / "promoted-x.md"
        slug_file.write_text("body")
        (memory_dir / "MEMORY.md").write_text("- [x](promoted-x.md) — promoted\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [
            [{"id": "a", "content": "c", "recall_count": 12,
              "learning_type": "CODEBASE_PATTERN",
              "promoted_tier": "MEMORY.md", "promoted_target": str(slug_file)}]
        ]
        order = []

        def _backup(dest, **kw):
            order.append("backup")
            assert slug_file.exists(), "backup must precede any file removal"
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("-- dump")
            return Path(dest)

        async def _exec(*a, **k):
            return "UPDATE 1"

        conn.execute = _exec
        monkeypatch.setattr(ma, "backup_database", _backup)
        await run_unpromote(
            pool, "opc", ["a"], execute=True, memory_dir=memory_dir,
            claude_md_path=claude_md, backup_dir=backup_dir, timestamp="ts", lock_dir=tmp_path,
        )
        assert order and order[0] == "backup"


class TestUnpromoteCli:
    def test_unpromote_flag_parses(self):
        args = parse_apply_args(["opc", "--unpromote", "--ids", _UUID])
        assert args.unpromote is True

    async def test_mutually_exclusive_with_merge(self, monkeypatch):
        import scripts.core.memory_apply as ma

        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        rc = await ma.main(["opc", "--unpromote", "--merge", "--ids", _UUID])
        assert rc == 2

    async def test_mutually_exclusive_with_archive(self, monkeypatch):
        import scripts.core.memory_apply as ma

        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        rc = await ma.main(["opc", "--unpromote", "--archive", "--ids", _UUID])
        assert rc == 2

    async def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        pool, conn = _pool()
        conn.fetch.side_effect = [[]]  # no promoted rows
        backup_called = False

        def _run(*a, **k):
            nonlocal backup_called
            backup_called = True
            return MagicMock(returncode=0)

        async def _get_pool():
            return pool

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        rc = await ma.main([
            "opc", "--unpromote", "--ids", _UUID,
            "--memory-dir", str(tmp_path / "m"), "--claude-md", str(tmp_path / "CLAUDE.md"),
            "--backup-dir", str(tmp_path / "b"),
        ])
        assert rc == 0
        assert backup_called is False
        conn.execute.assert_not_called()

    async def test_execute_backs_up_first(self, tmp_path, monkeypatch):
        import scripts.core.memory_apply as ma

        memory_dir = tmp_path / "m"
        memory_dir.mkdir()
        slug_file = memory_dir / "promoted-x.md"
        slug_file.write_text("body")
        (memory_dir / "MEMORY.md").write_text("- [x](promoted-x.md) — promoted\n")
        pool, conn = _pool()
        conn.fetch.side_effect = [
            [{"id": _UUID, "content": "c", "recall_count": 12,
              "learning_type": "CODEBASE_PATTERN",
              "promoted_tier": "MEMORY.md", "promoted_target": str(slug_file)}]
        ]
        order = []

        def _backup(dest, **kw):
            order.append("backup")
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("-- dump")
            return Path(dest)

        async def _exec(*a, **k):
            order.append("clear")
            return "UPDATE 1"

        conn.execute = _exec

        async def _get_pool():
            return pool

        monkeypatch.setattr(ma, "get_pool", _get_pool)
        monkeypatch.setattr(ma, "project_from_path", lambda _p: "opc")
        monkeypatch.setattr(ma, "backup_database", _backup)
        rc = await ma.main([
            "opc", "--unpromote", "--ids", _UUID, "--execute",
            "--memory-dir", str(memory_dir), "--claude-md", str(tmp_path / "CLAUDE.md"),
            "--backup-dir", str(tmp_path / "b"),
        ])
        assert rc == 0
        assert order and order[0] == "backup"
        assert "clear" in order
