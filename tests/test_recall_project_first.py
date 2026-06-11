"""Tests for issue #139 — fetch-time project scoping (--project-first).

Opt-in two-pass fetch: pass 1 scoped via SQL `AND project = $N`, pass 2 global,
merged own-first with id-dedupe and fetch_k truncation. Default path (no flag)
must stay byte-identical to today.
"""

from __future__ import annotations

from typing import Any

import pytest

# ==================== Unit 1: render_recall_sql project filter ====================


class TestRecallSqlProjectFilter:
    """render_recall_sql renders an optional `AND project = $N` clause.

    None (default) => SQL byte-identical to today; a clause string =>
    the clause is injected into the WHERE block (issue #139).
    """

    def _simple_templates(self):
        from scripts.core import recall_backends as rb

        return [
            rb._TEXT_ONLY_FTS_SQL,
            rb._TEXT_ONLY_FTS_NO_CHAIN_SQL,
            rb._TEXT_ONLY_ILIKE_SQL,
            rb._TEXT_ONLY_ILIKE_NO_CHAIN_SQL,
        ]

    def _chain_templates(self):
        from scripts.core import recall_backends as rb

        return [rb._PG_RECENCY_SQL, rb._PG_VECTOR_SQL, rb._PG_TEXT_FALLBACK_SQL]

    def test_default_render_unchanged(self):
        """No project_filter => identical to a render without the kwarg."""
        from scripts.core.recall_backends import render_recall_sql

        for tmpl in self._simple_templates():
            for include in (True, False):
                baseline = render_recall_sql(tmpl, include_project=include)
                with_kwarg = render_recall_sql(
                    tmpl, include_project=include, project_filter=None,
                )
                assert baseline == with_kwarg

        for tmpl in self._chain_templates():
            baseline = render_recall_sql(
                tmpl, include_project=True, chain_filter="",
            )
            with_kwarg = render_recall_sql(
                tmpl, include_project=True, chain_filter="", project_filter=None,
            )
            assert baseline == with_kwarg

    def test_simple_template_injects_clause(self):
        from scripts.core.recall_backends import render_recall_sql

        for tmpl in self._simple_templates():
            sql = render_recall_sql(
                tmpl, include_project=True, project_filter="AND project = $3",
            )
            assert "AND project = $3" in sql
            # Clause belongs in the WHERE block, before ORDER BY / LIMIT.
            assert sql.index("AND project = $3") < sql.index("LIMIT")

    def test_chain_template_injects_clause(self):
        from scripts.core.recall_backends import render_recall_sql

        for tmpl in self._chain_templates():
            sql = render_recall_sql(
                tmpl,
                include_project=True,
                chain_filter="AND superseded_by IS NULL",
                project_filter="AND project = $4",
            )
            assert "AND project = $4" in sql

    def test_no_unfilled_placeholders_with_filter(self):
        from scripts.core.recall_backends import render_recall_sql

        for tmpl in self._simple_templates():
            sql = render_recall_sql(
                tmpl, include_project=True, project_filter="AND project = $3",
            )
            assert "{" not in sql and "}" not in sql
        for tmpl in self._chain_templates():
            sql = render_recall_sql(
                tmpl,
                include_project=True,
                chain_filter="AND superseded_by IS NULL",
                project_filter="AND project = $4",
            )
            assert "{" not in sql and "}" not in sql


class TestRrfCteProjectFilter:
    """The RRF filter lives in the CTE subqueries (before ranking), not the
    tail — a project predicate must shrink the ranked pool (issue #139)."""

    def test_default_cte_unchanged(self):
        from scripts.core.recall_backends import build_rrf_cte

        for chain in (True, False):
            for use_ts in (True, False):
                baseline = build_rrf_cte(chain_filter=chain, use_tsquery=use_ts)
                with_kwarg = build_rrf_cte(
                    chain_filter=chain, use_tsquery=use_ts, project_filter=None,
                )
                assert baseline == with_kwarg

    def test_cte_injects_clause_into_both_subqueries(self):
        from scripts.core.recall_backends import build_rrf_cte

        cte = build_rrf_cte(
            chain_filter=True, use_tsquery=False, project_filter="AND project = $4",
        )
        # Both fts_ranked and vector_ranked subqueries must carry the filter.
        assert cte.count("AND project = $4") == 2


# ==================== Unit 2: merge_project_first ====================


def _row(rid: str, project: str | None = None) -> dict[str, Any]:
    return {"id": rid, "content": f"c-{rid}", "metadata": {"project": project}}


class TestMergeProjectFirst:
    """own-project rows first, then global rows deduped by id, truncated to
    fetch_k. Pure function, no I/O (issue #139)."""

    def test_own_rows_come_first(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [_row("a"), _row("b")]
        global_ = [_row("c"), _row("d")]
        merged = merge_project_first(own, global_, fetch_k=10)
        assert [r["id"] for r in merged] == ["a", "b", "c", "d"]

    def test_global_rows_deduped_by_id(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [_row("a"), _row("b")]
        # "a" and "b" reappear in the global fill — must not duplicate.
        global_ = [_row("a"), _row("c"), _row("b"), _row("d")]
        merged = merge_project_first(own, global_, fetch_k=10)
        assert [r["id"] for r in merged] == ["a", "b", "c", "d"]

    def test_truncated_to_fetch_k(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [_row("a"), _row("b")]
        global_ = [_row("c"), _row("d"), _row("e")]
        merged = merge_project_first(own, global_, fetch_k=3)
        assert [r["id"] for r in merged] == ["a", "b", "c"]

    def test_own_overflow_respects_global_quota(self):
        """The pool is capped at fetch_k AND a global slot is reserved
        (Finding 2): with fetch_k=2 the own quota is ceil(2/2)=1, so own
        contributes 'a' and global keeps its slot with 'd'."""
        from scripts.core.recall_learnings import merge_project_first

        own = [_row("a"), _row("b"), _row("c")]
        global_ = [_row("d")]
        merged = merge_project_first(own, global_, fetch_k=2)
        assert [r["id"] for r in merged] == ["a", "d"]

    def test_empty_own(self):
        from scripts.core.recall_learnings import merge_project_first

        merged = merge_project_first([], [_row("c"), _row("d")], fetch_k=10)
        assert [r["id"] for r in merged] == ["c", "d"]

    def test_empty_global(self):
        from scripts.core.recall_learnings import merge_project_first

        merged = merge_project_first([_row("a")], [], fetch_k=10)
        assert [r["id"] for r in merged] == ["a"]

    def test_both_empty(self):
        from scripts.core.recall_learnings import merge_project_first

        assert merge_project_first([], [], fetch_k=10) == []

    def test_does_not_mutate_inputs(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [_row("a")]
        global_ = [_row("a"), _row("b")]
        merge_project_first(own, global_, fetch_k=10)
        assert [r["id"] for r in own] == ["a"]
        assert [r["id"] for r in global_] == ["a", "b"]

    def test_first_seen_own_row_wins_on_id_collision(self):
        """When an id is in both lists, the own-project copy is kept."""
        from scripts.core.recall_learnings import merge_project_first

        own = [{"id": "a", "content": "own-copy"}]
        global_ = [{"id": "a", "content": "global-copy"}]
        merged = merge_project_first(own, global_, fetch_k=10)
        assert len(merged) == 1
        assert merged[0]["content"] == "own-copy"


# ==================== Unit 3: resolve_search_params project_scope ===========


def _base_params(**overrides: Any) -> dict[str, Any]:
    """Default kwargs for resolve_search_params (postgres hybrid_rrf)."""
    params: dict[str, Any] = dict(
        backend="postgres",
        text_only=False,
        vector_only=False,
        query="test",
        fetch_k=50,
        provider="local",
        threshold=0.2,
        recency=0.1,
        no_rerank=False,
        no_expand=False,
        expand_terms=5,
        rebuild_idf=False,
    )
    params.update(overrides)
    return params


class TestResolveSearchParamsProjectScope:
    """resolve_search_params carries an optional project_scope through to the
    backend dispatcher (issue #139). Default None => no behavior change."""

    def test_default_has_no_project_scope(self):
        from scripts.core.recall_learnings import resolve_search_params

        params = resolve_search_params(**_base_params())
        assert params.get("project_scope") is None

    def test_project_scope_carried_for_hybrid(self):
        from scripts.core.recall_learnings import resolve_search_params

        params = resolve_search_params(**_base_params(), project_scope="opc")
        assert params["mode"] == "hybrid_rrf"
        assert params["project_scope"] == "opc"

    def test_project_scope_carried_for_text_only(self):
        from scripts.core.recall_learnings import resolve_search_params

        params = resolve_search_params(
            **_base_params(text_only=True), project_scope="opc",
        )
        assert params["mode"] == "text_only"
        assert params["project_scope"] == "opc"

    def test_project_scope_carried_for_vector(self):
        from scripts.core.recall_learnings import resolve_search_params

        params = resolve_search_params(
            **_base_params(vector_only=True), project_scope="opc",
        )
        assert params["mode"] == "vector"
        assert params["project_scope"] == "opc"

    def test_project_scope_carried_for_sqlite(self):
        from scripts.core.recall_learnings import resolve_search_params

        params = resolve_search_params(
            **_base_params(backend="sqlite"), project_scope="opc",
        )
        assert params["mode"] == "sqlite"
        assert params["project_scope"] == "opc"


# ==================== Unit 4: CLI --project-first parse + resolve ===========


class TestProjectFirstCliFlag:
    """--project-first parses and defaults off (issue #139)."""

    def test_flag_defaults_false(self):
        from scripts.core.recall_learnings import _build_arg_parser

        args = _build_arg_parser().parse_args(["--query", "x"])
        assert args.project_first is False

    def test_flag_sets_true(self):
        from scripts.core.recall_learnings import _build_arg_parser

        args = _build_arg_parser().parse_args(["--query", "x", "--project-first"])
        assert args.project_first is True


class TestResolveProjectScope:
    """resolve_project_scope: explicit --project wins, else worktree-aware
    auto-detect; canonicalized; None when nothing resolves (issue #139)."""

    def test_off_returns_none(self):
        from scripts.core.recall_learnings import resolve_project_scope

        assert (
            resolve_project_scope(
                project_first=False, project="opc", project_dir="/x/opc",
            )
            is None
        )

    def test_explicit_project_wins_and_canonicalizes(self):
        from scripts.core.recall_learnings import resolve_project_scope

        # Upper-cased explicit value must come back canonicalized (lowercased).
        assert (
            resolve_project_scope(
                project_first=True, project="OPC", project_dir="/other/path",
            )
            == "opc"
        )

    def test_auto_detect_from_project_dir(self):
        from scripts.core.recall_learnings import resolve_project_scope

        assert (
            resolve_project_scope(
                project_first=True, project=None,
                project_dir="/Users/x/opc",
            )
            == "opc"
        )

    def test_worktree_path_resolves_to_repo(self):
        from scripts.core.recall_learnings import resolve_project_scope

        assert (
            resolve_project_scope(
                project_first=True, project=None,
                project_dir="/Users/x/opc/.worktrees/branch",
            )
            == "opc"
        )

    def test_unresolvable_returns_none(self):
        from scripts.core.recall_learnings import resolve_project_scope

        assert (
            resolve_project_scope(
                project_first=True, project=None, project_dir=None,
            )
            is None
        )


# ==================== Unit 5: two-pass backend + dispatch integration =======


class _CapturingConn:
    """Records (sql, args) for every fetch and returns canned rows.

    ``row_provider(sql, args)`` decides what rows to return so a test can
    distinguish the scoped pass (SQL contains 'project =') from the global
    pass. ``missing_columns`` simulates a pre-migration DB by raising
    UndefinedColumnError when the column name appears in the SQL.
    """

    def __init__(self, row_provider, missing_columns: set[str] | None = None):
        self.calls: list[tuple[str, tuple]] = []
        self._row_provider = row_provider
        self._missing = missing_columns or set()

    async def fetch(self, sql: str, *args):
        from asyncpg.exceptions import UndefinedColumnError

        for col in self._missing:
            # The capability probe selects bare 'project'; treat any project
            # reference as missing on a pre-migration DB.
            if col in sql:
                raise UndefinedColumnError(f'column "{col}" does not exist')
        self.calls.append((sql, args))
        return self._row_provider(sql, args)


class _CapturingPool:
    def __init__(self, conn: _CapturingConn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Acquire:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


def _patch_capturing_pool(monkeypatch, conn: _CapturingConn) -> None:
    async def fake_get_pool():
        return _CapturingPool(conn)

    import scripts.core.db.postgres_pool as pool_mod

    monkeypatch.setattr(pool_mod, "get_pool", fake_get_pool)


def _text_row(rid: str, project: str | None) -> dict[str, Any]:
    import uuid
    from datetime import datetime as _dt

    return {
        "id": uuid.uuid4(),
        "session_id": f"s-{rid}",
        "content": rid,
        "metadata": {"type": "session_learning"},
        "created_at": _dt(2026, 1, 1),
        "project": project,
        "similarity": 0.5,
    }


class TestTextOnlyProjectScopedFetch:
    """search_learnings_text_only_postgres(project=...) issues a project-bound
    scoped pass; project=None stays byte-identical to today (issue #139)."""

    async def test_scoped_pass_binds_project_clause(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()

        def provider(sql, args):
            return [_text_row("own", "opc")]

        conn = _CapturingConn(provider)
        _patch_capturing_pool(monkeypatch, conn)

        results = await rb.search_learnings_text_only_postgres(
            "query terms", k=3, project="opc",
        )
        assert results, "scoped pass should return rows"
        # At least one executed query must carry the case-tolerant project
        # predicate, and the canonical project value must be bound as a
        # positional arg (Finding 1: LOWER(project) = $N).
        scoped = [c for c in conn.calls if "LOWER(project) =" in c[0]]
        assert scoped, "expected a project-bound scoped query"
        sql, args = scoped[0]
        assert "opc" in args

    async def test_no_project_is_byte_identical(self, monkeypatch):
        """project=None must not add any project predicate (default path)."""
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn(lambda sql, args: [])
        _patch_capturing_pool(monkeypatch, conn)

        await rb.search_learnings_text_only_postgres("query terms", k=3)
        assert conn.calls, "expected SQL to run"
        for sql, _args in conn.calls:
            assert "LOWER(project)" not in sql


class TestProjectScopedDegradesOnOldDb:
    """A pre-migration DB (no project column) must skip the scoped pass and
    fall back to a single global fetch with no scoped retry loop (issue #139)."""

    async def test_old_db_skips_scoped_clause(self, monkeypatch):
        from scripts.core import recall_backends as rb

        rb.reset_project_column_cache()
        conn = _CapturingConn(lambda sql, args: [], missing_columns={"project"})
        _patch_capturing_pool(monkeypatch, conn)

        # Even with a project requested, the missing column must degrade to
        # project-free SQL — no UndefinedColumnError bubbling out.
        results = await rb.search_learnings_text_only_postgres(
            "query terms", k=3, project="opc",
        )
        assert results == []
        assert conn.calls, "expected fallback SQL to run"
        for sql, _args in conn.calls:
            assert "LOWER(project)" not in sql, sql[:120]


class TestDispatchProjectFirst:
    """_dispatch_search_project_first runs scoped + global passes and merges
    own-first (issue #139)."""

    async def test_merges_scoped_then_global(self, monkeypatch):
        import scripts.core.recall_learnings as rl

        calls: list[str | None] = []

        async def fake_text(query, k, *, project=None):
            calls.append(project)
            if project is not None:
                return [{"id": "own1"}, {"id": "own2"}]
            return [{"id": "own1"}, {"id": "glob1"}]

        monkeypatch.setattr(rl, "search_learnings_text_only_postgres", fake_text)

        params = {
            "mode": "text_only",
            "query": "q",
            "k": 10,
            "project_scope": "opc",
        }
        merged = await rl._dispatch_search_project_first(params)
        # Scoped pass ran with project, global pass ran with None.
        assert "opc" in calls and None in calls
        # own-first, deduped: own1, own2, then glob1.
        assert [r["id"] for r in merged] == ["own1", "own2", "glob1"]


class TestMainDegradesWithWarning:
    """main(): --project-first with no resolvable project warns to stderr and
    runs the global single-pass dispatch, not the two-pass path (issue #139)."""

    async def test_no_project_warns_and_uses_global_dispatch(
        self, monkeypatch, capsys,
    ):
        import scripts.core.recall_learnings as rl

        # No project anywhere: explicit None + no CLAUDE_PROJECT_DIR.
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(
            rl.sys, "argv",
            ["recall_learnings.py", "--query", "x", "--project-first",
             "--text-only", "--json", "--no-rerank"],
            raising=False,
        )

        dispatched: dict[str, bool] = {"global": False, "first": False}

        async def fake_global(params, *, project=None):
            dispatched["global"] = True
            return []

        async def fake_first(params):
            dispatched["first"] = True
            return []

        monkeypatch.setattr(rl, "_dispatch_search", fake_global)
        monkeypatch.setattr(rl, "_dispatch_search_project_first", fake_first)
        monkeypatch.setattr(rl, "get_backend", lambda: "postgres")

        async def noop_record(_ids):
            return None

        monkeypatch.setattr(rl, "record_recall", noop_record)

        rc = await rl.main()
        assert rc == 0
        assert dispatched["global"] is True
        assert dispatched["first"] is False
        err = capsys.readouterr().err
        assert "--project-first" in err and "global recall" in err



# ==================== Round-2 Finding 1: case-tolerant scoped predicate ======


class TestProjectFilterCaseTolerant:
    """The scoped predicate must be case-insensitive (LOWER(project) = $N) so
    un-migrated DBs holding 'OPC'/'Opc' still match the canonical 'opc' bind,
    matching the reranker's case tolerance (issue #139 review round 2)."""

    def test_clause_uses_lower(self):
        from scripts.core.recall_backends import project_filter_clause

        clause = project_filter_clause("opc", has_project=True, param_index=3)
        assert clause == "AND LOWER(project) = $3"

    def test_no_clause_when_unscoped(self):
        from scripts.core.recall_backends import project_filter_clause

        assert project_filter_clause(None, has_project=True, param_index=3) == ""
        assert project_filter_clause("opc", has_project=False, param_index=3) == ""

    def test_canonicalize_project_yields_lowercase_bind(self):
        """The bind value is lowercase by construction, so LOWER(project)=$N
        compares lower-to-lower (review round 2 assertion)."""
        from scripts.core.project_naming import canonicalize_project

        for raw in ("OPC", "Opc", "  oPc  ", "opc"):
            assert canonicalize_project(raw) == "opc"


# ==================== Round-2 Finding 2: global quota in merge ===============


class TestMergeProjectFirstGlobalQuota:
    """merge_project_first must reserve a global quota so an own-project pass
    that returns fetch_k rows cannot starve global candidates entirely
    (issue #139 review round 2)."""

    def test_global_survives_when_own_overflows(self):
        from scripts.core.recall_learnings import merge_project_first

        # 6 own rows, fetch_k=6 — without a quota all 6 slots are own.
        own = [_row(f"o{i}") for i in range(6)]
        global_ = [_row(f"g{i}") for i in range(6)]
        merged = merge_project_first(own, global_, fetch_k=6)
        ids = [r["id"] for r in merged]
        assert len(ids) == 6
        # own gets ceil(half)=3, global fills the rest -> at least 3 global rows.
        assert sum(1 for i in ids if i.startswith("g")) >= 3
        assert sum(1 for i in ids if i.startswith("o")) >= 3

    def test_own_first_ordering_preserved(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [_row(f"o{i}") for i in range(6)]
        global_ = [_row(f"g{i}") for i in range(6)]
        merged = merge_project_first(own, global_, fetch_k=6)
        ids = [r["id"] for r in merged]
        # Every own row that made the cut must precede every global row.
        last_own = max(i for i, x in enumerate(ids) if x.startswith("o"))
        first_glob = min(i for i, x in enumerate(ids) if x.startswith("g"))
        assert last_own < first_glob

    def test_global_empty_own_fills_all(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [_row(f"o{i}") for i in range(6)]
        merged = merge_project_first(own, [], fetch_k=6)
        assert [r["id"] for r in merged] == [f"o{i}" for i in range(6)]

    def test_own_empty_pure_global(self):
        from scripts.core.recall_learnings import merge_project_first

        global_ = [_row(f"g{i}") for i in range(4)]
        merged = merge_project_first([], global_, fetch_k=6)
        assert [r["id"] for r in merged] == [f"g{i}" for i in range(4)]

    def test_backfill_from_leftover_own_when_global_short(self):
        from scripts.core.recall_learnings import merge_project_first

        # fetch_k=6 -> own quota 3, global has only 1 -> 2 slots backfilled
        # from leftover own rows (o3, o4), all own-first ordered.
        own = [_row(f"o{i}") for i in range(6)]
        global_ = [_row("g0")]
        merged = merge_project_first(own, global_, fetch_k=6)
        ids = [r["id"] for r in merged]
        assert len(ids) == 6
        assert ids[-1] == "g0"
        assert ids[:5] == ["o0", "o1", "o2", "o3", "o4"]

    def test_overlap_dedupe_by_id(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [{"id": "a", "src": "own"}]
        global_ = [{"id": "a", "src": "global"}, {"id": "b", "src": "global"}]
        merged = merge_project_first(own, global_, fetch_k=10)
        assert [r["id"] for r in merged] == ["a", "b"]
        assert merged[0]["src"] == "own"  # own copy wins on collision

    def test_does_not_mutate_inputs_quota(self):
        from scripts.core.recall_learnings import merge_project_first

        own = [_row("o0"), _row("o1")]
        global_ = [_row("g0")]
        merge_project_first(own, global_, fetch_k=2)
        assert [r["id"] for r in own] == ["o0", "o1"]
        assert [r["id"] for r in global_] == ["g0"]


# ==================== Round-2 Finding 3: independent pass failure isolation ==


class TestDispatchProjectFirstFailureIsolation:
    """A failure in one pass must not discard the other pass's results
    (issue #139 review round 2)."""

    async def test_global_pass_failure_returns_scoped(self, monkeypatch, capsys):
        import scripts.core.recall_learnings as rl

        async def fake_dispatch(params, *, project=None):
            if project is None:
                raise RuntimeError("global pass boom")
            return [{"id": "own1"}, {"id": "own2"}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        params = {"mode": "text_only", "query": "q", "k": 10, "project_scope": "opc"}
        results = await rl._dispatch_search_project_first(params)
        assert [r["id"] for r in results] == ["own1", "own2"]
        err = capsys.readouterr().err
        assert "global" in err.lower()

    async def test_scoped_pass_failure_returns_global(self, monkeypatch, capsys):
        import scripts.core.recall_learnings as rl

        async def fake_dispatch(params, *, project=None):
            if project is not None:
                raise RuntimeError("scoped pass boom")
            return [{"id": "glob1"}, {"id": "glob2"}]

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        params = {"mode": "text_only", "query": "q", "k": 10, "project_scope": "opc"}
        results = await rl._dispatch_search_project_first(params)
        assert [r["id"] for r in results] == ["glob1", "glob2"]
        err = capsys.readouterr().err
        assert "scoped" in err.lower()

    async def test_both_passes_fail_propagates(self, monkeypatch):
        import scripts.core.recall_learnings as rl

        async def fake_dispatch(params, *, project=None):
            raise RuntimeError("both boom")

        monkeypatch.setattr(rl, "_dispatch_search", fake_dispatch)
        params = {"mode": "text_only", "query": "q", "k": 10, "project_scope": "opc"}
        with pytest.raises(RuntimeError):
            await rl._dispatch_search_project_first(params)
