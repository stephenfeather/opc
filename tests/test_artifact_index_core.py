"""Tests for artifact_index_core pure functions."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.core.artifact_index_core import (
    OUTCOME_MAP,
    adapt_for_postgres,
    classify_file,
    convert_pg_upsert,
    extract_files,
    extract_sections,
    extract_session_info,
    generate_file_id,
    normalize_outcome,
    parse_continuity_content,
    parse_frontmatter,
    parse_handoff_content,
    parse_handoff_yaml_content,
    parse_plan_content,
    parse_simple_yaml,
)

# =============================================================================
# parse_frontmatter
# =============================================================================


class TestParseFrontmatter:
    """Tests for parse_frontmatter(content) -> (dict, str)."""

    def test_valid_frontmatter(self):
        content = "---\ntitle: My Doc\nstatus: done\n---\nBody text here"
        fm, body = parse_frontmatter(content)
        assert fm == {"title": "My Doc", "status": "done"}
        assert body == "\nBody text here"

    def test_no_frontmatter(self):
        content = "Just plain text"
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == "Just plain text"

    def test_empty_string(self):
        fm, body = parse_frontmatter("")
        assert fm == {}
        assert body == ""

    def test_incomplete_frontmatter_missing_closing(self):
        content = "---\ntitle: Missing Close"
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_frontmatter_with_colon_in_value(self):
        content = "---\nurl: http://example.com:8080/path\n---\nBody"
        fm, body = parse_frontmatter(content)
        assert fm["url"] == "http://example.com:8080/path"

    def test_frontmatter_preserves_remaining_content(self):
        content = "---\nkey: val\n---\n\n## Section\nMore content"
        fm, body = parse_frontmatter(content)
        assert "## Section" in body
        assert "More content" in body

    def test_frontmatter_with_empty_value(self):
        content = "---\nkey:\n---\nBody"
        fm, body = parse_frontmatter(content)
        assert fm["key"] == ""


# =============================================================================
# extract_sections
# =============================================================================


class TestExtractSections:
    """Tests for extract_sections(content, level) -> dict."""

    def test_h2_sections(self):
        content = "## Summary\nSummary text\n## Details\nDetail text"
        sections = extract_sections(content)
        assert "summary" in sections
        assert "details" in sections
        assert sections["summary"] == "Summary text"
        assert sections["details"] == "Detail text"

    def test_h3_sections(self):
        content = "### First\nA\n### Second\nB"
        sections = extract_sections(content, level=3)
        assert "first" in sections
        assert "second" in sections

    def test_empty_content(self):
        assert extract_sections("") == {}

    def test_no_sections(self):
        content = "Just plain text with no headings"
        assert extract_sections(content) == {}

    def test_section_name_normalization(self):
        content = "## What Worked\nStuff\n## Key Decisions\nThings"
        sections = extract_sections(content)
        assert "what_worked" in sections
        assert "key_decisions" in sections

    def test_section_ends_at_higher_level_heading(self):
        content = "## Section A\nContent A\n# Top Level\nTop content"
        sections = extract_sections(content)
        assert "section_a" in sections
        assert sections["section_a"] == "Content A"

    def test_last_section_captured(self):
        content = "## Only Section\nAll the content\nMore lines"
        sections = extract_sections(content)
        assert "only_section" in sections
        assert "More lines" in sections["only_section"]

    def test_multiline_section_content(self):
        content = "## Findings\nLine 1\nLine 2\nLine 3"
        sections = extract_sections(content)
        assert "Line 1" in sections["findings"]
        assert "Line 3" in sections["findings"]


# =============================================================================
# extract_session_info
# =============================================================================


class TestExtractSessionInfo:
    """Tests for extract_session_info(file_path) -> (name, uuid|None)."""

    def test_simple_session_name(self):
        path = Path("thoughts/shared/handoffs/my-session/task-01.md")
        name, uuid = extract_session_info(path)
        assert name == "my-session"
        assert uuid is None

    def test_session_with_uuid_suffix(self):
        path = Path("thoughts/shared/handoffs/auth-refactor-550e8400/task-01.md")
        name, uuid = extract_session_info(path)
        assert name == "auth-refactor"
        assert uuid == "550e8400"

    def test_uuid_suffix_case_insensitive(self):
        path = Path("thoughts/shared/handoffs/my-session-AABBCCDD/file.md")
        name, uuid = extract_session_info(path)
        assert name == "my-session"
        assert uuid == "aabbccdd"

    def test_no_handoffs_in_path(self):
        path = Path("thoughts/shared/plans/my-plan.md")
        name, uuid = extract_session_info(path)
        assert name == ""
        assert uuid is None

    def test_handoffs_at_end_of_path(self):
        path = Path("thoughts/shared/handoffs")
        name, uuid = extract_session_info(path)
        assert name == ""
        assert uuid is None

    def test_deeply_nested_handoff(self):
        path = Path("thoughts/shared/handoffs/deep-session/sub/file.md")
        name, uuid = extract_session_info(path)
        assert name == "deep-session"
        assert uuid is None


# =============================================================================
# normalize_outcome
# =============================================================================


class TestNormalizeOutcome:
    """Tests for normalize_outcome(status) -> str."""

    def test_success_variants(self):
        assert normalize_outcome("SUCCESS") == "SUCCEEDED"
        assert normalize_outcome("SUCCEEDED") == "SUCCEEDED"

    def test_partial_variants(self):
        assert normalize_outcome("PARTIAL") == "PARTIAL_PLUS"
        assert normalize_outcome("PARTIAL_PLUS") == "PARTIAL_PLUS"
        assert normalize_outcome("PARTIAL_MINUS") == "PARTIAL_MINUS"

    def test_failure_variants(self):
        assert normalize_outcome("FAILED") == "FAILED"
        assert normalize_outcome("FAILURE") == "FAILED"

    def test_unknown(self):
        assert normalize_outcome("UNKNOWN") == "UNKNOWN"

    def test_case_insensitive(self):
        assert normalize_outcome("success") == "SUCCEEDED"
        assert normalize_outcome("Failed") == "FAILED"

    def test_unrecognized_maps_to_unknown(self):
        assert normalize_outcome("COMPLETED") == "UNKNOWN"
        assert normalize_outcome("") == "UNKNOWN"

    def test_outcome_map_is_dict(self):
        assert isinstance(OUTCOME_MAP, dict)
        assert "SUCCESS" in OUTCOME_MAP
        assert "SUCCEEDED" in OUTCOME_MAP


# =============================================================================
# extract_files
# =============================================================================


class TestExtractFiles:
    """Tests for extract_files(content) -> list."""

    def test_backtick_file_paths(self):
        content = "- `scripts/core/main.py`\n- `tests/test_main.py`"
        files = extract_files(content)
        assert "scripts/core/main.py" in files
        assert "tests/test_main.py" in files

    def test_backtick_with_line_number(self):
        content = "- `scripts/core/main.py:42`"
        files = extract_files(content)
        assert "scripts/core/main.py" in files
        assert len(files) == 1  # line number stripped

    def test_bold_file_format(self):
        content = "**File**: `scripts/core/main.py`"
        files = extract_files(content)
        assert "scripts/core/main.py" in files

    def test_no_files(self):
        content = "Just plain text with no file references"
        assert extract_files(content) == []

    def test_empty_string(self):
        assert extract_files("") == []

    def test_multiple_files_per_line(self):
        content = "Changed `a.py` and `b.py` in this commit"
        files = extract_files(content)
        assert "a.py" in files
        assert "b.py" in files


# =============================================================================
# parse_simple_yaml
# =============================================================================


class TestParseSimpleYaml:
    """Tests for parse_simple_yaml(text) -> dict."""

    def test_simple_key_value(self):
        text = "session: my-session\ndate: 2026-04-05\nstatus: partial"
        result = parse_simple_yaml(text)
        assert result["session"] == "my-session"
        assert result["date"] == "2026-04-05"
        assert result["status"] == "partial"

    def test_simple_list(self):
        text = 'blockers:\n- "Item one"\n- "Item two"'
        result = parse_simple_yaml(text)
        assert result["blockers"] == ["Item one", "Item two"]

    def test_empty_list(self):
        text = "blockers: []"
        result = parse_simple_yaml(text)
        assert result["blockers"] == []

    def test_empty_list_with_items(self):
        text = "blockers:\n"
        result = parse_simple_yaml(text)
        assert result["blockers"] == []

    def test_inline_list(self):
        text = 'files: [a.py, b.py, "c.py"]'
        result = parse_simple_yaml(text)
        assert "a.py" in result["files"]
        assert "b.py" in result["files"]
        assert "c.py" in result["files"]

    def test_dict_style_list_items(self):
        text = "done_this_session:\n- task: Fix the bug\n    files: [a.py, b.py]"
        result = parse_simple_yaml(text)
        items = result["done_this_session"]
        assert len(items) == 1
        assert items[0]["task"] == "Fix the bug"
        assert isinstance(items[0]["files"], list)

    def test_comments_ignored(self):
        text = "# This is a comment\nkey: value"
        result = parse_simple_yaml(text)
        assert result["key"] == "value"
        assert "#" not in str(result.keys())

    def test_empty_string(self):
        result = parse_simple_yaml("")
        assert result == {}

    def test_quoted_values_stripped(self):
        text = 'key: "quoted value"'
        result = parse_simple_yaml(text)
        assert result["key"] == "quoted value"

    def test_multiple_dict_list_items(self):
        text = (
            "decisions:\n"
            "- name: decision_one\n"
            '  value: "First decision"\n'
            "- name: decision_two\n"
            '  value: "Second decision"'
        )
        result = parse_simple_yaml(text)
        decisions = result["decisions"]
        assert len(decisions) == 2
        assert decisions[0]["name"] == "decision_one"
        assert decisions[1]["name"] == "decision_two"


# =============================================================================
# adapt_for_postgres
# =============================================================================


class TestAdaptForPostgres:
    """Tests for adapt_for_postgres(sql, params, table_hint) -> (sql, params)."""

    def test_question_mark_to_percent_s(self):
        sql = "INSERT INTO plans (id, name) VALUES (?, ?)"
        new_sql, new_params = adapt_for_postgres(sql, ("id1", "name1"), "plans")
        assert "?" not in new_sql
        assert "%s" in new_sql
        assert new_params == ("id1", "name1")

    def test_handoffs_table_rewrite(self):
        # 16 params for handoffs insert (includes session_uuid)
        params = tuple(f"val{i}" for i in range(16))
        sql = "INSERT INTO handoffs (col) VALUES (?)"
        new_sql, new_params = adapt_for_postgres(sql, params, "handoffs")
        assert "gen_random_uuid()" in new_sql
        assert "ON CONFLICT" in new_sql
        assert len(new_params) == 10  # reordered to 10 params (added session_uuid)

    def test_handoffs_param_reorder(self):
        # Verify specific param positions are mapped correctly
        # SQLite column order (16 params):
        #   0=id, 1=session_name, 2=session_uuid, 3=task_number, 4=file_path,
        #   5=task_summary, 6=what_worked, 7=what_failed, 8=key_decisions,
        #   9=files_modified, 10=outcome, 11=root_span_id, 12=turn_span_id,
        #   13=session_id, 14=braintrust_session_id, 15=created_at
        params = tuple(f"p{i}" for i in range(16))
        sql = "INSERT INTO handoffs (col) VALUES (?)"
        _, new_params = adapt_for_postgres(sql, params, "handoffs")
        assert new_params[0] == "p1"   # session_name
        assert new_params[1] == "p2"   # session_uuid
        assert new_params[2] == "p4"   # file_path
        assert new_params[3] == "p5"   # task_summary -> goal
        assert new_params[4] == "p6"   # what_worked
        assert new_params[5] == "p7"   # what_failed
        assert new_params[6] == "p8"   # key_decisions
        assert new_params[7] == "p10"  # outcome
        assert new_params[8] == "p11"  # root_span_id
        assert new_params[9] == "p13"  # session_id

    def test_handoffs_session_uuid_in_sql(self):
        """session_uuid column appears in the PG INSERT and ON CONFLICT UPDATE."""
        params = tuple(f"val{i}" for i in range(16))
        sql = "INSERT INTO handoffs (col) VALUES (?)"
        new_sql, _ = adapt_for_postgres(sql, params, "handoffs")
        assert "session_uuid" in new_sql

    def test_handoffs_coalesce_preserves_existing_uuid(self):
        """ON CONFLICT uses COALESCE so NULL incoming doesn't clobber existing."""
        params = tuple(f"val{i}" for i in range(16))
        sql = "INSERT INTO handoffs (col) VALUES (?)"
        new_sql, _ = adapt_for_postgres(sql, params, "handoffs")
        assert "COALESCE(EXCLUDED.session_uuid, handoffs.session_uuid)" in new_sql

    def test_handoffs_wrong_param_count_raises(self):
        """Fail fast when handoff params tuple has unexpected length."""
        import pytest
        params = tuple(f"val{i}" for i in range(15))  # wrong count
        sql = "INSERT INTO handoffs (col) VALUES (?)"
        with pytest.raises(ValueError, match="Expected 16 handoff params"):
            adapt_for_postgres(sql, params, "handoffs")

    def test_insert_or_replace_converts(self):
        sql = "INSERT OR REPLACE INTO plans (id, name) VALUES (?, ?)"
        new_sql, _ = adapt_for_postgres(sql, ("id1", "name1"), "plans")
        assert "INSERT OR REPLACE" not in new_sql
        assert "ON CONFLICT" in new_sql

    def test_plain_insert_unchanged(self):
        sql = "INSERT INTO other (id, val) VALUES (?, ?)"
        new_sql, params = adapt_for_postgres(sql, ("id1", "val1"), "other")
        assert "ON CONFLICT" not in new_sql
        assert params == ("id1", "val1")


# =============================================================================
# convert_pg_upsert
# =============================================================================


class TestConvertPgUpsert:
    """Tests for convert_pg_upsert(sql) -> str."""

    def test_basic_conversion(self):
        sql = "INSERT OR REPLACE INTO plans (id, name, status) VALUES (%s, %s, %s)"
        result = convert_pg_upsert(sql)
        assert "INSERT INTO plans" in result
        assert "INSERT OR REPLACE" not in result
        assert "ON CONFLICT (id) DO UPDATE SET" in result
        assert "name = EXCLUDED.name" in result
        assert "status = EXCLUDED.status" in result

    def test_id_excluded_from_update(self):
        sql = "INSERT OR REPLACE INTO plans (id, name) VALUES (%s, %s)"
        result = convert_pg_upsert(sql)
        assert "id = EXCLUDED.id" not in result

    def test_no_match_returns_unchanged(self):
        sql = "SELECT * FROM plans"
        result = convert_pg_upsert(sql)
        assert result == sql

    def test_case_insensitive(self):
        sql = "insert or replace into plans (id, title) values (%s, %s)"
        result = convert_pg_upsert(sql)
        assert "ON CONFLICT" in result

    def test_trailing_semicolon_removed(self):
        sql = "INSERT OR REPLACE INTO plans (id, name) VALUES (%s, %s);"
        result = convert_pg_upsert(sql)
        assert not result.endswith(";")
        assert "ON CONFLICT" in result


# =============================================================================
# generate_file_id
# =============================================================================


class TestGenerateFileId:
    """Tests for generate_file_id(file_path) -> str."""

    def test_returns_12_char_hex(self):
        result = generate_file_id("some/path/file.md")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        a = generate_file_id("same/path.md")
        b = generate_file_id("same/path.md")
        assert a == b

    def test_different_paths_differ(self):
        a = generate_file_id("path/a.md")
        b = generate_file_id("path/b.md")
        assert a != b


# =============================================================================
# classify_file
# =============================================================================


class TestClassifyFile:
    """Tests for classify_file(file_path) -> str | None."""

    def test_handoff_md(self):
        path = Path("thoughts/shared/handoffs/my-session/task-01.md")
        assert classify_file(path) == "handoff"

    def test_handoff_yaml(self):
        path = Path("thoughts/shared/handoffs/session/handoff.yaml")
        assert classify_file(path) == "handoff_yaml"

    def test_handoff_yml(self):
        path = Path("thoughts/shared/handoffs/session/handoff.yml")
        assert classify_file(path) == "handoff_yaml"

    def test_plan(self):
        path = Path("thoughts/shared/plans/my-plan.md")
        assert classify_file(path) == "plan"

    def test_continuity(self):
        path = Path("CONTINUITY_CLAUDE-my-session.md")
        assert classify_file(path) == "continuity"

    def test_unknown_file(self):
        path = Path("random/file.txt")
        assert classify_file(path) is None

    def test_plan_non_md(self):
        path = Path("thoughts/shared/plans/data.json")
        assert classify_file(path) is None


# =============================================================================
# parse_handoff_content
# =============================================================================


class TestParseHandoffContent:
    """Tests for parse_handoff_content(raw_content, file_path) -> dict."""

    def test_basic_handoff(self):
        content = (
            "---\nstatus: SUCCESS\ndate: 2026-01-01\n---\n"
            "## Summary\nDid the thing\n"
            "## What Worked\nEverything\n"
            "## What Failed\nNothing\n"
        )
        path = Path("thoughts/shared/handoffs/my-session/task-01.md")
        result = parse_handoff_content(content, path)
        assert result["session_name"] == "my-session"
        assert result["outcome"] == "SUCCEEDED"
        assert result["task_number"] == 1
        assert "Did the thing" in result["task_summary"]
        assert result["what_worked"] == "Everything"
        assert result["what_failed"] == "Nothing"

    def test_frontmatter_fields_extracted(self):
        content = (
            "---\nstatus: PARTIAL\nroot_span_id: abc123\n"
            "session_id: sess456\ndate: 2026-03-15\n---\nBody"
        )
        path = Path("thoughts/shared/handoffs/test-sess/task-02.md")
        result = parse_handoff_content(content, path)
        assert result["outcome"] == "PARTIAL_PLUS"
        assert result["root_span_id"] == "abc123"
        assert result["session_id"] == "sess456"
        assert result["created_at"] == "2026-03-15"
        assert result["task_number"] == 2

    def test_no_frontmatter(self):
        content = "## Summary\nJust text"
        path = Path("thoughts/shared/handoffs/sess/task-01.md")
        result = parse_handoff_content(content, path)
        assert result["outcome"] == "UNKNOWN"
        assert result["id"]  # still generates an ID

    def test_uuid_in_path(self):
        content = "---\nstatus: SUCCEEDED\n---\n## Summary\nDone"
        path = Path("thoughts/shared/handoffs/auth-refactor-aabb1122/task-01.md")
        result = parse_handoff_content(content, path)
        assert result["session_name"] == "auth-refactor"
        assert result["session_uuid"] == "aabb1122"

    def test_files_modified_is_json(self):
        content = (
            "---\nstatus: SUCCEEDED\n---\n"
            "## Files Modified\n- `src/main.py`\n- `tests/test_main.py`"
        )
        path = Path("thoughts/shared/handoffs/sess/task-01.md")
        result = parse_handoff_content(content, path)
        files = json.loads(result["files_modified"])
        assert "src/main.py" in files

    def test_h3_sections_merged(self):
        content = (
            "---\nstatus: SUCCEEDED\n---\n"
            "## Overview\nTop level\n"
            "### Decisions\nSome decisions here"
        )
        path = Path("thoughts/shared/handoffs/sess/task-01.md")
        result = parse_handoff_content(content, path)
        assert result["key_decisions"] == "Some decisions here"


# =============================================================================
# parse_handoff_yaml_content
# =============================================================================


class TestParseHandoffYamlContent:
    """Tests for parse_handoff_yaml_content(raw_content, file_path) -> dict."""

    def test_basic_yaml_handoff(self):
        content = (
            "---\nsession: my-session\nstatus: partial\ndate: 2026-04-01\n---\n"
            "goal: Do the thing\n"
            "done_this_session:\n"
            "- task: Fixed the bug\n"
            "worked:\n"
            '- "Pattern A worked"\n'
            "failed: []\n"
        )
        path = Path("thoughts/shared/handoffs/my-session/handoff.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert result["session_name"] == "my-session"
        assert result["outcome"] == "PARTIAL_PLUS"
        assert "Fixed the bug" in result["task_summary"]
        assert "Pattern A worked" in result["what_worked"]

    def test_decisions_extracted(self):
        content = (
            "---\nsession: test\nstatus: SUCCESS\n---\n"
            "decisions:\n"
            "- name: chose_option_a\n"
            '  value: "Because it was simpler"\n'
        )
        path = Path("thoughts/shared/handoffs/test/h.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert "chose_option_a" in result["key_decisions"]

    def test_files_section(self):
        content = (
            "---\nsession: test\nstatus: SUCCESS\n---\n"
            "done_this_session:\n"
            "- task: Did work\n"
            "    files: [new_file.py, old_file.py]\n"
        )
        path = Path("thoughts/shared/handoffs/test/h.yaml")
        result = parse_handoff_yaml_content(content, path)
        # files come from the done_this_session items, not a top-level files key
        assert "Did work" in result["task_summary"]

    def test_top_level_files_inline(self):
        content = (
            "---\nsession: test\nstatus: SUCCESS\n---\n"
            "files: [new_file.py, old_file.py]\n"
        )
        path = Path("thoughts/shared/handoffs/test/h.yaml")
        result = parse_handoff_yaml_content(content, path)
        files = json.loads(result["files_modified"])
        assert "new_file.py" in files
        assert "old_file.py" in files

    def test_session_from_path_fallback(self):
        content = "---\nstatus: SUCCESS\n---\ngoal: stuff"
        path = Path("thoughts/shared/handoffs/fallback-sess/h.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert result["session_name"] == "fallback-sess"

    def test_outcome_from_body(self):
        content = "---\nsession: test\n---\noutcome: FAILED"
        path = Path("thoughts/shared/handoffs/test/h.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert result["outcome"] == "FAILED"

    def test_uuid_in_yaml_path(self):
        content = "---\nsession: auth-refactor\nstatus: SUCCEEDED\n---\ngoal: stuff"
        path = Path("thoughts/shared/handoffs/auth-refactor-aabb1122/handoff.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert result["session_uuid"] == "aabb1122"
        assert result["session_name"] == "auth-refactor"

    def test_uuid_from_path_when_no_frontmatter_session(self):
        content = "---\nstatus: SUCCESS\n---\ngoal: stuff"
        path = Path("thoughts/shared/handoffs/my-task-deadbeef/handoff.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert result["session_uuid"] == "deadbeef"
        assert result["session_name"] == "my-task"

    def test_no_uuid_in_path_gives_none(self):
        content = "---\nsession: plain-sess\nstatus: SUCCESS\n---\ngoal: stuff"
        path = Path("thoughts/shared/handoffs/plain-sess/handoff.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert result["session_uuid"] is None

    def test_auto_handoff_session_uuid_is_none(self):
        """Auto-generated handoffs use auto/<session_id>.yaml — no dir UUID."""
        content = "---\nsession: auto-sess\nstatus: SUCCESS\n---\ngoal: stuff"
        path = Path("thoughts/shared/handoffs/auto/abc123.yaml")
        result = parse_handoff_yaml_content(content, path)
        assert result["session_uuid"] is None


# =============================================================================
# parse_plan_content
# =============================================================================


class TestParsePlanContent:
    """Tests for parse_plan_content(content, file_path) -> dict."""

    def test_basic_plan(self):
        content = (
            "# My Great Plan\n\n"
            "## Overview\nThis plan does things\n\n"
            "## Approach\nWe will do it carefully\n"
        )
        path = Path("thoughts/shared/plans/my-great-plan.md")
        result = parse_plan_content(content, path)
        assert result["title"] == "My Great Plan"
        assert "This plan does things" in result["overview"]
        assert "carefully" in result["approach"]

    def test_title_from_filename_fallback(self):
        content = "## Overview\nNo H1 heading here"
        path = Path("thoughts/shared/plans/fallback-name.md")
        result = parse_plan_content(content, path)
        assert result["title"] == "fallback-name"

    def test_phases_extracted(self):
        content = (
            "# Plan\n\n"
            "## Phase 1\nDo first thing\n\n"
            "## Phase 2\nDo second thing\n"
        )
        path = Path("thoughts/shared/plans/phased.md")
        result = parse_plan_content(content, path)
        phases = json.loads(result["phases"])
        assert len(phases) >= 2

    def test_constraints_section(self):
        content = (
            "# Plan\n\n"
            "## Overview\nStuff\n\n"
            "## What We're Not Doing\nNo gold plating\n"
        )
        path = Path("thoughts/shared/plans/constrained.md")
        result = parse_plan_content(content, path)
        assert "gold plating" in result["constraints"]

    def test_implementation_approach_alias(self):
        content = (
            "# Plan\n\n"
            "## Implementation Approach\nThe strategy here\n"
        )
        path = Path("thoughts/shared/plans/impl.md")
        result = parse_plan_content(content, path)
        assert "strategy" in result["approach"]

    def test_id_is_deterministic(self):
        content = "# Plan\n## Overview\nStuff"
        path = Path("thoughts/shared/plans/test.md")
        a = parse_plan_content(content, path)
        b = parse_plan_content(content, path)
        assert a["id"] == b["id"]


# =============================================================================
# parse_continuity_content
# =============================================================================


class TestParseContinuityContent:
    """Tests for parse_continuity_content(content, file_path) -> dict."""

    def test_basic_continuity(self):
        content = (
            "## Goal\nRefactor the system\n\n"
            "## State\n[x] Did step 1\n[->] Working on step 2\n[ ] Step 3 pending\n\n"
            "## Key Learnings\nLearned a lot\n"
        )
        path = Path("CONTINUITY_CLAUDE-my-session.md")
        result = parse_continuity_content(content, path)
        assert result["session_name"] == "my-session"
        assert "Refactor the system" in result["goal"]
        assert "step 2" in result["state_now"]
        assert "Step 3" in result["state_next"]
        assert "Learned a lot" in result["key_learnings"]

    def test_session_name_from_filename(self):
        path = Path("CONTINUITY_CLAUDE-auth-refactor.md")
        content = "## Goal\nDo auth stuff"
        result = parse_continuity_content(content, path)
        assert result["session_name"] == "auth-refactor"

    def test_fallback_session_name(self):
        path = Path("OTHER_FORMAT.md")
        content = "## Goal\nSomething"
        result = parse_continuity_content(content, path)
        assert result["session_name"] == "OTHER_FORMAT"

    def test_state_done_is_json(self):
        content = "## State\n[x] Done item 1\n[x] Done item 2"
        path = Path("CONTINUITY_CLAUDE-test.md")
        result = parse_continuity_content(content, path)
        done = json.loads(result["state_done"])
        assert len(done) == 2

    def test_key_decisions(self):
        content = (
            "## Goal\nStuff\n\n"
            "## Key Decisions\nDecided to use Python\n"
        )
        path = Path("CONTINUITY_CLAUDE-test.md")
        result = parse_continuity_content(content, path)
        assert "Python" in result["key_decisions"]

    def test_empty_state(self):
        content = "## Goal\nJust a goal, no state"
        path = Path("CONTINUITY_CLAUDE-test.md")
        result = parse_continuity_content(content, path)
        assert result["state_now"] == ""
        assert result["state_next"] == ""
        assert json.loads(result["state_done"]) == []


# =============================================================================
# CLI regression test — script must be executable by path
# =============================================================================


class TestCLIEntrypoint:
    """Verify artifact_index.py can be invoked directly by path."""

    def test_help_flag_works_by_path(self):
        result = subprocess.run(
            [sys.executable, "scripts/core/artifact_index.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Index context graph artifacts" in result.stdout
