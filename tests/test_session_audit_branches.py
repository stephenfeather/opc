"""UUID ancestry and branch annotation tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.core.session_audit.parser import events_share_lineage, parse_session


def _message(
    uuid: str,
    parent_uuid: str | None,
    text: str,
    *,
    record_type: str = "assistant",
    sidechain: bool = False,
) -> dict[str, object]:
    return {
        "type": record_type,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "isSidechain": sidechain,
        "message": {"content": text},
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_forks_preserve_chronology_and_mark_abandoned_sibling_work(tmp_path: Path) -> None:
    path = tmp_path / "forked.jsonl"
    _write_jsonl(
        path,
        [
            _message("root", None, "start"),
            _message("branch-a", "root", "abandoned work"),
            _message("branch-b", "root", "replacement work"),
            _message("main-tip", "branch-b", "continue", record_type="user"),
        ],
    )

    parsed = parse_session(path)

    assert [event.record_uuid for event in parsed.events] == [
        "root",
        "branch-a",
        "branch-b",
        "main-tip",
    ]
    assert parsed.events[0].lineage == ("root",)
    assert parsed.events[0].branch_path == ()
    assert parsed.events[1].lineage == ("root", "branch-a")
    assert parsed.events[1].branch_path == ("branch-a",)
    assert parsed.events[1].is_abandoned_branch is True
    assert parsed.events[2].branch_path == ("branch-b",)
    assert parsed.events[2].is_abandoned_branch is False
    assert parsed.events[3].lineage == ("root", "branch-b", "main-tip")
    assert parsed.events[3].branch_path == ("branch-b",)
    assert parsed.diagnostics.forks == 1
    assert events_share_lineage(parsed.events[0], parsed.events[1]) is True
    assert events_share_lineage(parsed.events[1], parsed.events[2]) is False


def test_invalid_ancestry_edges_are_diagnosed_and_excluded(tmp_path: Path) -> None:
    path = tmp_path / "invalid-ancestry.jsonl"
    _write_jsonl(
        path,
        [
            _message("dangling", "missing", "dangling parent"),
            _message("cycle-a", "cycle-b", "cycle a"),
            _message("cycle-b", "cycle-a", "cycle b"),
            _message("duplicate", None, "first duplicate"),
            _message("duplicate", None, "second duplicate"),
            _message("ambiguous-child", "duplicate", "ambiguous parent"),
            _message("tip", "dangling", "main tip", record_type="user"),
        ],
    )

    parsed = parse_session(path)

    dangling = parsed.events[0]
    cycle_a, cycle_b = parsed.events[1:3]
    duplicate_a, duplicate_b = parsed.events[3:5]
    assert dangling.lineage == ("dangling",)
    assert dangling.lineage_valid is True
    assert "dangling_parent" in dangling.warnings
    assert cycle_a.lineage == cycle_b.lineage == ()
    assert cycle_a.lineage_valid is cycle_b.lineage_valid is False
    assert "cyclic_ancestry" in cycle_a.warnings
    assert duplicate_a.lineage == duplicate_b.lineage == ()
    assert duplicate_a.lineage_valid is duplicate_b.lineage_valid is False
    assert "duplicate_uuid" in duplicate_a.warnings
    ambiguous_child = parsed.events[5]
    assert ambiguous_child.lineage == ("ambiguous-child",)
    assert "ambiguous_parent_uuid" in ambiguous_child.warnings
    assert parsed.diagnostics.dangling_parents == 1
    assert parsed.diagnostics.cyclic_uuids == 2
    assert parsed.diagnostics.duplicate_uuids == 1
    assert parsed.diagnostics.ambiguous_parent_uuids == 1


def test_tolerated_message_failure_remains_a_metadata_only_lineage_bridge(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata-bridge.jsonl"
    _write_jsonl(
        path,
        [
            _message("root", None, "root"),
            {
                "type": "assistant",
                "uuid": "bridge",
                "parentUuid": "root",
                "message": {"content": {"unsupported": "shape"}},
            },
            _message("child", "bridge", "child", record_type="user"),
        ],
    )

    parsed = parse_session(path)

    assert parsed.diagnostics.failed_message_records == 1
    assert parsed.diagnostics.dangling_parents == 0
    assert [event.record_uuid for event in parsed.events] == ["root", "child"]
    assert events_share_lineage(parsed.events[0], parsed.events[1]) is True


def test_non_message_records_remain_metadata_only_lineage_bridges(tmp_path: Path) -> None:
    path = tmp_path / "non-message-bridges.jsonl"
    _write_jsonl(
        path,
        [
            _message("root", None, "root"),
            {"type": "attachment", "uuid": "attachment", "parentUuid": "root"},
            {"type": "system", "uuid": "system", "parentUuid": "attachment"},
            _message("child", "system", "child", record_type="user"),
        ],
    )

    parsed = parse_session(path)

    assert parsed.diagnostics.unknown_record_types == 1
    assert parsed.diagnostics.dangling_parents == 0
    assert [event.record_uuid for event in parsed.events] == ["root", "child"]
    assert events_share_lineage(parsed.events[0], parsed.events[1]) is True


def test_transparent_metadata_children_do_not_create_semantic_forks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transparent-siblings.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use",
                "message": {"content": [{"type": "tool_use", "id": "read-1", "name": "Read"}]},
            },
            {"type": "attachment", "uuid": "attachment", "parentUuid": "use"},
            {
                "type": "user",
                "uuid": "result",
                "parentUuid": "use",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "read-1"}]},
            },
            _message("continuation", "attachment", "continued answer"),
        ],
    )

    parsed = parse_session(path)

    use, result, continuation = parsed.events
    assert parsed.diagnostics.structural_forks == 1
    assert parsed.diagnostics.forks == 0
    assert events_share_lineage(use, result) is True
    assert events_share_lineage(use, continuation) is True
    assert events_share_lineage(result, continuation) is True
    assert use.correlated_event_id == result.event_id
    assert result.correlated_event_id == use.event_id
    assert not any(event.is_abandoned_branch for event in parsed.events)


def test_visible_sibling_turns_remain_an_incompatible_semantic_fork(
    tmp_path: Path,
) -> None:
    path = tmp_path / "semantic-siblings.jsonl"
    _write_jsonl(
        path,
        [
            _message("root", None, "start"),
            _message("first", "root", "first visible branch"),
            _message(
                "second",
                "root",
                "second visible branch",
                record_type="user",
            ),
        ],
    )

    parsed = parse_session(path)

    root, first, second = parsed.events
    assert parsed.diagnostics.structural_forks == 1
    assert parsed.diagnostics.forks == 1
    assert events_share_lineage(root, first) is True
    assert events_share_lineage(root, second) is True
    assert events_share_lineage(first, second) is False
    assert first.is_abandoned_branch is True
    assert second.is_abandoned_branch is False


def test_transparent_root_event_is_isolated_from_a_later_semantic_turn(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transparent-root.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "user",
                "uuid": "result",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "reused-id"}]},
            },
            {
                "type": "assistant",
                "uuid": "later-use",
                "parentUuid": "result",
                "message": {"content": [{"type": "tool_use", "id": "reused-id", "name": "Read"}]},
            },
        ],
    )

    parsed = parse_session(path)

    result, later_use = parsed.events
    assert result.lineage_valid is False
    assert result.lineage == ()
    assert events_share_lineage(result, later_use) is False
    assert result.correlated_event_id is None
    assert later_use.correlated_event_id is None


def test_disconnected_semantic_roots_are_not_abandoned_by_a_later_component(
    tmp_path: Path,
) -> None:
    path = tmp_path / "disconnected-components.jsonl"
    _write_jsonl(
        path,
        [
            _message("first-root", None, "first component"),
            _message(
                "first-tip",
                "first-root",
                "first continuation",
                record_type="user",
            ),
            _message("second-root", None, "resumed component"),
            _message(
                "second-tip",
                "second-root",
                "second continuation",
                record_type="user",
            ),
        ],
    )

    parsed = parse_session(path)

    first_root, first_tip, second_root, second_tip = parsed.events
    assert events_share_lineage(first_root, first_tip) is True
    assert events_share_lineage(second_root, second_tip) is True
    assert events_share_lineage(first_tip, second_root) is False
    assert not any(event.is_abandoned_branch for event in parsed.events)


def test_sidechain_only_transcript_remains_analyzable_and_labeled(tmp_path: Path) -> None:
    path = tmp_path / "sidechain.jsonl"
    root = _message("side-root", None, "subagent work", sidechain=True)
    root["sourceToolAssistantUUID"] = "outer-tool-use"
    _write_jsonl(
        path,
        [
            root,
            _message(
                "side-tip",
                "side-root",
                "subagent response",
                record_type="user",
                sidechain=True,
            ),
        ],
    )

    parsed = parse_session(path)

    assert parsed.diagnostics.sidechain_only is True
    assert all(event.is_sidechain for event in parsed.events)
    assert not any(event.is_abandoned_branch for event in parsed.events)
    assert parsed.events[0].source_tool_assistant_uuid == "outer-tool-use"
    assert parsed.events[1].lineage == ("side-root", "side-tip")


def test_sidechain_only_diagnostic_does_not_require_record_uuids(tmp_path: Path) -> None:
    path = tmp_path / "uuidless-sidechain.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "isSidechain": True,
                "message": {"content": "subagent work"},
            },
            {
                "type": "user",
                "isSidechain": True,
                "message": {"content": "subagent response"},
            },
        ],
    )

    parsed = parse_session(path)

    assert parsed.diagnostics.sidechain_only is True


def test_deep_lineage_is_iterative_and_retains_a_bounded_summary(tmp_path: Path) -> None:
    path = tmp_path / "deep-chain.jsonl"
    records = [
        _message(
            f"node-{index}",
            f"node-{index - 1}" if index else None,
            "event",
        )
        for index in range(2_000)
    ]
    _write_jsonl(path, records)

    parsed = parse_session(path)

    assert len(parsed.events) == 2_000
    tip = parsed.events[-1]
    assert tip.lineage_valid is True
    assert tip.lineage_root_uuid == "node-0"
    assert tip.lineage_depth == 1_999
    assert tip.lineage_truncated is True
    assert len(tip.lineage) <= 8
    assert tip.ancestry_start is not None
    assert tip.ancestry_end is not None
    assert "lineage_summary_truncated" in tip.warnings
    assert parsed.diagnostics.lineage_summaries_truncated > 0


def test_unpaired_surrogate_uuid_is_invalid_instead_of_crashing_fork_hashing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "surrogate-uuid.jsonl"
    _write_jsonl(
        path,
        [
            _message("root", None, "root"),
            _message("\ud800", "root", "invalid identifier"),
            _message("sibling", "root", "valid sibling"),
        ],
    )

    parsed = parse_session(path)

    assert len(parsed.events) == 3
    invalid = parsed.events[1]
    assert invalid.record_uuid is None
    assert invalid.lineage_valid is False
    assert "invalid_record_uuid" in invalid.warnings
    assert "no_valid_record_uuid" in invalid.warnings
    assert parsed.diagnostics.invalid_field_types == 1


def test_missing_record_uuid_is_explicitly_lineage_invalid(tmp_path: Path) -> None:
    path = tmp_path / "missing-uuid.jsonl"
    _write_jsonl(
        path,
        [{"type": "assistant", "message": {"content": "visible answer"}}],
    )

    parsed = parse_session(path)

    event = parsed.events[0]
    assert event.record_uuid is None
    assert event.lineage_valid is False
    assert "no_valid_record_uuid" in event.warnings
