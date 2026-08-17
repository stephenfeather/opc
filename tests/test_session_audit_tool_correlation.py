"""Tool parsing and branch-safe correlation tests."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.core.session_audit.parser as parser_module
from scripts.core.session_audit.models import (
    ContentKind,
    EventActor,
    EvidenceSourceKind,
)
from scripts.core.session_audit.parser import parse_session


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_tool_result_is_not_a_human_prompt_and_correlates_to_prior_use(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use-record",
                "timestamp": "2026-08-11T12:00:00Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {
                                "command": "OPENAI_API_KEY=sk-12345678 pytest",
                                "bulk": "must not be retained",
                            },
                        }
                    ]
                },
            },
            {
                "type": "user",
                "uuid": "result-record",
                "parentUuid": "use-record",
                "timestamp": "2026-08-11T12:00:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "Bearer secret-token failed",
                        }
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    assert len(parsed.events) == 2
    use, result = parsed.events
    assert use.actor is EventActor.ASSISTANT
    assert use.kind is ContentKind.TOOL_USE
    assert use.source_kind is EvidenceSourceKind.TOOL_USE
    assert use.tool_use_id == "tool-1"
    assert use.tool_name == "Bash"
    assert dict(use.tool_input) == {"command": "OPENAI_API_KEY=<redacted-secret> pytest"}
    assert use.correlated_event_id == result.event_id
    assert result.actor is EventActor.TOOL
    assert result.kind is ContentKind.TOOL_RESULT
    assert result.source_kind is EvidenceSourceKind.TOOL_RESULT
    assert result.text == "Bearer <redacted-secret> failed"
    assert result.tool_result_is_error is None
    assert result.correlated_event_id == use.event_id
    assert not any(event.actor is EventActor.HUMAN for event in parsed.events)


def test_allowlisted_tool_input_replaces_unpaired_surrogates(tmp_path: Path) -> None:
    path = tmp_path / "unicode-tool-input.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use-record",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "printf '\ud800'"},
                        }
                    ]
                },
            }
        ],
    )

    parsed = parse_session(path)

    use = parsed.events[0]
    assert dict(use.tool_input) == {"command": "printf '\ufffd'"}
    assert "unpaired_surrogate_replaced" in use.warnings
    assert parsed.diagnostics.unicode_replacement_events == 1


def test_results_before_uses_and_reversed_result_order_correlate_by_id(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "user",
                "uuid": "result-record",
                "parentUuid": "use-record",
                "timestamp": "2026-08-11T12:00:02Z",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool-2", "content": "two"},
                        {"type": "tool_result", "tool_use_id": "tool-1", "content": "one"},
                    ]
                },
            },
            {
                "type": "assistant",
                "uuid": "use-record",
                "timestamp": "2026-08-11T12:00:00Z",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}},
                        {"type": "tool_use", "id": "tool-2", "name": "Read", "input": {}},
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    by_id_and_kind = {(event.tool_use_id, event.kind): event for event in parsed.events}
    for tool_id in ("tool-1", "tool-2"):
        use = by_id_and_kind[(tool_id, ContentKind.TOOL_USE)]
        result = by_id_and_kind[(tool_id, ContentKind.TOOL_RESULT)]
        assert use.correlated_event_id == result.event_id
        assert result.correlated_event_id == use.event_id


def test_result_ancestor_of_later_use_is_not_a_causal_tool_pair(tmp_path: Path) -> None:
    path = tmp_path / "impossible-tool-ancestry.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "user",
                "uuid": "result-record",
                "timestamp": "2026-08-11T12:00:00Z",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "reused-id"}]},
            },
            {
                "type": "assistant",
                "uuid": "use-record",
                "parentUuid": "result-record",
                "timestamp": "2026-08-11T12:00:01Z",
                "message": {"content": [{"type": "tool_use", "id": "reused-id", "name": "Read"}]},
            },
        ],
    )

    parsed = parse_session(path)

    assert all(event.correlated_event_id is None for event in parsed.events)
    assert parsed.diagnostics.unmatched_tool_uses == 1
    assert parsed.diagnostics.orphan_tool_results == 1
    assert parsed.diagnostics.incompatible_tool_pairs == 1


def test_tool_pairs_without_uuid_lineage_are_not_guessed(tmp_path: Path) -> None:
    path = tmp_path / "missing-lineage.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "tool"}]},
            },
        ],
    )

    parsed = parse_session(path)

    assert parsed.events[0].correlated_event_id is None
    assert parsed.events[1].correlated_event_id is None
    assert parsed.diagnostics.unmatched_tool_uses == 1
    assert parsed.diagnostics.orphan_tool_results == 1
    assert parsed.diagnostics.incompatible_tool_pairs == 1


def test_naive_timestamp_and_invalid_error_state_are_diagnosed_without_crashing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-tool-metadata.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use",
                "timestamp": "2026-08-11T12:00:00",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result",
                "parentUuid": "use",
                "timestamp": "2026-08-11T12:00:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool",
                            "is_error": "false",
                        }
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    assert parsed.events[0].timestamp is None
    assert "invalid_timestamp" in parsed.events[0].warnings
    assert parsed.events[1].tool_result_is_error is None
    assert "invalid_tool_error_state" in parsed.events[1].warnings
    assert parsed.diagnostics.invalid_field_types == 2


def test_duplicate_ids_are_disambiguated_only_by_compatible_lineage(tmp_path: Path) -> None:
    path = tmp_path / "branch-duplicates.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "root",
                "message": {"content": "root"},
            },
            {
                "type": "assistant",
                "uuid": "use-a",
                "parentUuid": "root",
                "message": {"content": [{"type": "tool_use", "id": "duplicate", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result-a",
                "parentUuid": "use-a",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "duplicate"}]},
            },
            {
                "type": "assistant",
                "uuid": "use-b",
                "parentUuid": "root",
                "message": {"content": [{"type": "tool_use", "id": "duplicate", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result-b",
                "parentUuid": "use-b",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "duplicate"}]},
            },
        ],
    )

    parsed = parse_session(path)

    tool_events = [event for event in parsed.events if event.tool_use_id == "duplicate"]
    by_record = {event.record_uuid: event for event in tool_events}
    assert by_record["use-a"].correlated_event_id == by_record["result-a"].event_id
    assert by_record["use-b"].correlated_event_id == by_record["result-b"].event_id
    assert parsed.diagnostics.duplicate_tool_ids == 1
    assert parsed.diagnostics.ambiguous_tool_ids == 0


def test_ambiguous_and_invalid_ids_are_never_guessed(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "root",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "ambiguous", "name": "Read"},
                        {"type": "tool_use", "id": "ambiguous", "name": "Read"},
                        {"type": "tool_use", "id": 42, "name": "Read"},
                        {"type": "tool_use", "id": "unmatched", "name": "Read"},
                    ]
                },
            },
            {
                "type": "user",
                "uuid": "child",
                "parentUuid": "root",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "ambiguous"},
                        {"type": "tool_result", "tool_use_id": []},
                        {"type": "tool_result", "tool_use_id": "orphan"},
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    ambiguous = [event for event in parsed.events if event.tool_use_id == "ambiguous"]
    assert all(event.correlated_event_id is None for event in ambiguous)
    assert parsed.diagnostics.duplicate_tool_ids == 1
    assert parsed.diagnostics.ambiguous_tool_ids == 1
    assert parsed.diagnostics.invalid_tool_ids == 2
    assert parsed.diagnostics.unmatched_tool_uses == 3
    assert parsed.diagnostics.orphan_tool_results == 2


def test_negative_tool_interval_is_correlated_but_diagnosed(tmp_path: Path) -> None:
    path = tmp_path / "negative-time.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use",
                "timestamp": "2026-08-11T12:00:03Z",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result",
                "parentUuid": "use",
                "timestamp": "2026-08-11T12:00:02Z",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "tool"}]},
            },
        ],
    )

    parsed = parse_session(path)

    assert parsed.events[0].correlated_event_id == parsed.events[1].event_id
    assert parsed.diagnostics.negative_tool_intervals == 1
    assert "negative_tool_interval" in parsed.events[0].warnings


def test_tool_inputs_retain_only_bounded_artifact_selectors_and_mask_flags(
    tmp_path: Path,
) -> None:
    path = tmp_path / "allowlisted-inputs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "bash",
                            "name": "Bash",
                            "input": {
                                "command": (
                                    "deploy --token=plain-token --password hunter2 "
                                    "--secret 'quoted value' --key VALUE"
                                ),
                                "description": "must not be retained",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "read",
                            "name": "Read",
                            "input": {"file_path": "/src/a.py", "offset": 100},
                        },
                        {
                            "type": "tool_use",
                            "id": "edit",
                            "name": "Edit",
                            "input": {
                                "file_path": "/src/a.py",
                                "old_string": "bulk old content",
                                "new_string": "bulk new content",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "glob",
                            "name": "Glob",
                            "input": {"pattern": "**/*.py", "path": "/src"},
                        },
                        {
                            "type": "tool_use",
                            "id": "grep",
                            "name": "Grep",
                            "input": {"pattern": "needle", "path": "/src"},
                        },
                        {
                            "type": "tool_use",
                            "id": "write",
                            "name": "Write",
                            "input": {"file_path": "/" + "x" * 3_000, "content": "bulk"},
                        },
                    ]
                },
            }
        ],
    )

    parsed = parse_session(path)

    by_id = {event.tool_use_id: event for event in parsed.events}
    assert dict(by_id["bash"].tool_input) == {
        "command": (
            "deploy --token=<redacted-secret> --password <redacted-secret> "
            "--secret <redacted-secret> --key <redacted-secret>"
        )
    }
    assert dict(by_id["read"].tool_input) == {"file_path": "/src/a.py"}
    assert dict(by_id["edit"].tool_input) == {"file_path": "/src/a.py"}
    assert dict(by_id["glob"].tool_input) == {"pattern": "**/*.py", "path": "/src"}
    assert dict(by_id["grep"].tool_input) == {"pattern": "needle", "path": "/src"}
    assert len(dict(by_id["write"].tool_input)["file_path"]) == 2_000
    assert by_id["write"].tool_input_truncated_fields == ("file_path",)
    assert "tool_input_truncated" in by_id["write"].warnings
    assert parsed.diagnostics.tool_input_fields_truncated == 1


def test_list_form_tool_result_extracts_only_bounded_text(tmp_path: Path) -> None:
    path = tmp_path / "nested-result.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool",
                            "content": [
                                {"type": "text", "text": "Bearer secret-token\n"},
                                {"type": "image", "source": {"data": "opaque-bulk"}},
                                {"type": "text", "text": "x" * 3_000},
                            ],
                        }
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    result = parsed.events[1]
    assert result.text.startswith("Bearer <redacted-secret>\n")
    assert "opaque-bulk" not in result.text
    assert len(result.text) == 2_000
    assert result.text_truncated is True
    assert "text_truncated" in result.warnings


def test_huge_tool_result_bounds_text_passed_to_redaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "huge-result.jsonl"
    original_redact = parser_module.redact_secrets
    largest_redaction_input = 0

    def observed_redact(value: object) -> str:
        nonlocal largest_redaction_input
        if isinstance(value, str):
            largest_redaction_input = max(largest_redaction_input, len(value))
        return original_redact(value)

    monkeypatch.setattr(parser_module, "redact_secrets", observed_redact)
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use-record",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result-record",
                "parentUuid": "use-record",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool",
                            "content": ("postgresql://user:" + "s" * 100_000 + "@example.com/db"),
                        }
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    result = parsed.events[1]
    assert largest_redaction_input <= 8_192
    assert "s" * 50 not in result.text
    assert "<redacted-secret>" in result.text
    assert result.text_truncated is True


def test_tool_result_depth_omission_is_marked_truncated(tmp_path: Path) -> None:
    path = tmp_path / "deep-result.jsonl"
    nested: object = [{"type": "text", "text": "deep diagnostic"}]
    for _ in range(6):
        nested = [{"content": nested}]
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use-record",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result-record",
                "parentUuid": "use-record",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool",
                            "content": [
                                {"type": "text", "text": "visible prefix"},
                                {"content": nested},
                            ],
                        }
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    result = parsed.events[1]
    assert result.text == "visible prefix"
    assert result.text_truncated is True
    assert "text_truncated" in result.warnings
    assert parsed.diagnostics.text_excerpts_truncated == 1


def test_object_tool_result_content_is_omitted_with_incompleteness_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "object-result.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use-record",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result-record",
                "parentUuid": "use-record",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool",
                            "content": {"text": "unsupported object payload"},
                        }
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    result = parsed.events[1]
    assert result.text == ""
    assert result.text_truncated is True
    assert "unsupported_tool_result_content" in result.warnings
    assert "text_truncated" in result.warnings
    assert parsed.diagnostics.invalid_field_types == 1
    assert parsed.diagnostics.text_excerpts_truncated == 1


def test_invalid_and_unknown_list_result_blocks_mark_excerpt_incomplete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unsupported-result-blocks.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use-record",
                "message": {"content": [{"type": "tool_use", "id": "tool", "name": "Read"}]},
            },
            {
                "type": "user",
                "uuid": "result-record",
                "parentUuid": "use-record",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool",
                            "content": [
                                42,
                                {"type": "text", "text": 42},
                                {"type": "image", "source": "omitted"},
                                {"mystery": True},
                            ],
                        }
                    ]
                },
            },
        ],
    )

    parsed = parse_session(path)

    result = parsed.events[1]
    assert result.text == ""
    assert result.text_truncated is True
    assert "unsupported_tool_result_content" in result.warnings
    assert "text_truncated" in result.warnings
    assert parsed.diagnostics.invalid_field_types == 2
    assert parsed.diagnostics.unknown_content_blocks == 2
    assert parsed.diagnostics.text_excerpts_truncated == 1


def test_oversized_duplicate_id_group_is_bounded_and_left_uncorrelated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized-id-group.jsonl"
    uses = [{"type": "tool_use", "id": "repeated", "name": "Read"} for _ in range(65)]
    results = [{"type": "tool_result", "tool_use_id": "repeated"} for _ in range(65)]
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "use-record",
                "message": {"content": uses},
            },
            {
                "type": "user",
                "uuid": "result-record",
                "parentUuid": "use-record",
                "message": {"content": results},
            },
        ],
    )

    parsed = parse_session(path)

    assert not any(event.correlated_event_id for event in parsed.events)
    assert parsed.diagnostics.oversized_tool_id_groups == 1
    assert parsed.diagnostics.correlation_pairs_examined == 0
    assert parsed.diagnostics.unmatched_tool_uses == 65
    assert parsed.diagnostics.orphan_tool_results == 65
