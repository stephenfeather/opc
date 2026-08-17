"""Thinking-provenance tests for the session-audit parser."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.core.session_audit.models import (
    AnalysisStatus,
    ContentKind,
    EventActor,
    EvidenceSourceKind,
)
from scripts.core.session_audit.parser import parse_session


def _thinking_session(path: Path) -> None:
    record = {
        "type": "assistant",
        "uuid": "assistant-1",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "I used sk-12345678 by mistake"},
                {"type": "text", "text": "I will correct it."},
            ]
        },
    }
    path.write_text(json.dumps(record) + "\n")


def test_thinking_is_included_with_explicit_provenance_by_default(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _thinking_session(path)

    parsed = parse_session(path)

    thinking = parsed.events[0]
    assert thinking.event_id == "L1:B0"
    assert thinking.actor is EventActor.ASSISTANT
    assert thinking.kind is ContentKind.THINKING
    assert thinking.source_kind is EvidenceSourceKind.ASSISTANT_THINKING
    assert thinking.text == "I used <redacted-secret> by mistake"


def test_thinking_and_tool_result_text_replace_unpaired_surrogates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unicode-thinking-result.jsonl"
    records = [
        {
            "type": "assistant",
            "uuid": "use",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "reason\ud800ing"},
                    {"type": "tool_use", "id": "read-1", "name": "Read"},
                ]
            },
        },
        {
            "type": "user",
            "uuid": "result",
            "parentUuid": "use",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read-1",
                        "content": "out\udcffput",
                    }
                ]
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    parsed = parse_session(path)

    thinking, _, result = parsed.events
    assert thinking.text == "reason\ufffding"
    assert result.text == "out\ufffdput"
    assert "unpaired_surrogate_replaced" in thinking.warnings
    assert "unpaired_surrogate_replaced" in result.warnings
    assert parsed.diagnostics.unicode_replacement_events == 2


def test_thinking_can_be_excluded_without_renumbering_source_blocks(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _thinking_session(path)

    parsed = parse_session(path, include_thinking=False)

    assert len(parsed.events) == 1
    assert parsed.events[0].event_id == "L1:B1"
    assert parsed.events[0].kind is ContentKind.VISIBLE_TEXT


def test_thinking_only_session_is_unsupported_when_thinking_is_excluded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "thinking-only.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "thinking", "thinking": "private reasoning"}]},
            }
        )
        + "\n"
    )

    parsed = parse_session(path, include_thinking=False)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)


def test_empty_thinking_only_session_is_known_empty_and_unsupported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty-thinking-only.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "thinking", "thinking": ""}]},
            }
        )
        + "\n"
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.empty_message_records == 1
    assert parsed.diagnostics.failed_message_records == 0
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)


def test_empty_thinking_record_is_omitted_beside_valid_content(tmp_path: Path) -> None:
    path = tmp_path / "mixed-empty-thinking.jsonl"
    records = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": ""}]},
        },
        {"type": "assistant", "message": {"content": "visible answer"}},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.COMPLETE
    assert [event.text for event in parsed.events] == ["visible answer"]
    assert parsed.diagnostics.empty_message_records == 1
    assert parsed.diagnostics.failed_message_records == 0
