"""Tests for bounded Claude session JSONL normalization."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

from scripts.core.session_audit.models import (
    AnalysisStatus,
    AuditLimits,
    ContentKind,
    EventActor,
    EvidenceSourceKind,
)
from scripts.core.session_audit.parser import _normalize_unicode_scalars, parse_session


def _write_jsonl(path: Path, records: list[object]) -> bytes:
    payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n" for record in records
    )
    path.write_bytes(payload)
    return payload


def test_parse_session_normalizes_string_and_list_text_with_streaming_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    payload = _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "parentUuid": None,
                "timestamp": "2026-08-11T12:00:00Z",
                "message": {"content": "Visible answer with sk-12345678"},
            },
            {
                "type": "user",
                "uuid": "user-1",
                "parentUuid": "assistant-1",
                "timestamp": "2026-08-11T12:00:03+00:00",
                "message": {"content": [{"type": "text", "text": "Please continue"}]},
            },
        ],
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.COMPLETE
    assert parsed.session.input_bytes == len(payload)
    assert parsed.session.sha256 == hashlib.sha256(payload).hexdigest()
    assert [event.event_id for event in parsed.events] == ["L1:B0", "L2:B0"]
    assert [event.chronological_index for event in parsed.events] == [0, 1]
    assert parsed.events[0].actor is EventActor.ASSISTANT
    assert parsed.events[0].kind is ContentKind.VISIBLE_TEXT
    assert parsed.events[0].source_kind is EvidenceSourceKind.ASSISTANT_TEXT
    assert parsed.events[0].text == "Visible answer with <redacted-secret>"
    assert parsed.events[0].timestamp == datetime(2026, 8, 11, 12, tzinfo=UTC)
    assert parsed.events[1].actor is EventActor.HUMAN
    assert parsed.events[1].source_kind is EvidenceSourceKind.USER_PROMPT
    assert parsed.events[1].source_line == 2
    assert parsed.events[1].content_block_index == 0
    assert parsed.diagnostics.nonblank_lines == 2
    assert parsed.diagnostics.valid_json_objects == 2
    assert parsed.diagnostics.recognized_message_records == 2
    assert parsed.diagnostics.normalized_events == 2


def test_semantic_text_replaces_unpaired_surrogates_and_preserves_valid_emoji(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unicode-scalars.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {
                    "content": [
                        {"type": "text", "text": "before\ud800after"},
                        {"type": "text", "text": "valid emoji: 😀"},
                    ]
                },
            }
        ],
    )

    parsed = parse_session(path)

    replaced, valid = parsed.events
    assert replaced.text == "before\ufffdafter"
    assert "unpaired_surrogate_replaced" in replaced.warnings
    assert valid.text == "valid emoji: 😀"
    assert "unpaired_surrogate_replaced" not in valid.warnings
    assert parsed.diagnostics.unicode_replacement_events == 1


def test_valid_unicode_normalization_fast_path_has_constant_extra_memory() -> None:
    text = "x" * 1_000_000

    tracemalloc.start()
    try:
        normalized, replaced = _normalize_unicode_scalars(text)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert normalized == text
    assert replaced is False
    assert peak_bytes < 100_000


def test_trailing_unpaired_surrogate_normalization_has_bounded_extra_memory() -> None:
    text = "x" * 1_000_000 + "\ud800"

    tracemalloc.start()
    try:
        normalized, replaced = _normalize_unicode_scalars(text)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert normalized == "x" * 1_000_000 + "\ufffd"
    assert replaced is True
    assert peak_bytes < 4_000_000


def test_malformed_middle_line_and_unknown_record_are_tolerated_and_counted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.jsonl"
    records = [
        {
            "type": "assistant",
            "uuid": "a",
            "message": {"content": "answer"},
        },
        {
            "type": "attachment",
            "payload": "unknown bulk data must not become an event",
        },
        {
            "type": "user",
            "uuid": "u",
            "parentUuid": "a",
            "message": {"content": "continue"},
        },
    ]
    path.write_bytes(
        json.dumps(records[0]).encode()
        + b"\n"
        + b"{not-json}\n"
        + json.dumps(records[1]).encode()
        + b"\n"
        + json.dumps(records[2]).encode()
        + b"\n"
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.COMPLETE
    assert len(parsed.events) == 2
    assert parsed.diagnostics.nonblank_lines == 4
    assert parsed.diagnostics.malformed_lines == 1
    assert parsed.diagnostics.trailing_partial_lines == 0
    assert parsed.diagnostics.unknown_record_types == 1
    assert parsed.diagnostics.message_like_records == 2
    assert parsed.diagnostics.failed_message_records == 0


def test_single_malformed_final_nonblank_line_is_tolerated_as_trailing_partial(
    tmp_path: Path,
) -> None:
    path = tmp_path / "growing.jsonl"
    complete = json.dumps({"type": "assistant", "message": {"content": "complete"}}).encode()
    payload = complete + b"\n" + b'{"type":"assistant"'
    path.write_bytes(payload)

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.COMPLETE
    assert parsed.session.input_bytes == len(payload)
    assert parsed.session.sha256 == hashlib.sha256(payload).hexdigest()
    assert parsed.diagnostics.malformed_lines == 1
    assert parsed.diagnostics.trailing_partial_lines == 1


def test_only_trailing_partial_refuses_but_blank_input_remains_complete(
    tmp_path: Path,
) -> None:
    partial_path = tmp_path / "partial-only.jsonl"
    partial_path.write_bytes(b'{"type":"assistant"')
    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_bytes(b"\n  \n")

    partial = parse_session(partial_path)
    blank = parse_session(blank_path)

    assert partial.status is AnalysisStatus.REFUSED
    assert partial.events == ()
    assert partial.diagnostics.nonblank_lines == 1
    assert partial.diagnostics.malformed_lines == 1
    assert partial.diagnostics.trailing_partial_lines == 1
    assert partial.diagnostics.refusal_reasons == ("no_complete_records",)
    assert blank.status is AnalysisStatus.COMPLETE
    assert blank.events == ()
    assert blank.diagnostics.nonblank_lines == 0
    assert blank.diagnostics.refusal_reasons == ()


def test_excess_malformed_middle_lines_refuse_without_returning_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corrupt.jsonl"
    path.write_bytes(
        b"{bad-one}\n{bad-two}\n{bad-three}\n"
        + json.dumps({"type": "assistant", "message": {"content": "valid suffix"}}).encode()
        + b"\n"
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.malformed_lines == 3
    assert parsed.diagnostics.trailing_partial_lines == 0
    assert parsed.diagnostics.refusal_reasons == ("malformed_line_threshold",)


def test_one_failed_message_shape_is_tolerated_at_the_exact_floor(tmp_path: Path) -> None:
    path = tmp_path / "one-schema-failure.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "assistant", "message": {"content": "valid"}},
            {"type": "user", "message": "invalid"},
        ],
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.COMPLETE
    assert len(parsed.events) == 1
    assert parsed.diagnostics.message_like_records == 2
    assert parsed.diagnostics.recognized_message_records == 1
    assert parsed.diagnostics.failed_message_records == 1


def test_message_normalization_failures_over_five_percent_refuse(tmp_path: Path) -> None:
    path = tmp_path / "schema-drift.jsonl"
    records: list[object] = [
        {"type": "assistant", "message": {"content": f"valid {index}"}} for index in range(18)
    ]
    records.extend(
        [
            {"type": "assistant", "message": "invalid"},
            {"type": "user", "message": {"content": {"unexpected": "object"}}},
        ]
    )
    _write_jsonl(path, records)

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.message_like_records == 20
    assert parsed.diagnostics.recognized_message_records == 18
    assert parsed.diagnostics.failed_message_records == 2
    assert parsed.diagnostics.refusal_reasons == ("message_normalization_threshold",)


def test_known_blocks_with_unusable_payloads_count_as_message_normalization_failures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "block-schema-drift.jsonl"
    records: list[object] = [
        {"type": "assistant", "message": {"content": f"valid {index}"}} for index in range(18)
    ]
    records.extend(
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": 42}]},
            },
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": None}]},
            },
        ]
    )
    _write_jsonl(path, records)

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.message_like_records == 20
    assert parsed.diagnostics.failed_message_records == 2
    assert "message_normalization_threshold" in parsed.diagnostics.refusal_reasons


def test_valid_json_without_normalizable_human_or_assistant_events_refuses(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unsupported.jsonl"
    _write_jsonl(path, [{"type": "attachment", "payload": "opaque"}])

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.valid_json_objects == 1
    assert parsed.diagnostics.unknown_record_types == 1
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)


def test_missing_and_invalid_timestamps_remain_null_and_are_not_invented(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nullable-time.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "assistant", "message": {"content": "missing time"}},
            {
                "type": "user",
                "uuid": 42,
                "parentUuid": [],
                "isSidechain": "yes",
                "timestamp": 123,
                "message": {"content": "invalid metadata types"},
            },
        ],
    )

    parsed = parse_session(path)

    assert [event.timestamp for event in parsed.events] == [None, None]
    assert parsed.events[1].record_uuid is None
    assert parsed.events[1].parent_uuid is None
    assert parsed.events[1].is_sidechain is False
    assert {
        "invalid_record_uuid",
        "invalid_parent_uuid",
        "invalid_sidechain",
        "invalid_timestamp",
    }.issubset(parsed.events[1].warnings)
    assert parsed.diagnostics.invalid_field_types == 4


def test_internal_user_metadata_is_not_labeled_as_a_human_prompt(tmp_path: Path) -> None:
    path = tmp_path / "internal-user-metadata.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "uuid": "root",
                "message": {"content": "visible answer"},
            },
            {
                "type": "user",
                "uuid": "meta",
                "parentUuid": "root",
                "isMeta": True,
                "message": {"content": [{"type": "text", "text": "internal note"}]},
            },
            {
                "type": "user",
                "uuid": "summary",
                "parentUuid": "meta",
                "isVisibleInTranscriptOnly": True,
                "isCompactSummary": True,
                "message": {"content": "compacted context"},
            },
        ],
    )

    parsed = parse_session(path)

    assert [event.actor for event in parsed.events] == [
        EventActor.ASSISTANT,
        EventActor.SYSTEM,
        EventActor.SYSTEM,
    ]
    assert [event.kind for event in parsed.events[1:]] == [
        ContentKind.METADATA,
        ContentKind.METADATA,
    ]
    assert [event.source_kind for event in parsed.events[1:]] == [
        EvidenceSourceKind.DERIVED,
        EvidenceSourceKind.DERIVED,
    ]
    assert parsed.diagnostics.internal_metadata_records == 2
    assert not any(event.actor is EventActor.HUMAN for event in parsed.events)


def test_initial_input_byte_limit_refuses_without_reading_or_digesting(tmp_path: Path) -> None:
    path = tmp_path / "oversized.jsonl"
    path.write_bytes(b"x" * 65)

    parsed = parse_session(path, limits=AuditLimits(max_input_bytes=64))

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.session.input_bytes == 65
    assert parsed.session.sha256 == ""
    assert parsed.diagnostics.digest_complete is False
    assert parsed.diagnostics.refusal_reasons == ("input_byte_limit",)


def test_input_line_byte_limit_refuses_before_parsing_and_returns_no_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized-line.jsonl"
    valid_prefix = json.dumps(
        {"type": "assistant", "message": {"content": "visible prefix"}}
    ).encode()
    oversized_record = json.dumps({"type": "assistant", "message": {"content": "x" * 256}}).encode()
    path.write_bytes(valid_prefix + b"\n" + oversized_record + b"\n")

    parsed = parse_session(path, limits=AuditLimits(max_input_line_bytes=128))

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.session.sha256 == ""
    assert parsed.diagnostics.digest_complete is False
    assert parsed.diagnostics.valid_json_objects == 1
    assert parsed.diagnostics.normalized_events == 1
    assert parsed.diagnostics.refusal_reasons == ("input_line_byte_limit",)


def test_input_record_limit_refuses_before_parsing_the_overflow_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "too-many-records.jsonl"
    path.write_bytes(b"0\n1\n2\n3\n")

    parsed = parse_session(path, limits=AuditLimits(max_input_records=3))

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.session.sha256 == ""
    assert parsed.diagnostics.digest_complete is False
    assert parsed.diagnostics.nonblank_lines == 4
    assert parsed.diagnostics.non_object_json_values == 3
    assert parsed.diagnostics.refusal_reasons == ("input_record_limit",)


def test_uuidless_unknown_records_do_not_accumulate_lineage_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "uuidless-unknown.jsonl"
    path.write_bytes(b'{"type":"attachment"}\n' * 20_000)

    tracemalloc.start()
    try:
        parsed = parse_session(path)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.diagnostics.unknown_record_types == 20_000
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)
    assert peak_bytes < 1_000_000


def test_streaming_byte_limit_catches_file_growth_after_stat(tmp_path: Path) -> None:
    path = tmp_path / "growing.pipe"
    os.mkfifo(path)

    def write_growing_input() -> None:
        with path.open("wb", buffering=0) as handle:
            handle.write(b"x" * 128)

    writer = threading.Thread(target=write_growing_input, daemon=True)
    writer.start()
    parsed = parse_session(path, limits=AuditLimits(max_input_bytes=64))
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.session.input_bytes == 65
    assert parsed.session.sha256 == ""
    assert parsed.diagnostics.digest_complete is False
    assert parsed.diagnostics.refusal_reasons == ("input_byte_limit",)


def test_normalized_event_limit_refuses_without_returning_retained_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "too-many-events.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "assistant", "message": {"content": "one"}},
            {"type": "assistant", "message": {"content": "two"}},
            {"type": "assistant", "message": {"content": "three"}},
        ],
    )

    parsed = parse_session(path, limits=AuditLimits(max_normalized_events=2))

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.normalized_events == 3
    assert parsed.diagnostics.digest_complete is False
    assert parsed.diagnostics.refusal_reasons == ("normalized_event_limit",)


def test_excessively_nested_json_refuses_instead_of_raising_recursion_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested-json-bomb.jsonl"
    path.write_bytes(b"[" * 10_000 + b"0" + b"]" * 10_000 + b"\n")

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.digest_complete is False
    assert parsed.diagnostics.refusal_reasons == ("json_complexity_limit",)


def test_oversized_json_integer_refuses_instead_of_raising_value_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large-integer.jsonl"
    path.write_bytes(b'{"type":"assistant","message":{"content":' + b"9" * 5_000 + b"}}\n")

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.digest_complete is False
    assert parsed.diagnostics.refusal_reasons == ("json_complexity_limit",)


def test_semantic_text_is_bounded_and_truncation_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "long-text.jsonl"
    _write_jsonl(
        path,
        [{"type": "assistant", "message": {"content": "x" * 5_000}}],
    )

    parsed = parse_session(path)

    assert len(parsed.events[0].text) == 4_000
    assert parsed.events[0].text_truncated is True
    assert "text_truncated" in parsed.events[0].warnings
    assert parsed.diagnostics.text_excerpts_truncated == 1


def test_known_block_with_invalid_payload_type_does_not_count_as_normalizable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-text-payload.jsonl"
    _write_jsonl(
        path,
        [{"type": "assistant", "message": {"content": [{"type": "text", "text": 42}]}}],
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.recognized_message_records == 1
    assert parsed.diagnostics.invalid_field_types == 1
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)


def test_empty_assistant_content_is_diagnosed_without_schema_refusal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "known-empty-messages.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "assistant", "message": {"content": "visible answer"}},
            {"type": "assistant", "message": {"content": []}},
            {"type": "assistant", "message": {"content": []}},
        ],
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.COMPLETE
    assert len(parsed.events) == 1
    assert parsed.diagnostics.empty_message_records == 2
    assert parsed.diagnostics.failed_message_records == 0


def test_empty_string_and_empty_text_block_only_transcript_is_unsupported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty-text-only.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "assistant", "message": {"content": ""}},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": ""}]},
            },
        ],
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.empty_message_records == 2
    assert parsed.diagnostics.failed_message_records == 0
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)


def test_empty_string_and_empty_text_blocks_are_omitted_beside_valid_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed-empty-text.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "assistant", "message": {"content": ""}},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": ""}]},
            },
            {"type": "assistant", "message": {"content": "visible answer"}},
        ],
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.COMPLETE
    assert [event.text for event in parsed.events] == ["visible answer"]
    assert parsed.diagnostics.empty_message_records == 2
    assert parsed.diagnostics.failed_message_records == 0


def test_non_object_valid_json_is_counted_and_refused_as_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "array.jsonl"
    path.write_text('[{"type": "assistant"}]\n')

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.valid_json_objects == 0
    assert parsed.diagnostics.non_object_json_values == 1
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)


def test_unhashable_record_type_is_diagnosed_instead_of_crashing(tmp_path: Path) -> None:
    path = tmp_path / "invalid-record-type.jsonl"
    _write_jsonl(
        path,
        [{"type": [], "message": {"content": "not a declared message"}}],
    )

    parsed = parse_session(path)

    assert parsed.status is AnalysisStatus.REFUSED
    assert parsed.events == ()
    assert parsed.diagnostics.unknown_record_types == 1
    assert parsed.diagnostics.invalid_field_types == 1
    assert parsed.diagnostics.refusal_reasons == ("unsupported_schema",)
