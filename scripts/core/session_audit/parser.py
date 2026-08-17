"""Bounded streaming normalization for Claude Code session JSONL files."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TypeGuard

from scripts.core.log_safety import redact_secrets
from scripts.core.session_audit.models import (
    AnalysisStatus,
    AuditLimits,
    ContentKind,
    EventActor,
    EvidenceSourceKind,
    NormalizedEvent,
    ParsedSession,
    ParserDiagnostics,
    SessionMetadata,
)

MAX_LINEAGE_SUMMARY_IDS = 8
MAX_BRANCH_SUMMARY_IDS = 4
MAX_SEMANTIC_TEXT_CHARS = 4_000
MAX_TOOL_RESULT_TEXT_CHARS = 2_000
MAX_TOOL_INPUT_VALUE_CHARS = 2_000
MAX_TOOL_CORRELATION_GROUP_ENDPOINTS = 128
MAX_REDACTION_SCAN_SLACK_CHARS = 4_096

TOOL_INPUT_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "Bash": ("command",),
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "Write": ("file_path",),
    "MultiEdit": ("file_path",),
    "Glob": ("pattern", "path"),
    "Grep": ("pattern", "path"),
    "NotebookEdit": ("notebook_path",),
}

_CREDENTIAL_FLAG_RE = re.compile(
    r"(--(?:password|token|secret|(?:api-)?key)(?:=|\s+))" r"(?:\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
_SURROGATE_RE = re.compile(r"[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]")
_PARTIAL_CONNECTION_SECRET_RE = re.compile(r"(://[^:/@\s]+:)[^@\s]*\Z")


@dataclass(frozen=True)
class _Block:
    index: int
    actor: EventActor
    kind: ContentKind
    source_kind: EvidenceSourceKind
    text: str = ""
    text_truncated: bool = False
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: tuple[tuple[str, str], ...] = ()
    tool_input_truncated_fields: tuple[str, ...] = ()
    tool_result_is_error: bool | None = None
    unicode_replaced: bool = False
    invalid_field_types: int = 0
    unknown_content_blocks: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RecordMeta:
    line: int
    record_type: str
    uuid: str | None
    parent_uuid: str | None
    is_sidechain: bool
    source_tool_assistant_uuid: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CorrelationStats:
    unmatched_tool_uses: int = 0
    orphan_tool_results: int = 0
    duplicate_tool_ids: int = 0
    ambiguous_tool_ids: int = 0
    invalid_tool_ids: int = 0
    negative_tool_intervals: int = 0
    incompatible_tool_pairs: int = 0
    oversized_tool_id_groups: int = 0
    correlation_pairs_examined: int = 0


@dataclass
class _LineageInfo:
    root_uuid: str
    depth: int
    lineage: tuple[str, ...]
    lineage_truncated: bool
    branch_path: tuple[str, ...]
    branch_id: str
    branch_depth: int
    branch_path_truncated: bool
    ancestry_start: int = 0
    ancestry_end: int = 0


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _is_valid_identifier(value: object) -> TypeGuard[str]:
    if not isinstance(value, str) or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _redact_command(text: str) -> str:
    return _CREDENTIAL_FLAG_RE.sub(r"\1<redacted-secret>", redact_secrets(text))


def _normalize_unicode_scalars(text: str) -> tuple[str, bool]:
    if _SURROGATE_RE.search(text) is None:
        return text, False
    replaced = False

    def replace_surrogate(match: re.Match[str]) -> str:
        nonlocal replaced
        value = match.group(0)
        if len(value) == 2:
            high, low = map(ord, value)
            scalar = 0x10000 + (high - 0xD800) * 0x400 + (low - 0xDC00)
            return chr(scalar)
        replaced = True
        return "\ufffd"

    return _SURROGATE_RE.sub(replace_surrogate, text), replaced


def _bounded_text(text: str, maximum: int) -> tuple[str, bool, bool]:
    normalized, unicode_replaced = _normalize_unicode_scalars(text)
    redacted = redact_secrets(normalized)
    return redacted[:maximum], len(redacted) > maximum, unicode_replaced


def _bounded_tool_input(
    tool_name: str | None, value: object
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], bool]:
    if tool_name is None or not isinstance(value, dict):
        return (), (), False
    retained: list[tuple[str, str]] = []
    truncated: list[str] = []
    unicode_replaced = False
    for field in TOOL_INPUT_ALLOWLIST.get(tool_name, ()):
        raw_value = value.get(field)
        if not isinstance(raw_value, str):
            continue
        normalized, field_unicode_replaced = _normalize_unicode_scalars(raw_value)
        unicode_replaced = unicode_replaced or field_unicode_replaced
        redacted = _redact_command(normalized) if field == "command" else redact_secrets(normalized)
        if len(redacted) > MAX_TOOL_INPUT_VALUE_CHARS:
            truncated.append(field)
        retained.append((field, redacted[:MAX_TOOL_INPUT_VALUE_CHARS]))
    return tuple(retained), tuple(truncated), unicode_replaced


def _tool_result_text(value: object) -> tuple[str, bool, bool, int, int, bool]:
    parts: list[str] = []
    retained_length = 0
    ends_with_newline = False
    truncated = False
    unicode_replaced = False
    invalid_fields = 0
    unknown_blocks = 0
    unsupported_content = False

    def append_text(raw_text: str) -> None:
        nonlocal ends_with_newline, retained_length, truncated, unicode_replaced
        available = MAX_TOOL_RESULT_TEXT_CHARS - retained_length
        scan_limit = available + MAX_REDACTION_SCAN_SLACK_CHARS
        bounded_raw_text = raw_text[:scan_limit]
        raw_text_truncated = len(raw_text) > scan_limit
        normalized, text_unicode_replaced = _normalize_unicode_scalars(bounded_raw_text)
        unicode_replaced = unicode_replaced or text_unicode_replaced
        if raw_text_truncated:
            normalized = _PARTIAL_CONNECTION_SECRET_RE.sub(r"\1<redacted-secret>", normalized)
        text = redact_secrets(normalized)
        if not text:
            truncated = truncated or raw_text_truncated
            return
        separator = "" if not parts or ends_with_newline else "\n"
        if available <= 0:
            truncated = True
            return
        addition = f"{separator}{text}"
        if len(addition) > available:
            addition = addition[:available]
            truncated = True
        parts.append(addition)
        retained_length += len(addition)
        ends_with_newline = addition.endswith("\n")
        truncated = truncated or raw_text_truncated

    def visit(node: object, *, depth: int) -> None:
        nonlocal invalid_fields, truncated, unknown_blocks, unsupported_content
        if retained_length >= MAX_TOOL_RESULT_TEXT_CHARS:
            truncated = True
            return
        if isinstance(node, str):
            append_text(node)
            return
        if not isinstance(node, list):
            return
        for item in node:
            if retained_length >= MAX_TOOL_RESULT_TEXT_CHARS:
                truncated = True
                break
            if not isinstance(item, dict):
                invalid_fields += 1
                unsupported_content = True
                truncated = True
                continue
            block_type = item.get("type")
            if block_type == "text":
                if isinstance(item.get("text"), str):
                    append_text(item["text"])
                else:
                    invalid_fields += 1
                    unsupported_content = True
                    truncated = True
            elif block_type is None and "content" in item:
                nested_content = item["content"]
                if not isinstance(nested_content, (str, list)):
                    invalid_fields += 1
                    unsupported_content = True
                    truncated = True
                    continue
                if depth >= 4:
                    truncated = True
                else:
                    visit(nested_content, depth=depth + 1)
            else:
                unknown_blocks += 1
                unsupported_content = True
                truncated = True

    visit(value, depth=0)
    return (
        "".join(parts),
        truncated,
        unicode_replaced,
        invalid_fields,
        unknown_blocks,
        unsupported_content,
    )


def _has_normalizable_actor_content(
    content: object,
    record_type: str,
    *,
    include_thinking: bool,
) -> bool:
    if isinstance(content, str):
        return bool(content)
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        block_type = item.get("type")
        if block_type == "text" and isinstance(item.get("text"), str) and bool(item["text"]):
            return True
        if record_type == "assistant":
            if (
                include_thinking
                and block_type == "thinking"
                and isinstance(item.get("thinking"), str)
                and bool(item["thinking"])
            ):
                return True
            if block_type == "tool_use":
                return True
    return False


def _has_normalizable_message_content(content: object, record_type: str) -> bool:
    """Return whether content contains a block the normalizer can emit."""
    if isinstance(content, str):
        return bool(content)
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        block_type = item.get("type")
        if block_type == "text" and isinstance(item.get("text"), str) and bool(item["text"]):
            return True
        if record_type == "assistant":
            if (
                block_type == "thinking"
                and isinstance(item.get("thinking"), str)
                and bool(item["thinking"])
            ):
                return True
            if block_type == "tool_use":
                return True
        if record_type == "user" and block_type == "tool_result":
            return True
    return False


def _is_known_empty_content(content: object, record_type: str) -> bool:
    if content == "":
        return True
    if not isinstance(content, list):
        return False
    return not content or all(
        isinstance(item, dict)
        and (
            (item.get("type") == "text" and item.get("text") == "")
            or (
                record_type == "assistant"
                and item.get("type") == "thinking"
                and item.get("thinking") == ""
            )
        )
        for item in content
    )


def _content_diagnostics(content: object, record_type: str) -> tuple[int, int]:
    if isinstance(content, str):
        return 0, 0
    if not isinstance(content, list):
        return 1, 0
    invalid_fields = 0
    unknown_blocks = 0
    for item in content:
        if not isinstance(item, dict):
            invalid_fields += 1
            continue
        block_type = item.get("type")
        if block_type == "text":
            invalid_fields += not isinstance(item.get("text"), str)
        elif block_type == "thinking" and record_type == "assistant":
            invalid_fields += not isinstance(item.get("thinking"), str)
        elif block_type == "tool_use" and record_type == "assistant":
            invalid_fields += not isinstance(item.get("name"), str)
            invalid_fields += not isinstance(item.get("input", {}), dict)
        elif block_type == "tool_result" and record_type == "user":
            is_error = item.get("is_error")
            invalid_fields += is_error is not None and not isinstance(is_error, bool)
            result_content = item.get("content")
            invalid_fields += (
                "content" in item
                and result_content is not None
                and not isinstance(result_content, (str, list))
            )
        else:
            unknown_blocks += 1
    return invalid_fields, unknown_blocks


def _record_metadata(
    value: dict[str, object],
    *,
    line_number: int,
    record_type: str,
) -> tuple[_RecordMeta, datetime | None, int]:
    record_uuid = value.get("uuid")
    parent_uuid = value.get("parentUuid")
    source_tool_uuid = value.get("sourceToolAssistantUUID")
    warnings: list[str] = []
    if "uuid" in value and not _is_valid_identifier(record_uuid):
        warnings.append("invalid_record_uuid")
    if "parentUuid" in value and parent_uuid is not None and not _is_valid_identifier(parent_uuid):
        warnings.append("invalid_parent_uuid")
    if "isSidechain" in value and not isinstance(value.get("isSidechain"), bool):
        warnings.append("invalid_sidechain")
    if "sourceToolAssistantUUID" in value and not _is_valid_identifier(source_tool_uuid):
        warnings.append("invalid_source_tool_assistant_uuid")
    timestamp = _parse_timestamp(value.get("timestamp"))
    if "timestamp" in value and value.get("timestamp") is not None and timestamp is None:
        warnings.append("invalid_timestamp")
    return (
        _RecordMeta(
            line=line_number,
            record_type=record_type,
            uuid=record_uuid if _is_valid_identifier(record_uuid) else None,
            parent_uuid=parent_uuid if _is_valid_identifier(parent_uuid) else None,
            is_sidechain=value.get("isSidechain") is True,
            source_tool_assistant_uuid=(
                source_tool_uuid if _is_valid_identifier(source_tool_uuid) else None
            ),
            warnings=tuple(warnings),
        ),
        timestamp,
        len(warnings),
    )


def _is_internal_metadata_record(value: dict[str, object], record_type: str) -> bool:
    return record_type == "user" and (
        value.get("isMeta") is True
        or (
            value.get("isVisibleInTranscriptOnly") is True and value.get("isCompactSummary") is True
        )
    )


def _message_blocks(
    content: object,
    *,
    record_type: str,
    include_thinking: bool,
    internal_metadata: bool,
) -> Iterator[_Block]:
    if internal_metadata:
        text_source = EvidenceSourceKind.DERIVED
        text_actor = EventActor.SYSTEM
        text_kind = ContentKind.METADATA
    else:
        text_source = (
            EvidenceSourceKind.ASSISTANT_TEXT
            if record_type == "assistant"
            else EvidenceSourceKind.USER_PROMPT
        )
        text_actor = EventActor.ASSISTANT if record_type == "assistant" else EventActor.HUMAN
        text_kind = ContentKind.VISIBLE_TEXT
    if isinstance(content, str):
        if not content:
            return
        yield _Block(
            index=0,
            actor=text_actor,
            kind=text_kind,
            source_kind=text_source,
            text=content,
        )
        return
    if not isinstance(content, list):
        return
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        block_type = item.get("type")
        if block_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                yield _Block(
                    index=index,
                    actor=text_actor,
                    kind=text_kind,
                    source_kind=text_source,
                    text=text,
                )
        elif block_type == "thinking" and record_type == "assistant" and include_thinking:
            thinking = item.get("thinking")
            if isinstance(thinking, str) and thinking:
                yield _Block(
                    index=index,
                    actor=EventActor.ASSISTANT,
                    kind=ContentKind.THINKING,
                    source_kind=EvidenceSourceKind.ASSISTANT_THINKING,
                    text=thinking,
                )
        elif block_type == "tool_use" and record_type == "assistant":
            tool_id = item.get("id")
            tool_name = item.get("name")
            tool_input = item.get("input")
            normalized_tool_name = tool_name if _is_valid_identifier(tool_name) else None
            retained_input, truncated_fields, unicode_replaced = _bounded_tool_input(
                normalized_tool_name,
                tool_input,
            )
            warnings: list[str] = []
            if not _is_valid_identifier(tool_id):
                warnings.append("invalid_tool_use_id")
            if truncated_fields:
                warnings.append("tool_input_truncated")
            yield _Block(
                index=index,
                actor=EventActor.ASSISTANT,
                kind=ContentKind.TOOL_USE,
                source_kind=EvidenceSourceKind.TOOL_USE,
                tool_use_id=tool_id if _is_valid_identifier(tool_id) else None,
                tool_name=normalized_tool_name,
                tool_input=retained_input,
                tool_input_truncated_fields=truncated_fields,
                unicode_replaced=unicode_replaced,
                warnings=tuple(warnings),
            )
        elif block_type == "tool_result" and record_type == "user":
            tool_id = item.get("tool_use_id")
            is_error = item.get("is_error")
            raw_result_content = item.get("content")
            unsupported_result_content = (
                "content" in item
                and raw_result_content is not None
                and not isinstance(raw_result_content, (str, list))
            )
            (
                result_content,
                result_truncated,
                unicode_replaced,
                result_invalid_fields,
                result_unknown_blocks,
                nested_unsupported_content,
            ) = _tool_result_text(raw_result_content)
            result_truncated = result_truncated or unsupported_result_content
            result_warnings: list[str] = []
            if not _is_valid_identifier(tool_id):
                result_warnings.append("invalid_tool_result_id")
            if is_error is not None and not isinstance(is_error, bool):
                result_warnings.append("invalid_tool_error_state")
            if unsupported_result_content or nested_unsupported_content:
                result_warnings.append("unsupported_tool_result_content")
            yield _Block(
                index=index,
                actor=EventActor.TOOL,
                kind=ContentKind.TOOL_RESULT,
                source_kind=EvidenceSourceKind.TOOL_RESULT,
                text=result_content,
                text_truncated=result_truncated,
                tool_use_id=tool_id if _is_valid_identifier(tool_id) else None,
                tool_result_is_error=is_error if isinstance(is_error, bool) else None,
                unicode_replaced=unicode_replaced,
                invalid_field_types=result_invalid_fields,
                unknown_content_blocks=result_unknown_blocks,
                warnings=tuple(result_warnings),
            )


def _event_is_ancestor_or_same(
    ancestor: NormalizedEvent,
    descendant: NormalizedEvent,
) -> bool:
    if not ancestor.lineage_valid or not descendant.lineage_valid:
        return False
    if ancestor.record_uuid is None or descendant.record_uuid is None:
        return False
    if (
        ancestor.lineage_root_uuid != descendant.lineage_root_uuid
        or ancestor.ancestry_start is None
        or ancestor.ancestry_end is None
        or descendant.ancestry_start is None
        or descendant.ancestry_end is None
    ):
        return False
    return (
        ancestor.ancestry_start <= descendant.ancestry_start
        and descendant.ancestry_end <= ancestor.ancestry_end
    )


def events_share_lineage(first: NormalizedEvent, second: NormalizedEvent) -> bool:
    """Return whether two events have a validated ancestor/descendant relationship."""
    return _event_is_ancestor_or_same(first, second) or _event_is_ancestor_or_same(second, first)


def _correlate_tool_events(
    events: list[NormalizedEvent],
) -> tuple[list[NormalizedEvent], _CorrelationStats]:
    correlated = list(events)
    uses_by_id: dict[str, list[int]] = {}
    results_by_id: dict[str, list[int]] = {}
    for index, event in enumerate(correlated):
        if event.kind is ContentKind.TOOL_USE and event.tool_use_id:
            uses_by_id.setdefault(event.tool_use_id, []).append(index)
        elif event.kind is ContentKind.TOOL_RESULT and event.tool_use_id:
            results_by_id.setdefault(event.tool_use_id, []).append(index)
    duplicate_ids = {
        tool_id
        for tool_id in uses_by_id.keys() | results_by_id.keys()
        if len(uses_by_id.get(tool_id, ())) > 1 or len(results_by_id.get(tool_id, ())) > 1
    }
    ambiguous_ids: set[str] = set()
    oversized_ids: set[str] = set()
    negative_intervals = 0
    incompatible_pairs = 0
    pairs_examined = 0
    for tool_id in uses_by_id.keys() & results_by_id.keys():
        use_indexes = uses_by_id[tool_id]
        result_indexes = results_by_id[tool_id]
        if len(use_indexes) + len(result_indexes) > MAX_TOOL_CORRELATION_GROUP_ENDPOINTS:
            oversized_ids.add(tool_id)
            ambiguous_ids.add(tool_id)
            continue
        pairs_examined += len(use_indexes) * len(result_indexes)
        compatible_pairs = {
            (use_index, result_index)
            for use_index in use_indexes
            for result_index in result_indexes
            if _event_is_ancestor_or_same(correlated[use_index], correlated[result_index])
        }
        results_for_use = {
            use_index: [
                result_index
                for result_index in result_indexes
                if (use_index, result_index) in compatible_pairs
            ]
            for use_index in use_indexes
        }
        uses_for_result = {
            result_index: [
                use_index
                for use_index in use_indexes
                if (use_index, result_index) in compatible_pairs
            ]
            for result_index in result_indexes
        }
        incompatible_pairs += len(use_indexes) * len(result_indexes) - len(compatible_pairs)
        if any(len(indexes) > 1 for indexes in results_for_use.values()) or any(
            len(indexes) > 1 for indexes in uses_for_result.values()
        ):
            ambiguous_ids.add(tool_id)
        for use_index, candidates in results_for_use.items():
            if len(candidates) != 1:
                continue
            result_index = candidates[0]
            if uses_for_result[result_index] != [use_index]:
                continue
            use = correlated[use_index]
            result = correlated[result_index]
            warnings = use.warnings
            result_warnings = result.warnings
            if (
                use.timestamp is not None
                and result.timestamp is not None
                and result.timestamp < use.timestamp
            ):
                negative_intervals += 1
                warnings = (*warnings, "negative_tool_interval")
                result_warnings = (*result_warnings, "negative_tool_interval")
            correlated[use_index] = replace(
                use,
                correlated_event_id=result.event_id,
                warnings=warnings,
            )
            correlated[result_index] = replace(
                result,
                correlated_event_id=use.event_id,
                warnings=result_warnings,
            )
    stats = _CorrelationStats(
        unmatched_tool_uses=sum(
            event.kind is ContentKind.TOOL_USE
            and event.tool_use_id is not None
            and event.correlated_event_id is None
            for event in correlated
        ),
        orphan_tool_results=sum(
            event.kind is ContentKind.TOOL_RESULT
            and event.tool_use_id is not None
            and event.correlated_event_id is None
            for event in correlated
        ),
        duplicate_tool_ids=len(duplicate_ids),
        ambiguous_tool_ids=len(ambiguous_ids),
        invalid_tool_ids=sum(
            "invalid_tool_use_id" in event.warnings or "invalid_tool_result_id" in event.warnings
            for event in correlated
        ),
        negative_tool_intervals=negative_intervals,
        incompatible_tool_pairs=incompatible_pairs,
        oversized_tool_id_groups=len(oversized_ids),
        correlation_pairs_examined=pairs_examined,
    )
    return correlated, stats


def _annotate_lineages(
    events: list[NormalizedEvent], records: list[_RecordMeta]
) -> tuple[list[NormalizedEvent], int, int, int, int, int, int]:
    uuid_counts = Counter(record.uuid for record in records if record.uuid is not None)
    duplicate_uuids = {uuid for uuid, count in uuid_counts.items() if count > 1}
    records_by_uuid = {
        record.uuid: record
        for record in records
        if record.uuid is not None and record.uuid not in duplicate_uuids
    }
    dangling_uuids = {
        record.uuid
        for record in records_by_uuid.values()
        if record.parent_uuid is not None and record.parent_uuid not in uuid_counts
    }
    ambiguous_parent_uuids = {
        record.uuid for record in records_by_uuid.values() if record.parent_uuid in duplicate_uuids
    }
    raw_parent_by_uuid = {
        uuid: record.parent_uuid
        for uuid, record in records_by_uuid.items()
        if record.parent_uuid in records_by_uuid
    }

    # ``parent_by_uuid`` is a functional graph (at most one parent per node),
    # so cycles can be found iteratively in linear time without risking the
    # Python recursion limit on a valid 100k-record chain.
    processed: set[str] = set()
    cyclic_uuids: set[str] = set()
    for start_uuid in records_by_uuid:
        if start_uuid in processed:
            continue
        trail: list[str] = []
        positions: dict[str, int] = {}
        current: str | None = start_uuid
        while current is not None and current not in processed and current not in positions:
            positions[current] = len(trail)
            trail.append(current)
            current = raw_parent_by_uuid.get(current)
        if current is not None and current in positions:
            cyclic_uuids.update(trail[positions[current] :])
        processed.update(trail)

    raw_parent_by_uuid = {
        child: parent
        for child, parent in raw_parent_by_uuid.items()
        if child not in cyclic_uuids and parent not in cyclic_uuids
    }
    raw_children_by_uuid: dict[str, list[str]] = {}
    for child, parent in raw_parent_by_uuid.items():
        if parent is not None:
            raw_children_by_uuid.setdefault(parent, []).append(child)
    record_order = {uuid: record.line for uuid, record in records_by_uuid.items()}
    for children in raw_children_by_uuid.values():
        children.sort(key=record_order.__getitem__)
    structural_fork_parents = {
        parent for parent, children in raw_children_by_uuid.items() if len(children) > 1
    }

    semantic_uuids = {
        event.record_uuid
        for event in events
        if event.actor in {EventActor.ASSISTANT, EventActor.HUMAN}
        and event.record_uuid in records_by_uuid
        and event.record_uuid not in cyclic_uuids
    }
    raw_roots = sorted(
        (
            uuid
            for uuid in records_by_uuid
            if uuid not in cyclic_uuids and uuid not in raw_parent_by_uuid
        ),
        key=record_order.__getitem__,
    )

    # Project the raw metadata forest onto semantic turns. Non-semantic records
    # remain ancestry bridges, but do not create branch choices of their own.
    semantic_anchor_by_uuid: dict[str, str | None] = {}
    semantic_parent_by_uuid: dict[str, str] = {}
    for raw_root in raw_roots:
        projection_stack: list[tuple[str, str | None]] = [(raw_root, None)]
        while projection_stack:
            uuid, nearest_semantic = projection_stack.pop()
            semantic_anchor: str | None
            if uuid in semantic_uuids:
                if nearest_semantic is not None:
                    semantic_parent_by_uuid[uuid] = nearest_semantic
                semantic_anchor = uuid
            else:
                semantic_anchor = nearest_semantic
            semantic_anchor_by_uuid[uuid] = semantic_anchor
            projection_stack.extend(
                (child, semantic_anchor) for child in reversed(raw_children_by_uuid.get(uuid, ()))
            )

    semantic_children_by_uuid: dict[str, list[str]] = {}
    for child, parent in semantic_parent_by_uuid.items():
        semantic_children_by_uuid.setdefault(parent, []).append(child)
    for children in semantic_children_by_uuid.values():
        children.sort(key=record_order.__getitem__)
    semantic_fork_parents = {
        parent for parent, children in semantic_children_by_uuid.items() if len(children) > 1
    }

    def extend_summary(
        summary: tuple[str, ...], value: str, maximum: int
    ) -> tuple[tuple[str, ...], bool]:
        if len(summary) < maximum:
            return (*summary, value), False
        return (summary[0], *summary[-(maximum - 2) :], value), True

    semantic_roots = sorted(
        (uuid for uuid in semantic_uuids if uuid not in semantic_parent_by_uuid),
        key=record_order.__getitem__,
    )
    lineage_info: dict[str, _LineageInfo] = {}
    for root in semantic_roots:
        lineage_info[root] = _LineageInfo(
            root_uuid=root,
            depth=0,
            lineage=(root,),
            lineage_truncated=False,
            branch_path=(),
            branch_id=root,
            branch_depth=0,
            branch_path_truncated=False,
        )
        stack = [root]
        while stack:
            parent = stack.pop()
            parent_info = lineage_info[parent]
            children = semantic_children_by_uuid.get(parent, [])
            for child in reversed(children):
                lineage, just_truncated = extend_summary(
                    parent_info.lineage,
                    child,
                    MAX_LINEAGE_SUMMARY_IDS,
                )
                branch_path = parent_info.branch_path
                branch_id = parent_info.branch_id
                branch_depth = parent_info.branch_depth
                branch_just_truncated = False
                if parent in semantic_fork_parents:
                    branch_path, branch_just_truncated = extend_summary(
                        branch_path,
                        child,
                        MAX_BRANCH_SUMMARY_IDS,
                    )
                    branch_id = hashlib.sha256(f"{branch_id}\0{child}".encode()).hexdigest()[:16]
                    branch_depth += 1
                lineage_info[child] = _LineageInfo(
                    root_uuid=parent_info.root_uuid,
                    depth=parent_info.depth + 1,
                    lineage=lineage,
                    lineage_truncated=(parent_info.lineage_truncated or just_truncated),
                    branch_path=branch_path,
                    branch_id=branch_id,
                    branch_depth=branch_depth,
                    branch_path_truncated=(
                        parent_info.branch_path_truncated or branch_just_truncated
                    ),
                )
                stack.append(child)

    # Pre/post ancestry intervals make ancestor checks constant-time and avoid
    # retaining an unbounded full lineage on every normalized event.
    traversal_index = 0
    for root in semantic_roots:
        interval_stack: list[tuple[str, bool]] = [(root, False)]
        while interval_stack:
            uuid, exiting = interval_stack.pop()
            info = lineage_info[uuid]
            if exiting:
                info.ancestry_end = traversal_index
                continue
            info.ancestry_start = traversal_index
            traversal_index += 1
            interval_stack.append((uuid, True))
            interval_stack.extend(
                (child, False) for child in reversed(semantic_children_by_uuid.get(uuid, ()))
            )

    latest_main_by_root: dict[str, str] = {}
    for record in records:
        if not record.is_sidechain and record.uuid is not None and record.uuid in semantic_uuids:
            latest_main_by_root[lineage_info[record.uuid].root_uuid] = record.uuid
    main_lineage_by_root: dict[str, set[str]] = {}
    for root_uuid, latest_main_uuid in latest_main_by_root.items():
        main_lineage: set[str] = set()
        main_current: str | None = latest_main_uuid
        while main_current is not None and main_current in lineage_info:
            main_lineage.add(main_current)
            main_current = semantic_parent_by_uuid.get(main_current)
        main_lineage_by_root[root_uuid] = main_lineage

    annotated: list[NormalizedEvent] = []
    for event in events:
        if event.record_uuid is None:
            annotated.append(
                replace(
                    event,
                    lineage_valid=False,
                    warnings=(*event.warnings, "no_valid_record_uuid"),
                )
            )
            continue
        if event.record_uuid in duplicate_uuids:
            annotated.append(
                replace(
                    event,
                    lineage=(),
                    lineage_valid=False,
                    warnings=(*event.warnings, "duplicate_uuid"),
                )
            )
            continue
        if event.record_uuid in cyclic_uuids:
            annotated.append(
                replace(
                    event,
                    lineage=(),
                    lineage_valid=False,
                    warnings=(*event.warnings, "cyclic_ancestry"),
                )
            )
            continue
        event_semantic_anchor = semantic_anchor_by_uuid.get(event.record_uuid)
        warnings = list(event.warnings)
        if event.record_uuid in dangling_uuids:
            warnings.append("dangling_parent")
        if event.record_uuid in ambiguous_parent_uuids:
            warnings.append("ambiguous_parent_uuid")
        if event_semantic_anchor is None:
            annotated.append(
                replace(
                    event,
                    lineage=(),
                    lineage_root_uuid=None,
                    branch_path=(),
                    branch_id=None,
                    ancestry_start=None,
                    ancestry_end=None,
                    lineage_valid=False,
                    warnings=(*warnings, "no_semantic_ancestor"),
                )
            )
            continue
        info = lineage_info[event_semantic_anchor]
        if info.lineage_truncated:
            warnings.append("lineage_summary_truncated")
        if info.branch_path_truncated:
            warnings.append("branch_summary_truncated")
        anchor_record = records_by_uuid[event_semantic_anchor]
        component_main_lineage = main_lineage_by_root.get(info.root_uuid, set())
        annotated.append(
            replace(
                event,
                lineage=info.lineage,
                lineage_root_uuid=info.root_uuid,
                lineage_depth=info.depth,
                lineage_truncated=info.lineage_truncated,
                branch_path=info.branch_path,
                branch_id=info.branch_id,
                branch_depth=info.branch_depth,
                branch_path_truncated=info.branch_path_truncated,
                ancestry_start=info.ancestry_start,
                ancestry_end=info.ancestry_end,
                is_abandoned_branch=(
                    not anchor_record.is_sidechain
                    and bool(component_main_lineage)
                    and event_semantic_anchor not in component_main_lineage
                ),
                warnings=tuple(warnings),
            )
        )
    return (
        annotated,
        len(semantic_fork_parents),
        len(structural_fork_parents),
        len(dangling_uuids),
        len(cyclic_uuids),
        len(duplicate_uuids),
        len(ambiguous_parent_uuids),
    )


def parse_session(
    path: Path,
    *,
    include_thinking: bool = True,
    limits: AuditLimits | None = None,
) -> ParsedSession:
    """Stream *path* and normalize supported visible message text."""
    effective_limits = limits or AuditLimits()
    source_path = Path(path)
    initial_size = source_path.stat().st_size
    if initial_size > effective_limits.max_input_bytes:
        return ParsedSession(
            status=AnalysisStatus.REFUSED,
            session=SessionMetadata(input_bytes=initial_size, sha256=""),
            events=(),
            diagnostics=ParserDiagnostics(
                digest_complete=False,
                refusal_reasons=("input_byte_limit",),
            ),
        )

    digest = hashlib.sha256()
    input_bytes = 0
    normalized_event_count = 0
    nonblank_lines = 0
    valid_json_objects = 0
    non_object_json_values = 0
    message_like_records = 0
    all_message_records_sidechain = True
    recognized_message_records = 0
    failed_message_records = 0
    empty_message_records = 0
    internal_metadata_records = 0
    invalid_field_types = 0
    unknown_content_blocks = 0
    normalizable_actor_records = 0
    unknown_record_types = 0
    malformed_lines = 0
    last_malformed_line = 0
    last_nonblank_line = 0
    events: list[NormalizedEvent] = []
    records: list[_RecordMeta] = []
    resource_refusal: str | None = None
    digest_complete = False

    with source_path.open("rb") as handle:
        line_number = 0
        while True:
            read_limit = min(
                effective_limits.max_input_bytes - input_bytes + 1,
                effective_limits.max_input_line_bytes + 1,
            )
            raw_line = handle.readline(read_limit)
            if not raw_line:
                digest_complete = True
                break
            line_number += 1
            input_bytes += len(raw_line)
            digest.update(raw_line)
            if input_bytes > effective_limits.max_input_bytes:
                resource_refusal = "input_byte_limit"
                break
            if len(raw_line) > effective_limits.max_input_line_bytes:
                resource_refusal = "input_line_byte_limit"
                break
            if not raw_line.strip():
                continue
            nonblank_lines += 1
            last_nonblank_line = line_number
            if nonblank_lines > effective_limits.max_input_records:
                resource_refusal = "input_record_limit"
                break
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed_lines += 1
                last_malformed_line = line_number
                continue
            except (RecursionError, ValueError):
                resource_refusal = "json_complexity_limit"
                break
            if not isinstance(value, dict):
                non_object_json_values += 1
                continue
            valid_json_objects += 1
            record_type = value.get("type")
            if not isinstance(record_type, str):
                unknown_record_types += 1
                invalid_field_types += 1
                continue
            record, timestamp, metadata_invalid_fields = _record_metadata(
                value,
                line_number=line_number,
                record_type=record_type,
            )
            if record.uuid is not None:
                records.append(record)
            invalid_field_types += metadata_invalid_fields
            if record_type not in {"assistant", "user", "system"}:
                unknown_record_types += 1
                continue
            if record_type not in {"assistant", "user"}:
                continue
            message_like_records += 1
            all_message_records_sidechain = all_message_records_sidechain and record.is_sidechain
            internal_metadata = _is_internal_metadata_record(value, record_type)
            if internal_metadata:
                internal_metadata_records += 1
            message = value.get("message")
            if not isinstance(message, dict):
                failed_message_records += 1
                invalid_field_types += 1
                continue
            content = message.get("content")
            if not isinstance(content, (str, list)):
                failed_message_records += 1
                invalid_field_types += 1
                continue
            recognized_message_records += 1
            content_invalid_fields, content_unknown_blocks = _content_diagnostics(
                content, record_type
            )
            invalid_field_types += content_invalid_fields
            unknown_content_blocks += content_unknown_blocks
            if _is_known_empty_content(content, record_type):
                empty_message_records += 1
            elif not _has_normalizable_message_content(content, record_type):
                failed_message_records += 1
            if not internal_metadata and _has_normalizable_actor_content(
                content,
                record_type,
                include_thinking=include_thinking,
            ):
                normalizable_actor_records += 1
            for block in _message_blocks(
                content,
                record_type=record_type,
                include_thinking=include_thinking,
                internal_metadata=internal_metadata,
            ):
                invalid_field_types += block.invalid_field_types
                unknown_content_blocks += block.unknown_content_blocks
                normalized_event_count += 1
                if normalized_event_count > effective_limits.max_normalized_events:
                    resource_refusal = "normalized_event_limit"
                    break
                text_limit = (
                    MAX_TOOL_RESULT_TEXT_CHARS
                    if block.kind is ContentKind.TOOL_RESULT
                    else MAX_SEMANTIC_TEXT_CHARS
                )
                (
                    retained_text,
                    bounded_text_truncated,
                    bounded_unicode_replaced,
                ) = _bounded_text(block.text, text_limit)
                text_truncated = block.text_truncated or bounded_text_truncated
                event_warnings = (*record.warnings, *block.warnings)
                if text_truncated:
                    event_warnings = (*event_warnings, "text_truncated")
                if block.unicode_replaced or bounded_unicode_replaced:
                    event_warnings = (*event_warnings, "unpaired_surrogate_replaced")
                events.append(
                    NormalizedEvent(
                        event_id=f"L{line_number}:B{block.index}",
                        chronological_index=len(events),
                        source_line=line_number,
                        content_block_index=block.index,
                        timestamp=timestamp,
                        actor=block.actor,
                        kind=block.kind,
                        source_kind=block.source_kind,
                        text=retained_text,
                        text_truncated=text_truncated,
                        record_uuid=record.uuid,
                        parent_uuid=record.parent_uuid,
                        is_sidechain=record.is_sidechain,
                        source_tool_assistant_uuid=record.source_tool_assistant_uuid,
                        tool_use_id=block.tool_use_id,
                        tool_name=block.tool_name,
                        tool_input=block.tool_input,
                        tool_input_truncated_fields=block.tool_input_truncated_fields,
                        tool_result_is_error=block.tool_result_is_error,
                        warnings=event_warnings,
                    )
                )
            if resource_refusal is not None:
                break

    if resource_refusal is not None:
        return ParsedSession(
            status=AnalysisStatus.REFUSED,
            session=SessionMetadata(input_bytes=input_bytes, sha256=""),
            events=(),
            diagnostics=ParserDiagnostics(
                nonblank_lines=nonblank_lines,
                valid_json_objects=valid_json_objects,
                non_object_json_values=non_object_json_values,
                message_like_records=message_like_records,
                recognized_message_records=recognized_message_records,
                normalized_events=normalized_event_count,
                unknown_record_types=unknown_record_types,
                malformed_lines=malformed_lines,
                failed_message_records=failed_message_records,
                empty_message_records=empty_message_records,
                internal_metadata_records=internal_metadata_records,
                invalid_field_types=invalid_field_types,
                unknown_content_blocks=unknown_content_blocks,
                sidechain_only=message_like_records > 0 and all_message_records_sidechain,
                digest_complete=False,
                refusal_reasons=(resource_refusal,),
            ),
        )

    (
        events,
        forks,
        structural_forks,
        dangling_parents,
        cyclic_uuids,
        duplicate_uuids,
        ambiguous_parent_uuids,
    ) = _annotate_lineages(events, records)
    events, correlation = _correlate_tool_events(events)
    trailing_partial_lines = int(malformed_lines > 0 and last_malformed_line == last_nonblank_line)
    malformed_for_threshold = malformed_lines - trailing_partial_lines
    refusal_reasons: list[str] = []
    if nonblank_lines > 0 and valid_json_objects + non_object_json_values == 0:
        refusal_reasons.append("no_complete_records")
    if malformed_for_threshold > max(1, nonblank_lines // 100):
        refusal_reasons.append("malformed_line_threshold")
    if failed_message_records > max(1, message_like_records // 20):
        refusal_reasons.append("message_normalization_threshold")
    if valid_json_objects + non_object_json_values > 0 and normalizable_actor_records == 0:
        refusal_reasons.append("unsupported_schema")
    status = AnalysisStatus.REFUSED if refusal_reasons else AnalysisStatus.COMPLETE
    diagnostics = ParserDiagnostics(
        nonblank_lines=nonblank_lines,
        valid_json_objects=valid_json_objects,
        non_object_json_values=non_object_json_values,
        message_like_records=message_like_records,
        recognized_message_records=recognized_message_records,
        normalized_events=len(events),
        unknown_record_types=unknown_record_types,
        malformed_lines=malformed_lines,
        trailing_partial_lines=trailing_partial_lines,
        failed_message_records=failed_message_records,
        empty_message_records=empty_message_records,
        internal_metadata_records=internal_metadata_records,
        invalid_field_types=invalid_field_types,
        unknown_content_blocks=unknown_content_blocks,
        forks=forks,
        structural_forks=structural_forks,
        dangling_parents=dangling_parents,
        cyclic_uuids=cyclic_uuids,
        duplicate_uuids=duplicate_uuids,
        ambiguous_parent_uuids=ambiguous_parent_uuids,
        sidechain_only=message_like_records > 0 and all_message_records_sidechain,
        unmatched_tool_uses=correlation.unmatched_tool_uses,
        orphan_tool_results=correlation.orphan_tool_results,
        duplicate_tool_ids=correlation.duplicate_tool_ids,
        ambiguous_tool_ids=correlation.ambiguous_tool_ids,
        invalid_tool_ids=correlation.invalid_tool_ids,
        negative_tool_intervals=correlation.negative_tool_intervals,
        incompatible_tool_pairs=correlation.incompatible_tool_pairs,
        oversized_tool_id_groups=correlation.oversized_tool_id_groups,
        correlation_pairs_examined=correlation.correlation_pairs_examined,
        text_excerpts_truncated=sum(event.text_truncated for event in events),
        tool_input_fields_truncated=sum(len(event.tool_input_truncated_fields) for event in events),
        lineage_summaries_truncated=sum(event.lineage_truncated for event in events),
        branch_summaries_truncated=sum(event.branch_path_truncated for event in events),
        unicode_replacement_events=sum(
            "unpaired_surrogate_replaced" in event.warnings for event in events
        ),
        digest_complete=digest_complete,
        refusal_reasons=tuple(refusal_reasons),
    )
    return ParsedSession(
        status=status,
        session=SessionMetadata(
            input_bytes=input_bytes,
            sha256=digest.hexdigest() if digest_complete else "",
        ),
        events=() if status is AnalysisStatus.REFUSED else tuple(events),
        diagnostics=diagnostics,
    )
