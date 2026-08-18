"""High-precision deterministic mistake candidate and episode detection."""

from __future__ import annotations

import hashlib
import heapq
import re
import shlex
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace

from scripts.core.session_audit.models import (
    MAX_LOCAL_CANDIDATES,
    Classification,
    ContentKind,
    DetectionResult,
    DetectorDiagnostics,
    Episode,
    EventActor,
    EvidenceKind,
    EvidenceRef,
    EvidenceSourceKind,
    GapRef,
    NormalizedEvent,
    ObjectiveChain,
    ParsedSession,
    SignalKind,
)
from scripts.core.session_audit.parser import events_share_lineage

_MAX_DEICTIC_EVENT_DISTANCE = 8
_ARTIFACT_FIELDS = frozenset({"file_path", "notebook_path"})
_CAUSAL_ACTION_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit", "Bash"})
_COMMAND_OBJECT_NOUNS = frozenset({"command", "flag", "argument", "option", "profile"})


@dataclass(frozen=True)
class DetectionPolicy:
    """Versioned, bounded local detector policy."""

    max_candidates: int = MAX_LOCAL_CANDIDATES
    backward_window_events: int = 64
    forward_window_events: int = 64

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.backward_window_events < 0 or self.forward_window_events < 0:
            raise ValueError("window event limits must be nonnegative")


_ADMISSION_LEXEME_RE = re.compile(
    r"\b(?:"
    r"I\s+was\s+wrong\b|"
    r"I\s+(?:misunderstood|misread|misinterpreted)\b|"
    r"I\s+(?:incorrectly|wrongly)\s+(?:assumed|thought|said|stated|changed|edited|"
    r"implemented|used|claimed)\b|"
    r"my\s+(?:assumption|diagnosis|interpretation|change|edit)\s+(?:was|is)\s+"
    r"(?:wrong|incorrect)\b|"
    r"(?:that|this)\s+was\s+my\s+(?:mistake|error)\b|"
    r"I\s+(?:made|introduced|caused)\s+(?:an?|the)\s+(?:mistake|error|bug)\b|"
    r"I\s+should\s+have\s+(?:checked|read|verified|tested|confirmed)\b"
    r")",
    re.IGNORECASE,
)
_USER_CORRECTION_RE = re.compile(
    r"\b(?:you\s+(?:were|are)\s+(?:wrong|incorrect)|"
    r"you\s+(?:misunderstood|assumed|changed\s+the\s+wrong|used\s+the\s+wrong)|"
    r"(?:that|this)\s+(?:is|was|'s)\s+(?:wrong|incorrect)|"
    r"not\s+what\s+I\s+(?:asked|said))\b",
    re.IGNORECASE,
)
_FENCE_LINE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*(?:`|$)")
_QUOTED_TEXT_RE = re.compile(r'("[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|(?<!\w)\'[^\'\n]*\'(?!\w))')
_NON_MISTAKE_ACTION_TAIL_RE = re.compile(
    r"\b(?:as\s+requested|intentionally|deliberately|purposely|on\s+purpose|by\s+design|"
    r"to\s+test|(?:for|as)\s+(?:(?:a|the)\s+)?(?:negative\s+)?test(?:ing)?)\b",
    re.IGNORECASE,
)
_ADMISSION_CLAUSE_RE = re.compile(
    r"^\s*(?:(?:actually|admittedly|clearly|yes|right|you(?:'re|\s+are)\s+right|"
    r"sorry(?:\s+about\s+that)?|I\s+apologize|on\s+reflection|after\s+checking)"
    r"(?:(?:\s*[—–]\s*)|(?:\s*[,:-]\s+)|\s+))?(?:"
    r"Correction\s+first\s*:\s*[^.!?;\n]{1,160}?[—–]\s*"
    r"I\s+gave\s+you\s+the\s+wrong\s+command\b|"
    r"(?P<guarded_action>I\s+(?:(?:gave|sent)\s+you|provided|ran|used|passed)\s+the\s+"
    r"(?:wrong|incorrect)\s+(?:command|flag|argument|option|profile)\b"
    r"|The\s+command\s+I\s+gave\s+you\s+was\s+(?:wrong|incorrect)\b"
    r"|Correction\s*:\s*I\s+dropped\s+the\s+[\w./-]+\s+flag\b)|"
    r"I\s+accidentally\s+omitted\s+the\s+[\w./-]+\s+argument\b|"
    r"I\s+forgot\s+to\s+include\s+the\s+[\w./-]+\s+option\b|"
    r"I\s+should\s+have\s+passed\s+the\s+[\w./-]+\s+profile\b|"
    r"I\s+was\s+wrong\b|"
    r"I\s+(?:misunderstood|misread|misinterpreted)\b|"
    r"I\s+(?:incorrectly|wrongly)\s+(?:assumed|thought|said|stated|changed|edited|"
    r"implemented|used|claimed)\b|"
    r"my\s+(?:assumption|diagnosis|interpretation|change|edit)\s+(?:was|is)\s+"
    r"(?:wrong|incorrect)\b|"
    r"(?:that|this)\s+was\s+my\s+(?:mistake|error)\b|"
    r"I\s+(?:made|introduced|caused)\s+(?:an?|the)\s+(?:mistake|error|bug)\b|"
    r"I\s+should\s+have\s+(?:checked|read|verified|tested|confirmed)\b"
    r")",
    re.IGNORECASE,
)
_RETRACTION_TAIL_RE = re.compile(
    r"\b(?:or\s+maybe\s+not|but\s+perhaps\s+not|unless\b|"
    r"I(?:'m|\s+am)\s+not\s+sure|I\s+still\s+need\s+to\s+verify)\b",
    re.IGNORECASE,
)
_EXPECTED_FAILURE_RE = re.compile(
    r"\b(?:tdd|red\s+phase|write\s+(?:a\s+)?failing\s+test|"
    r"(?:expect(?:ed)?|should)\b[^.!?]*\bfail)\b",
    re.IGNORECASE,
)
_TRANSIENT_FAILURE_RE = re.compile(
    r"\b(?:timed?\s*out|timeout|rate\s*limit|too\s+many\s+requests|permission\s+denied|"
    r"connection\s+(?:reset|refused|failed|timed)|temporary(?:ily)?\s+unavailable|"
    r"service\s+unavailable|http\s*429)\b",
    re.IGNORECASE,
)
_REQUIREMENT_CHANGE_RE = re.compile(
    r"\b(?:I\s+changed\s+my\s+mind|I\s+(?:want|prefer|decided)|"
    r"let(?:'s|\s+us)\b[^.!?]*\binstead|new\s+requirement|"
    r"different\s+tradeoff|from\s+now\s+on)\b",
    re.IGNORECASE,
)
_GENERIC_APOLOGY_RE = re.compile(r"\b(?:sorry|I\s+apologize)\b", re.IGNORECASE)
_DEICTIC_ACTION_RE = re.compile(
    r"\b(?:that|this|the)\s+(?:[\w./-]+\s+){0,3}"
    r"(?P<action_noun>edit|change|write|command|flag|argument|option|profile|revert|patch|"
    r"implementation)\b",
    re.IGNORECASE,
)
_ARTIFACT_MENTION_RE = re.compile(
    r"(?<![\w.-])(?P<artifact>(?:[A-Za-z]:)?"
    r"(?:(?:[/\\]?[\w.-]+)[/\\])*[\w.-]*[\w-]\.[A-Za-z0-9]+)"
    r"(?::\d+)?(?=$|[\s),;!?]|\.(?=$|\s))"
)
_EXPLICIT_ACKNOWLEDGMENT_RE = re.compile(
    r"^\s*you(?:'re|\s+are)\s+right\b",
    re.IGNORECASE,
)


def _strip_fenced_blocks(text: str) -> str:
    retained: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in text.splitlines():
        match = _FENCE_LINE_RE.match(line)
        if fence_character is not None:
            if (
                match is not None
                and match.group(1)[0] == fence_character
                and len(match.group(1)) >= fence_length
            ):
                fence_character = None
                fence_length = 0
            continue
        if match is not None:
            fence_character = match.group(1)[0]
            fence_length = len(match.group(1))
            continue
        retained.append(line)
    return "\n".join(retained)


def _prose_for_signal_matching(text: str) -> str:
    without_fences = _strip_fenced_blocks(text)
    without_inline = _INLINE_CODE_RE.sub(" ", without_fences)
    without_quotes = _QUOTED_TEXT_RE.sub(" ", without_inline)
    return "\n".join(
        line for line in without_quotes.splitlines() if not line.lstrip().startswith(">")
    )


def _has_specific_visible_admission(text: str) -> bool:
    prose = _prose_for_signal_matching(text)
    if _RETRACTION_TAIL_RE.search(prose):
        return False
    clauses = re.split(r"(?<=[.!?])\s+|[;\n]+", prose)
    for clause in clauses:
        if not clause.strip() or clause.rstrip().endswith("?"):
            continue
        match = _ADMISSION_CLAUSE_RE.search(clause)
        if match is None:
            continue
        if match.group("guarded_action") is not None and _NON_MISTAKE_ACTION_TAIL_RE.search(
            clause, match.end("guarded_action")
        ):
            continue
        return True
    return False


def _is_suppressed_admission_like(event: NormalizedEvent) -> bool:
    if (
        event.actor is not EventActor.ASSISTANT
        or event.kind is not ContentKind.VISIBLE_TEXT
        or event.source_kind is not EvidenceSourceKind.ASSISTANT_TEXT
    ):
        return False
    if _has_specific_visible_admission(event.text):
        return False
    return bool(
        _GENERIC_APOLOGY_RE.search(_prose_for_signal_matching(event.text))
        or _ADMISSION_LEXEME_RE.search(event.text)
    )


def _text_names_selector(normalized_text: str, selector: str) -> bool:
    normalized_selector = " ".join(selector.casefold().split())
    if not normalized_selector:
        return False
    trailing_boundary = r"(?=$|[\s),;!?]|\.(?=$|\s)|:\d+(?=$|[\s),.;!?]))"
    return (
        re.search(
            rf"(?<![\w.-]){re.escape(normalized_selector)}{trailing_boundary}",
            normalized_text,
        )
        is not None
    )


def _has_conflicting_artifact_reference(text: str, root: NormalizedEvent) -> bool:
    root_artifacts = {
        value.casefold().replace("\\", "/")
        for field, value in root.tool_input
        if field in _ARTIFACT_FIELDS and value
    }
    if not root_artifacts:
        return False
    accepted_identities = root_artifacts | {
        re.split(r"[/\\]", artifact)[-1] for artifact in root_artifacts
    }
    mentions = {
        match.group("artifact").casefold().replace("\\", "/")
        for match in _ARTIFACT_MENTION_RE.finditer(text)
    }
    return bool(mentions) and not mentions <= accepted_identities


def _text_names_event_selector(normalized_text: str, event: NormalizedEvent) -> bool:
    for field, value in event.tool_input:
        if _text_names_selector(normalized_text, value):
            return True
        if field in _ARTIFACT_FIELDS:
            basename = re.split(r"[/\\]", value)[-1]
            if _text_names_selector(normalized_text, basename):
                return True
    return False


def _is_ignorable_bridge_context(
    event: NormalizedEvent,
    detection: NormalizedEvent,
) -> bool:
    if (
        event.record_uuid is None
        or event.record_uuid != detection.record_uuid
        or event.actor is not EventActor.ASSISTANT
    ):
        return False
    return (
        event.kind is ContentKind.THINKING
        and event.source_kind is EvidenceSourceKind.ASSISTANT_THINKING
    ) or (event.kind is ContentKind.METADATA and event.source_kind is EvidenceSourceKind.DERIVED)


_LineageKey = tuple[str, int, int]


def _lineage_key(event: NormalizedEvent) -> _LineageKey | None:
    if (
        not event.lineage_valid
        or event.record_uuid is None
        or event.lineage_root_uuid is None
        or event.ancestry_start is None
        or event.ancestry_end is None
    ):
        return None
    return event.lineage_root_uuid, event.ancestry_start, event.ancestry_end


class _LineageWindowIndex:
    """Exact bounded windows indexed by validated ancestry interval."""

    def __init__(self, ordered: tuple[NormalizedEvent, ...]) -> None:
        self._ordered = ordered
        key_by_position: list[_LineageKey | None] = []
        mutable_positions: dict[_LineageKey, list[int]] = {}
        mutable_keys_by_root: dict[str, set[_LineageKey]] = {}
        for position, event in enumerate(ordered):
            key = _lineage_key(event)
            key_by_position.append(key)
            if key is None:
                continue
            mutable_positions.setdefault(key, []).append(position)
            mutable_keys_by_root.setdefault(key[0], set()).add(key)
        self._key_by_position = tuple(key_by_position)
        self._positions_by_key = {
            key: tuple(positions) for key, positions in mutable_positions.items()
        }
        self._keys_by_root = {
            root: tuple(sorted(keys, key=lambda key: (key[1], -key[2])))
            for root, keys in mutable_keys_by_root.items()
        }
        self._parent_by_key: dict[_LineageKey, _LineageKey | None] = {}
        mutable_children: dict[_LineageKey, list[_LineageKey]] = {}
        self._laminar_roots: set[str] = set()
        for root, keys in self._keys_by_root.items():
            stack: list[_LineageKey] = []
            root_parents: dict[_LineageKey, _LineageKey | None] = {}
            root_children: dict[_LineageKey, list[_LineageKey]] = {}
            laminar = True
            for key in keys:
                start, end = key[1], key[2]
                if start >= end:
                    laminar = False
                    break
                while stack and start >= stack[-1][2]:
                    stack.pop()
                parent = stack[-1] if stack else None
                if parent is not None and end > parent[2]:
                    laminar = False
                    break
                root_parents[key] = parent
                if parent is not None:
                    root_children.setdefault(parent, []).append(key)
                stack.append(key)
            if not laminar:
                continue
            self._laminar_roots.add(root)
            self._parent_by_key.update(root_parents)
            for key, children in root_children.items():
                mutable_children.setdefault(key, []).extend(children)
        self._children_by_key = {key: tuple(children) for key, children in mutable_children.items()}
        self._window_cache: dict[tuple[int, int, bool], tuple[NormalizedEvent, ...]] = {}

    @staticmethod
    def _keys_are_compatible(first: _LineageKey, second: _LineageKey) -> bool:
        if first[0] != second[0]:
            return False
        return (first[1] <= second[1] and second[2] <= first[2]) or (
            second[1] <= first[1] and first[2] <= second[2]
        )

    def _compatible_keys(self, anchor_key: _LineageKey) -> tuple[_LineageKey, ...]:
        if anchor_key[0] not in self._laminar_roots:
            return tuple(
                key
                for key in self._keys_by_root.get(anchor_key[0], ())
                if self._keys_are_compatible(anchor_key, key)
            )
        compatible: list[_LineageKey] = []
        current: _LineageKey | None = anchor_key
        while current is not None:
            compatible.append(current)
            current = self._parent_by_key[current]
        stack = list(self._children_by_key.get(anchor_key, ()))
        while stack:
            descendant = stack.pop()
            compatible.append(descendant)
            stack.extend(self._children_by_key.get(descendant, ()))
        return tuple(compatible)

    def side(
        self,
        anchor_index: int,
        anchor: NormalizedEvent,
        *,
        limit: int,
        backwards: bool,
    ) -> tuple[NormalizedEvent, ...]:
        if limit <= 0:
            return ()
        cache_key = (anchor_index, limit, backwards)
        cached = self._window_cache.get(cache_key)
        if cached is not None:
            return cached
        anchor_key = self._key_by_position[anchor_index]
        if anchor_key is None:
            result = self._physical_side(
                anchor_index,
                anchor,
                limit=limit,
                backwards=backwards,
            )
            self._window_cache[cache_key] = result
            return result
        selected: list[int] = []
        for key in self._compatible_keys(anchor_key):
            positions = self._positions_by_key[key]
            if backwards:
                boundary = bisect_left(positions, anchor_index)
                relevant = positions[max(0, boundary - limit) : boundary]
                for position in relevant:
                    if len(selected) < limit:
                        heapq.heappush(selected, position)
                    elif position > selected[0]:
                        heapq.heapreplace(selected, position)
            else:
                boundary = bisect_right(positions, anchor_index)
                relevant = positions[boundary : boundary + limit]
                for position in relevant:
                    if len(selected) < limit:
                        heapq.heappush(selected, -position)
                    elif position < -selected[0]:
                        heapq.heapreplace(selected, -position)
        selected_positions = (
            sorted(selected) if backwards else sorted(-position for position in selected)
        )
        result = tuple(self._ordered[position] for position in selected_positions)
        self._window_cache[cache_key] = result
        return result

    def _physical_side(
        self,
        anchor_index: int,
        anchor: NormalizedEvent,
        *,
        limit: int,
        backwards: bool,
    ) -> tuple[NormalizedEvent, ...]:
        step = -1 if backwards else 1
        index = anchor_index + step
        selected: list[NormalizedEvent] = []
        while 0 <= index < len(self._ordered) and len(selected) < limit:
            event = self._ordered[index]
            if events_share_lineage(event, anchor):
                selected.append(event)
            index += step
        if backwards:
            selected.reverse()
        return tuple(selected)


def _linked_affected_event(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    event_position_by_id: dict[str, int],
    detection_index: int,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
    *,
    allow_user_bridge: bool = True,
) -> NormalizedEvent | None:
    detection = ordered[detection_index]
    prior_window = _compatible_window_side(
        lineage_windows,
        detection_index,
        detection,
        limit=policy.backward_window_events,
        backwards=True,
    )
    candidates = tuple(
        event
        for event in prior_window
        if event.kind is ContentKind.TOOL_USE and event.tool_name in _CAUSAL_ACTION_TOOLS
    )
    detection_text = " ".join(_prose_for_signal_matching(detection.text).casefold().split())

    explicitly_referenced = tuple(
        event for event in candidates if _text_names_event_selector(detection_text, event)
    )
    if len(explicitly_referenced) == 1:
        return explicitly_referenced[0]
    last_boundary = next(
        (
            index
            for index, event in reversed(tuple(enumerate(prior_window)))
            if event.actor in {EventActor.HUMAN, EventActor.SYSTEM}
        ),
        -1,
    )
    deictic_match = _DEICTIC_ACTION_RE.search(_prose_for_signal_matching(detection.text))
    candidates_since_boundary = (
        tuple(
            event
            for event in prior_window[last_boundary + 1 :]
            if event.kind is ContentKind.TOOL_USE
            and event.tool_name in _CAUSAL_ACTION_TOOLS
            and _matches_deictic_action(event, deictic_match.group("action_noun"))
        )
        if deictic_match is not None
        else ()
    )
    if len(candidates_since_boundary) == 1:
        candidate = candidates_since_boundary[0]
        candidate_position = prior_window.index(candidate)
        if len(prior_window) - candidate_position - 1 <= _MAX_DEICTIC_EVENT_DISTANCE:
            return candidate
    if (
        len(candidates_since_boundary) > 1
        and deictic_match is not None
        and deictic_match.group("action_noun").casefold() in _COMMAND_OBJECT_NOUNS
    ):
        retry_root = _unique_completed_material_retry_root(
            ordered,
            event_by_id,
            event_position_by_id,
            prior_window[last_boundary + 1 :],
            policy,
            lineage_windows,
        )
        if retry_root is not None:
            return retry_root
    if not allow_user_bridge or not prior_window:
        return None
    previous = next(
        (
            event
            for event in reversed(prior_window)
            if not _is_ignorable_bridge_context(event, detection)
        ),
        None,
    )
    if (
        previous is not None
        and detection.actor is EventActor.ASSISTANT
        and previous.actor is EventActor.HUMAN
        and previous.kind is ContentKind.VISIBLE_TEXT
        and previous.source_kind is EvidenceSourceKind.USER_PROMPT
        and _USER_CORRECTION_RE.search(_prose_for_signal_matching(previous.text))
    ):
        bridged_root = _linked_affected_event(
            ordered,
            event_by_id,
            event_position_by_id,
            event_position_by_id[previous.event_id],
            policy,
            lineage_windows,
            allow_user_bridge=False,
        )
        if bridged_root is None:
            return None
        if _has_conflicting_artifact_reference(detection_text, bridged_root):
            return None
        deictic_match = _DEICTIC_ACTION_RE.search(_prose_for_signal_matching(detection.text))
        bridge_is_supported = (
            _text_names_event_selector(detection_text, bridged_root)
            or (
                deictic_match is not None
                and _matches_deictic_action(
                    bridged_root,
                    deictic_match.group("action_noun"),
                )
            )
            or _EXPLICIT_ACKNOWLEDGMENT_RE.search(detection_text) is not None
        )
        return bridged_root if bridge_is_supported else None
    return None


def _compatible_window_side(
    lineage_windows: _LineageWindowIndex,
    anchor_index: int,
    anchor: NormalizedEvent,
    *,
    limit: int,
    backwards: bool,
) -> tuple[NormalizedEvent, ...]:
    return lineage_windows.side(
        anchor_index,
        anchor,
        limit=limit,
        backwards=backwards,
    )


def _artifact_selector(event: NormalizedEvent) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    return event.tool_name, event.tool_input


def _tool_value(event: NormalizedEvent, field: str) -> str:
    for key, value in reversed(event.tool_input):
        if key == field:
            return value if isinstance(value, str) else ""
    return ""


def _is_validation_use(event: NormalizedEvent) -> bool:
    if event.kind is not ContentKind.TOOL_USE or event.tool_name != "Bash":
        return False
    command = _tool_value(event, "command").strip().lower()
    return bool(
        re.match(
            r"^(?:uv\s+run\s+)?(?:pytest\b|python\s+-m\s+pytest\b|cargo\s+test\b|"
            r"npm\s+(?:run\s+)?test\b|xcodebuild\b|vendor/bin/phpunit\b)",
            command,
        )
    )


def _is_revert_use(event: NormalizedEvent) -> bool:
    if event.kind is not ContentKind.TOOL_USE or event.tool_name != "Bash":
        return False
    return bool(
        re.search(
            r"(?:^|\s)git\s+(?:restore\b|revert\b|checkout\s+--(?=\s|$))",
            _tool_value(event, "command"),
            re.IGNORECASE,
        )
    )


def _is_mutation_use(event: NormalizedEvent) -> bool:
    return event.kind is ContentKind.TOOL_USE and (
        event.tool_name in {"Edit", "Write", "MultiEdit", "NotebookEdit"} or _is_revert_use(event)
    )


def _matches_deictic_action(event: NormalizedEvent, noun: str) -> bool:
    if noun.casefold() in _COMMAND_OBJECT_NOUNS:
        return event.kind is ContentKind.TOOL_USE and event.tool_name == "Bash"
    if noun.casefold() == "revert":
        return _is_revert_use(event)
    return _is_mutation_use(event)


def _shell_parts(command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return tuple(command.split())


def _correction_matches_root(root: NormalizedEvent, correction: NormalizedEvent) -> bool:
    if _artifact_selector(root) == _artifact_selector(correction):
        return True
    root_artifacts = {
        value for field, value in root.tool_input if field in _ARTIFACT_FIELDS and value
    }
    correction_artifacts = {
        value for field, value in correction.tool_input if field in _ARTIFACT_FIELDS and value
    }
    if root_artifacts & correction_artifacts:
        return True
    if _is_revert_use(correction):
        correction_tokens = set(_shell_parts(_tool_value(correction, "command")))
        if root_artifacts & correction_tokens:
            return True
    if _is_revert_use(root):
        root_tokens = set(_shell_parts(_tool_value(root, "command")))
        if correction_artifacts & root_tokens:
            return True
    return False


def _failure_names_root_artifact(root: NormalizedEvent, failure: NormalizedEvent) -> bool:
    root_artifacts = tuple(
        value for field, value in root.tool_input if field in _ARTIFACT_FIELDS and value
    )
    if len(root_artifacts) != 1:
        return False
    normalized_failure = " ".join(failure.text.casefold().split())
    root_artifact = root_artifacts[0]
    basename = re.split(r"[/\\]", root_artifact)[-1]
    return _text_names_selector(normalized_failure, root_artifact) or _text_names_selector(
        normalized_failure, basename
    )


def _correlated_success(
    event: NormalizedEvent, event_by_id: dict[str, NormalizedEvent]
) -> NormalizedEvent | None:
    if event.correlated_event_id is None:
        return None
    result = event_by_id.get(event.correlated_event_id)
    if result is None or result.tool_result_is_error is not False:
        return None
    return result


def _events_are_strictly_ordered(*events: NormalizedEvent) -> bool:
    keys = tuple((event.chronological_index, event.event_id) for event in events)
    return all(first < second for first, second in zip(keys, keys[1:], strict=False))


def _complete_objective_chain(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    contradiction_index: int,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> (
    tuple[
        NormalizedEvent,
        NormalizedEvent,
        NormalizedEvent,
        NormalizedEvent,
        NormalizedEvent,
        NormalizedEvent,
    ]
    | None
):
    contradiction = ordered[contradiction_index]
    if (
        contradiction.kind is not ContentKind.TOOL_RESULT
        or contradiction.tool_result_is_error is not True
    ):
        return None
    failed_use = event_by_id.get(contradiction.correlated_event_id or "")
    if failed_use is None or not _is_validation_use(failed_use):
        return None
    root = next(
        (
            event
            for event in reversed(
                _compatible_window_side(
                    lineage_windows,
                    contradiction_index,
                    contradiction,
                    limit=policy.backward_window_events,
                    backwards=True,
                )
            )
            if _is_mutation_use(event)
        ),
        None,
    )
    if root is None:
        return None
    root_completion = _correlated_success(root, event_by_id)
    if root_completion is None or not _events_are_strictly_ordered(
        root, root_completion, failed_use, contradiction
    ):
        return None
    correction_use: NormalizedEvent | None = None
    correction_result: NormalizedEvent | None = None
    for event in _compatible_window_side(
        lineage_windows,
        contradiction_index,
        contradiction,
        limit=policy.forward_window_events,
        backwards=False,
    ):
        if (
            correction_use is None
            and _is_mutation_use(event)
            and _correction_matches_root(root, event)
            and (_is_revert_use(event) or _failure_names_root_artifact(root, contradiction))
        ):
            successful_correction = _correlated_success(event, event_by_id)
            if successful_correction is not None and _events_are_strictly_ordered(
                contradiction, event, successful_correction
            ):
                correction_use = event
                correction_result = successful_correction
            continue
        if (
            correction_use is not None
            and _is_validation_use(event)
            and _artifact_selector(event) == _artifact_selector(failed_use)
        ):
            recovery = _correlated_success(event, event_by_id)
            if (
                recovery is not None
                and events_share_lineage(contradiction, recovery)
                and correction_result is not None
                and _events_are_strictly_ordered(correction_result, event, recovery)
                and all(
                    events_share_lineage(contradiction, component)
                    for component in (
                        root,
                        root_completion,
                        failed_use,
                        correction_use,
                        correction_result,
                        event,
                        recovery,
                    )
                )
            ):
                assert correction_result is not None
                return root, failed_use, correction_use, correction_result, event, recovery
    return None


def _is_expected_validation_failure(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    result_index: int,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> bool:
    result = ordered[result_index]
    if result.kind is not ContentKind.TOOL_RESULT or result.tool_result_is_error is not True:
        return False
    failed_use = event_by_id.get(result.correlated_event_id or "")
    if failed_use is None or not _is_validation_use(failed_use):
        return False
    return any(
        event.kind in {ContentKind.VISIBLE_TEXT, ContentKind.THINKING}
        and _EXPECTED_FAILURE_RE.search(_prose_for_signal_matching(event.text))
        for event in _compatible_window_side(
            lineage_windows,
            result_index,
            result,
            limit=policy.backward_window_events,
            backwards=True,
        )
    )


def _is_transient_unchanged_retry(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    result_index: int,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> bool:
    result = ordered[result_index]
    if (
        result.kind is not ContentKind.TOOL_RESULT
        or result.tool_result_is_error is not True
        or not _TRANSIENT_FAILURE_RE.search(result.text)
    ):
        return False
    failed_use = event_by_id.get(result.correlated_event_id or "")
    if failed_use is None:
        return False
    return any(
        event.kind is ContentKind.TOOL_USE
        and _artifact_selector(event) == _artifact_selector(failed_use)
        and _correlated_success(event, event_by_id) is not None
        for event in _compatible_window_side(
            lineage_windows,
            result_index,
            result,
            limit=policy.forward_window_events,
            backwards=False,
        )
    )


def _command_intent(command: str) -> str | None:
    parts = _shell_parts(command)
    if parts[:2] == ("uv", "run"):
        parts = parts[2:]
    if parts[:3] == ("python", "-m", "pytest"):
        return "pytest"
    return parts[0] if parts else None


def _completed_material_retry_chains(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    result_index: int,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> tuple[tuple[NormalizedEvent, NormalizedEvent, NormalizedEvent], ...]:
    result = ordered[result_index]
    if result.kind is not ContentKind.TOOL_RESULT or result.tool_result_is_error is not True:
        return ()
    failed_use = event_by_id.get(result.correlated_event_id or "")
    if failed_use is None or failed_use.tool_name != "Bash":
        return ()
    failed_command = _tool_value(failed_use, "command")
    failed_intent = _command_intent(failed_command)
    chains: list[tuple[NormalizedEvent, NormalizedEvent, NormalizedEvent]] = []
    for event in _compatible_window_side(
        lineage_windows,
        result_index,
        result,
        limit=policy.forward_window_events,
        backwards=False,
    ):
        if event.kind is not ContentKind.TOOL_USE or event.tool_name != failed_use.tool_name:
            continue
        retry_command = _tool_value(event, "command")
        if retry_command == failed_command or _command_intent(retry_command) != failed_intent:
            continue
        recovery = _correlated_success(event, event_by_id)
        if recovery is not None and events_share_lineage(result, recovery):
            chains.append((failed_use, event, recovery))
    return tuple(chains)


def _material_retry(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    result_index: int,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> tuple[NormalizedEvent, NormalizedEvent, NormalizedEvent] | None:
    chains = _completed_material_retry_chains(
        ordered,
        event_by_id,
        result_index,
        policy,
        lineage_windows,
    )
    return chains[0] if chains else None


def _unique_completed_material_retry_root(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    event_position_by_id: dict[str, int],
    active_window: tuple[NormalizedEvent, ...],
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> NormalizedEvent | None:
    active_positions = {event.event_id: index for index, event in enumerate(active_window)}
    active_ids = set(active_positions)
    chains: dict[tuple[str, str, str, str], NormalizedEvent] = {}
    for result in active_window:
        result_index = event_position_by_id[result.event_id]
        for failed_use, retry_use, recovery in _completed_material_retry_chains(
            ordered,
            event_by_id,
            result_index,
            policy,
            lineage_windows,
        ):
            chain_ids = (
                result.event_id,
                failed_use.event_id,
                retry_use.event_id,
                recovery.event_id,
            )
            if not set(chain_ids) <= active_ids:
                continue
            root_position = active_positions[failed_use.event_id]
            if len(active_window) - root_position - 1 > _MAX_DEICTIC_EVENT_DISTANCE:
                continue
            chains[chain_ids] = failed_use
    return next(iter(chains.values())) if len(chains) == 1 else None


def _thinking_correction(
    ordered: tuple[NormalizedEvent, ...],
    event_by_id: dict[str, NormalizedEvent],
    thinking_index: int,
    root: NormalizedEvent | None,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> tuple[NormalizedEvent, NormalizedEvent] | None:
    if root is None:
        return None
    thinking = ordered[thinking_index]
    for event in _compatible_window_side(
        lineage_windows,
        thinking_index,
        thinking,
        limit=policy.forward_window_events,
        backwards=False,
    ):
        if event.kind is not ContentKind.TOOL_USE or event.tool_name not in {
            "Edit",
            "Write",
            "MultiEdit",
            "NotebookEdit",
            "Bash",
        }:
            continue
        if not _correction_matches_root(root, event):
            continue
        recovery = _correlated_success(event, event_by_id)
        if recovery is not None and events_share_lineage(thinking, recovery):
            return event, recovery
    return None


def _context_window(
    ordered: tuple[NormalizedEvent, ...],
    detection_index: int,
    policy: DetectionPolicy,
    lineage_windows: _LineageWindowIndex,
) -> tuple[str, ...]:
    detection = ordered[detection_index]
    before = _compatible_window_side(
        lineage_windows,
        detection_index,
        detection,
        limit=policy.backward_window_events,
        backwards=True,
    )
    after = _compatible_window_side(
        lineage_windows,
        detection_index,
        detection,
        limit=policy.forward_window_events,
        backwards=False,
    )
    return tuple(event.event_id for event in (*before, detection, *after))


def _gap_ref(
    human_chronology: tuple[int, ...],
    first: NormalizedEvent,
    second: NormalizedEvent,
) -> GapRef:
    lower, upper = sorted((first.chronological_index, second.chronological_index))
    crosses_human = bisect_right(human_chronology, lower) < bisect_left(human_chronology, upper)
    return GapRef(
        start_event_id=first.event_id,
        end_event_id=second.event_id,
        same_lineage=events_share_lineage(first, second),
        crosses_human_boundary=crosses_human,
    )


def _episode_id(candidate_id: str) -> str:
    return f"episode-{hashlib.sha256(candidate_id.encode()).hexdigest()[:12]}"


_CLASSIFICATION_RANK = {
    Classification.UNCONFIRMED: 0,
    Classification.PROBABLE: 1,
    Classification.CONFIRMED: 2,
}


@dataclass(frozen=True)
class _ReverseLex:
    value: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReverseLex):
            return NotImplemented
        return self.value > other.value


_CandidateRank = tuple[int, int, int, int, int, _ReverseLex]


def _candidate_rank(episode: Episode, event_by_id: dict[str, NormalizedEvent]) -> _CandidateRank:
    evidence_kinds = {evidence.evidence_kind for evidence in episode.evidence}
    if EvidenceKind.VISIBLE_ADMISSION in evidence_kinds:
        strength = 5
    elif episode.objective_chains:
        strength = 4
    elif EvidenceKind.THINKING_ADMISSION in evidence_kinds:
        strength = 3
    elif EvidenceKind.USER_CORRECTION in evidence_kinds:
        strength = 2
    else:
        strength = 1
    independent_groups = len(
        {
            evidence.corroboration_group
            for evidence in episode.evidence
            if evidence.qualifies_for_promotion
        }
    )
    chronology = event_by_id[episode.detection_event_id].chronological_index
    return (
        _CLASSIFICATION_RANK[episode.local_classification],
        strength,
        independent_groups,
        len(episode.affected_event_ids),
        -chronology,
        _ReverseLex(episode.detection_event_id),
    )


def _dedupe_sorted_ids(
    ids: tuple[str, ...], event_by_id: dict[str, NormalizedEvent]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(ids),
            key=lambda event_id: (
                event_by_id[event_id].chronological_index,
                event_id,
            ),
        )
    )


def _merge_two_episodes(
    first: Episode,
    second: Episode,
    event_by_id: dict[str, NormalizedEvent],
) -> Episode:
    stronger = max(
        (first, second),
        key=lambda episode: (
            _CLASSIFICATION_RANK[episode.local_classification],
            -event_by_id[episode.detection_event_id].chronological_index,
        ),
    )
    detection = min(
        (first.detection_event_id, second.detection_event_id),
        key=lambda event_id: (event_by_id[event_id].chronological_index, event_id),
    )
    recovery_ids = tuple(
        event_id
        for event_id in (first.recovery_end_event_id, second.recovery_end_event_id)
        if event_id is not None
    )
    recovery = (
        max(
            recovery_ids,
            key=lambda event_id: (event_by_id[event_id].chronological_index, event_id),
        )
        if recovery_ids
        else None
    )
    onset_ids = tuple(
        event_id
        for event_id in (first.onset_event_id, second.onset_event_id)
        if event_id is not None
    )
    onset = (
        min(
            onset_ids,
            key=lambda event_id: (event_by_id[event_id].chronological_index, event_id),
        )
        if onset_ids
        else None
    )
    evidence = tuple(dict.fromkeys((*first.evidence, *second.evidence)))
    objective_chains = tuple(dict.fromkeys((*first.objective_chains, *second.objective_chains)))
    affected_gaps = tuple(dict.fromkeys((*first.affected_gap_refs, *second.affected_gap_refs)))
    ambiguous_gaps = tuple(dict.fromkeys((*first.ambiguous_gap_refs, *second.ambiguous_gap_refs)))
    merged_key = "|".join(sorted((first.episode_id, second.episode_id)))
    return Episode(
        episode_id=_episode_id(merged_key),
        category=stronger.category,
        local_classification=stronger.local_classification,
        onset_event_id=onset,
        detection_event_id=detection,
        recovery_end_event_id=recovery,
        affected_event_ids=_dedupe_sorted_ids(
            (*first.affected_event_ids, *second.affected_event_ids), event_by_id
        ),
        evidence=evidence,
        objective_chains=objective_chains,
        context_window_event_ids=_dedupe_sorted_ids(
            (*first.context_window_event_ids, *second.context_window_event_ids), event_by_id
        ),
        affected_gap_refs=affected_gaps,
        ambiguous_gap_refs=ambiguous_gaps,
        retry_event_ids=_dedupe_sorted_ids(
            (*first.retry_event_ids, *second.retry_event_ids), event_by_id
        ),
        reverted_edit_event_ids=_dedupe_sorted_ids(
            (*first.reverted_edit_event_ids, *second.reverted_edit_event_ids), event_by_id
        ),
    )


def _merge_candidates(
    candidates: tuple[Episode, ...], event_by_id: dict[str, NormalizedEvent]
) -> tuple[Episode, ...]:
    merged: list[Episode] = []
    for candidate in sorted(
        candidates,
        key=lambda episode: (
            event_by_id[episode.detection_event_id].chronological_index,
            episode.detection_event_id,
        ),
    ):
        candidate_affected = set(candidate.affected_event_ids)
        matching_indices: list[int] = []
        for index, existing in enumerate(merged):
            shares_cause = (
                candidate.onset_event_id is not None
                and candidate.onset_event_id == existing.onset_event_id
            ) or bool(candidate_affected & set(existing.affected_event_ids))
            if not shares_cause or not events_share_lineage(
                event_by_id[candidate.detection_event_id],
                event_by_id[existing.detection_event_id],
            ):
                continue
            if not all(
                events_share_lineage(
                    event_by_id[existing.detection_event_id],
                    event_by_id[merged[matched_index].detection_event_id],
                )
                for matched_index in matching_indices
            ):
                continue
            matching_indices.append(index)
        if not matching_indices:
            merged.append(candidate)
            continue
        component = candidate
        for index in matching_indices:
            component = _merge_two_episodes(merged[index], component, event_by_id)
        matched = set(matching_indices)
        merged = [existing for index, existing in enumerate(merged) if index not in matched]
        merged.append(component)
    return tuple(
        sorted(
            merged,
            key=lambda episode: (
                event_by_id[episode.detection_event_id].chronological_index,
                episode.detection_event_id,
            ),
        )
    )


def detect_mistakes(
    parsed: ParsedSession,
    *,
    include_thinking: bool = True,
    policy: DetectionPolicy = DetectionPolicy(),
) -> DetectionResult:
    """Detect high-precision local mistake candidates without I/O or model calls."""

    ordered = tuple(
        sorted(
            (
                event
                for event in parsed.events
                if include_thinking or event.kind is not ContentKind.THINKING
            ),
            key=lambda event: (event.chronological_index, event.event_id),
        )
    )
    event_by_id = {event.event_id: event for event in ordered}
    event_position_by_id = {event.event_id: index for index, event in enumerate(ordered)}
    lineage_windows = _LineageWindowIndex(ordered)
    human_chronology = tuple(
        event.chronological_index for event in ordered if event.actor is EventActor.HUMAN
    )
    retained_heap: list[tuple[_CandidateRank, str, Episode]] = []
    eligible_candidates = 0

    def retain(candidate: Episode) -> None:
        nonlocal eligible_candidates
        eligible_candidates += 1
        rank = _candidate_rank(candidate, event_by_id)
        item = (rank, candidate.episode_id, candidate)
        if len(retained_heap) < policy.max_candidates:
            heapq.heappush(retained_heap, item)
        elif rank > retained_heap[0][0]:
            heapq.heapreplace(retained_heap, item)

    suppressed_non_mistakes = 0
    for index, event in enumerate(ordered):
        if _is_suppressed_admission_like(event):
            suppressed_non_mistakes += 1
            continue
        if (
            event.kind is ContentKind.VISIBLE_TEXT
            and event.source_kind is EvidenceSourceKind.USER_PROMPT
            and _REQUIREMENT_CHANGE_RE.search(_prose_for_signal_matching(event.text))
        ):
            suppressed_non_mistakes += 1
            continue
        if _is_expected_validation_failure(
            ordered,
            event_by_id,
            index,
            policy,
            lineage_windows,
        ):
            suppressed_non_mistakes += 1
            continue
        if _is_transient_unchanged_retry(
            ordered,
            event_by_id,
            index,
            policy,
            lineage_windows,
        ):
            suppressed_non_mistakes += 1
            continue
        objective = _complete_objective_chain(
            ordered,
            event_by_id,
            index,
            policy,
            lineage_windows,
        )
        if objective is not None:
            (
                objective_root,
                failed_use,
                correction_use,
                correction_result,
                retry_use,
                recovery,
            ) = objective
            affected_events = tuple(
                item
                for item in (
                    objective_root,
                    event_by_id.get(objective_root.correlated_event_id or ""),
                    failed_use,
                    event,
                    correction_use,
                    correction_result,
                    retry_use,
                    recovery,
                )
                if item is not None
            )
            correction_kind = (
                EvidenceKind.REVERT
                if _is_revert_use(correction_use)
                else EvidenceKind.CORRECTIVE_ACTION
            )
            chain_id = f"objective:{event.event_id}:{recovery.event_id}"
            retain(
                Episode(
                    episode_id=_episode_id(chain_id),
                    category="incorrect_change",
                    local_classification=Classification.CONFIRMED,
                    onset_event_id=objective_root.event_id,
                    detection_event_id=event.event_id,
                    recovery_end_event_id=recovery.event_id,
                    affected_event_ids=tuple(item.event_id for item in affected_events),
                    evidence=(
                        EvidenceRef(
                            objective_root.event_id,
                            objective_root.source_kind,
                            SignalKind.TOOL_FAILURE,
                            EvidenceKind.AFFECTED_WORK,
                            f"affected:{objective_root.event_id}",
                        ),
                        EvidenceRef(
                            event.event_id,
                            event.source_kind,
                            SignalKind.TOOL_FAILURE,
                            EvidenceKind.OBJECTIVE_CONTRADICTION,
                            f"objective:{event.event_id}",
                            True,
                        ),
                        EvidenceRef(
                            correction_use.event_id,
                            correction_use.source_kind,
                            (
                                SignalKind.REVERT
                                if correction_kind is EvidenceKind.REVERT
                                else SignalKind.AGENT_SELF_CORRECTION
                            ),
                            correction_kind,
                            f"correction:{correction_use.event_id}",
                            True,
                        ),
                        EvidenceRef(
                            recovery.event_id,
                            recovery.source_kind,
                            SignalKind.RECOVERY_VALIDATION,
                            EvidenceKind.SUCCESSFUL_RECOVERY,
                            f"recovery:{recovery.event_id}",
                            True,
                        ),
                    ),
                    objective_chains=(
                        ObjectiveChain(
                            chain_id=chain_id,
                            root_event_id=objective_root.event_id,
                            contradiction_event_id=event.event_id,
                            correction_event_ids=(
                                correction_use.event_id,
                                correction_result.event_id,
                            ),
                            recovery_event_id=recovery.event_id,
                        ),
                    ),
                    context_window_event_ids=(),
                    affected_gap_refs=tuple(
                        _gap_ref(human_chronology, first, second)
                        for first, second in zip(affected_events, affected_events[1:], strict=False)
                    ),
                    retry_event_ids=(retry_use.event_id,),
                    reverted_edit_event_ids=(
                        (correction_use.event_id,) if _is_revert_use(correction_use) else ()
                    ),
                )
            )
            continue
        retry = _material_retry(
            ordered,
            event_by_id,
            index,
            policy,
            lineage_windows,
        )
        if retry is not None:
            failed_use, retry_use, recovery = retry
            affected_events = (failed_use, event, retry_use, recovery)
            retain(
                Episode(
                    episode_id=_episode_id(f"retry:{event.event_id}:{retry_use.event_id}"),
                    category="invalid_command",
                    local_classification=Classification.UNCONFIRMED,
                    onset_event_id=failed_use.event_id,
                    detection_event_id=event.event_id,
                    recovery_end_event_id=recovery.event_id,
                    affected_event_ids=tuple(item.event_id for item in affected_events),
                    evidence=(
                        EvidenceRef(
                            event.event_id,
                            event.source_kind,
                            SignalKind.TOOL_FAILURE,
                            EvidenceKind.WEAK_FRICTION,
                            f"failure:{event.event_id}",
                        ),
                        EvidenceRef(
                            retry_use.event_id,
                            retry_use.source_kind,
                            SignalKind.MATERIAL_RETRY,
                            EvidenceKind.CORRECTIVE_ACTION,
                            f"retry:{retry_use.event_id}",
                        ),
                        EvidenceRef(
                            recovery.event_id,
                            recovery.source_kind,
                            SignalKind.RECOVERY_VALIDATION,
                            EvidenceKind.SUCCESSFUL_RECOVERY,
                            f"recovery:{recovery.event_id}",
                        ),
                    ),
                    context_window_event_ids=(),
                    affected_gap_refs=tuple(
                        _gap_ref(human_chronology, first, second)
                        for first, second in zip(affected_events, affected_events[1:], strict=False)
                    ),
                    retry_event_ids=(retry_use.event_id,),
                )
            )
            continue
        if (
            include_thinking
            and event.actor is EventActor.ASSISTANT
            and event.kind is ContentKind.THINKING
            and event.source_kind is EvidenceSourceKind.ASSISTANT_THINKING
            and _has_specific_visible_admission(event.text)
        ):
            thinking_root = _linked_affected_event(
                ordered,
                event_by_id,
                event_position_by_id,
                index,
                policy,
                lineage_windows,
            )
            correction = _thinking_correction(
                ordered,
                event_by_id,
                index,
                thinking_root,
                policy,
                lineage_windows,
            )
            classification = (
                Classification.PROBABLE
                if thinking_root is not None and correction is not None
                else Classification.UNCONFIRMED
            )
            evidence = [
                EvidenceRef(
                    event_id=event.event_id,
                    source_kind=event.source_kind,
                    signal_kind=SignalKind.AGENT_SELF_CORRECTION,
                    evidence_kind=EvidenceKind.THINKING_ADMISSION,
                    corroboration_group=f"thinking:{event.event_id}",
                    qualifies_for_promotion=True,
                )
            ]
            thinking_affected = (
                [event.event_id]
                if thinking_root is None
                else [thinking_root.event_id, event.event_id]
            )
            recovery_end: str | None = None
            if correction is not None:
                corrective_action, recovery = correction
                thinking_affected.extend((corrective_action.event_id, recovery.event_id))
                recovery_end = recovery.event_id
                evidence.extend(
                    (
                        EvidenceRef(
                            event_id=corrective_action.event_id,
                            source_kind=corrective_action.source_kind,
                            signal_kind=SignalKind.AGENT_SELF_CORRECTION,
                            evidence_kind=EvidenceKind.CORRECTIVE_ACTION,
                            corroboration_group=f"correction:{corrective_action.event_id}",
                            qualifies_for_promotion=True,
                        ),
                        EvidenceRef(
                            event_id=recovery.event_id,
                            source_kind=recovery.source_kind,
                            signal_kind=SignalKind.RECOVERY_VALIDATION,
                            evidence_kind=EvidenceKind.SUCCESSFUL_RECOVERY,
                            corroboration_group=f"recovery:{recovery.event_id}",
                            qualifies_for_promotion=True,
                        ),
                    )
                )
            affected_events = tuple(event_by_id[event_id] for event_id in thinking_affected)
            episode = Episode(
                episode_id=_episode_id(f"thinking-admission:{event.event_id}"),
                category="wrong_assumption",
                local_classification=classification,
                onset_event_id=(thinking_root.event_id if thinking_root is not None else None),
                detection_event_id=event.event_id,
                recovery_end_event_id=recovery_end,
                affected_event_ids=tuple(thinking_affected),
                evidence=tuple(evidence),
                context_window_event_ids=(),
                affected_gap_refs=tuple(
                    _gap_ref(human_chronology, first, second)
                    for first, second in zip(affected_events, affected_events[1:], strict=False)
                ),
            )
            retain(episode)
            continue
        if (
            event.kind is ContentKind.VISIBLE_TEXT
            and event.source_kind is EvidenceSourceKind.USER_PROMPT
            and _USER_CORRECTION_RE.search(_prose_for_signal_matching(event.text))
        ):
            user_root = _linked_affected_event(
                ordered,
                event_by_id,
                event_position_by_id,
                index,
                policy,
                lineage_windows,
            )
            candidate_id = f"user-correction:{event.event_id}"
            user_affected = () if user_root is None else (user_root.event_id,)
            retain(
                Episode(
                    episode_id=_episode_id(candidate_id),
                    category="user_correction",
                    local_classification=Classification.UNCONFIRMED,
                    onset_event_id=(user_root.event_id if user_root is not None else None),
                    detection_event_id=event.event_id,
                    recovery_end_event_id=None,
                    affected_event_ids=user_affected,
                    evidence=(
                        EvidenceRef(
                            event_id=event.event_id,
                            source_kind=event.source_kind,
                            signal_kind=SignalKind.USER_CORRECTION,
                            evidence_kind=EvidenceKind.USER_CORRECTION,
                            corroboration_group=f"user:{event.event_id}",
                        ),
                    ),
                    context_window_event_ids=(),
                    affected_gap_refs=(),
                )
            )
            continue
        if (
            event.actor is not EventActor.ASSISTANT
            or event.kind is not ContentKind.VISIBLE_TEXT
            or event.source_kind is not EvidenceSourceKind.ASSISTANT_TEXT
            or not _has_specific_visible_admission(event.text)
        ):
            continue
        admission_root = _linked_affected_event(
            ordered,
            event_by_id,
            event_position_by_id,
            index,
            policy,
            lineage_windows,
        )
        classification = (
            Classification.CONFIRMED if admission_root is not None else Classification.UNCONFIRMED
        )
        candidate_id = f"visible-admission:{event.event_id}"
        admission_affected = (
            (event.event_id,)
            if admission_root is None
            else (admission_root.event_id, event.event_id)
        )
        evidence_items = []
        if admission_root is not None:
            evidence_items.append(
                EvidenceRef(
                    event_id=admission_root.event_id,
                    source_kind=admission_root.source_kind,
                    signal_kind=SignalKind.AGENT_ADMISSION,
                    evidence_kind=EvidenceKind.AFFECTED_WORK,
                    corroboration_group=f"affected:{admission_root.event_id}",
                )
            )
        evidence_items.append(
            EvidenceRef(
                event_id=event.event_id,
                source_kind=event.source_kind,
                signal_kind=SignalKind.AGENT_ADMISSION,
                evidence_kind=EvidenceKind.VISIBLE_ADMISSION,
                corroboration_group=f"admission:{event.event_id}",
                qualifies_for_promotion=True,
            )
        )
        gap_refs = (
            () if admission_root is None else (_gap_ref(human_chronology, admission_root, event),)
        )
        episode = Episode(
            episode_id=_episode_id(candidate_id),
            category="wrong_assumption",
            local_classification=classification,
            onset_event_id=(admission_root.event_id if admission_root is not None else None),
            detection_event_id=event.event_id,
            recovery_end_event_id=None,
            affected_event_ids=admission_affected,
            evidence=tuple(evidence_items),
            context_window_event_ids=(),
            affected_gap_refs=gap_refs,
        )
        retain(episode)
    candidates = tuple(
        replace(
            item[2],
            context_window_event_ids=_context_window(
                ordered,
                event_position_by_id[item[2].detection_event_id],
                policy,
                lineage_windows,
            ),
        )
        for item in retained_heap
    )
    eligible = eligible_candidates
    retained_count = len(candidates)
    omitted = eligible - retained_count
    merged = _merge_candidates(candidates, event_by_id)
    episodes = [
        episode
        for episode in merged
        if episode.local_classification is not Classification.UNCONFIRMED
    ]
    unconfirmed = [
        episode for episode in merged if episode.local_classification is Classification.UNCONFIRMED
    ]
    diagnostics = DetectorDiagnostics(
        raw_signal_candidates=eligible + suppressed_non_mistakes,
        suppressed_non_mistakes=suppressed_non_mistakes,
        eligible_candidates=eligible,
        retained_candidates=retained_count,
        omitted_candidates=omitted,
    )
    return DetectionResult(
        episodes=tuple(episodes),
        unconfirmed_candidates=tuple(unconfirmed),
        eligible_candidates=eligible,
        retained_candidates=retained_count,
        omitted_candidates=omitted,
        diagnostics=diagnostics,
    )


__all__ = ["DetectionPolicy", "detect_mistakes"]
