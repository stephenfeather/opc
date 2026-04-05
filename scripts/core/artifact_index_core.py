"""Pure functions for artifact indexing.

This module contains all pure (no I/O, no side effects) functions extracted
from artifact_index.py. Each function takes data in and returns data out —
no file reads, no database calls, no environment access.
"""

import re

# =============================================================================
# OUTCOME NORMALIZATION
# =============================================================================

OUTCOME_MAP = {
    "SUCCESS": "SUCCEEDED",
    "SUCCEEDED": "SUCCEEDED",
    "PARTIAL": "PARTIAL_PLUS",
    "PARTIAL_PLUS": "PARTIAL_PLUS",
    "PARTIAL_MINUS": "PARTIAL_MINUS",
    "FAILED": "FAILED",
    "FAILURE": "FAILED",
    "UNKNOWN": "UNKNOWN",
}


def normalize_outcome(status: str) -> str:
    """Normalize status string to canonical outcome value.

    Uses dispatch table for O(1) lookup instead of if/elif chains.
    """
    return OUTCOME_MAP.get(status.upper(), "UNKNOWN")


# =============================================================================
# FRONTMATTER & SECTIONS
# =============================================================================


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown content.

    Returns:
        Tuple of (frontmatter_dict, remaining_content)
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter = {}
    for line in parts[1].strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter, parts[2]


def extract_sections(content: str, level: int = 2) -> dict:
    """Extract markdown sections at the specified heading level.

    Args:
        content: Markdown content to parse
        level: Heading level (2 for ##, 3 for ###)

    Returns:
        Dict mapping normalized section names to content
    """
    if not content:
        return {}

    prefix = "#" * level + " "
    next_level_prefix = "#" * (level - 1) + " " if level > 1 else None

    sections = {}
    current_section = None
    current_content = []

    for line in content.split("\n"):
        if line.startswith(prefix):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = line[len(prefix):].strip().lower().replace(" ", "_")
            current_content = []
        elif next_level_prefix and line.startswith(next_level_prefix):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = None
            current_content = []
        elif current_section:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


# =============================================================================
# SESSION & FILE EXTRACTION
# =============================================================================


def extract_session_info(file_path) -> tuple[str, str | None]:
    """Extract session name and optional UUID from handoff file path.

    Supports paths like:
    - thoughts/shared/handoffs/my-session/task-01.md
    - thoughts/shared/handoffs/my-session-550e8400/task-01.md (with UUID suffix)

    Returns:
        Tuple of (session_name, session_uuid or None)
    """
    parts = file_path.parts

    if "handoffs" not in parts:
        return "", None

    idx = parts.index("handoffs")
    if idx + 1 >= len(parts):
        return "", None

    raw_name = parts[idx + 1]

    uuid_match = re.match(r"^(.+)-([0-9a-f]{8})$", raw_name, re.IGNORECASE)
    if uuid_match:
        return uuid_match.group(1), uuid_match.group(2).lower()

    return raw_name, None


def extract_files(content: str) -> list:
    """Extract file paths from markdown content."""
    files = []
    for line in content.split("\n"):
        matches = re.findall(r"`([^`]+\.[a-z]+)(:[^`]*)?`", line)
        files.extend([m[0] for m in matches])
        matches = re.findall(r"\*\*File\*\*:\s*`?([^\s`]+)`?", line)
        files.extend(matches)
    return files


# =============================================================================
# SIMPLE YAML PARSER
# =============================================================================


def _parse_inline_list(value: str) -> list[str]:
    """Parse an inline YAML list like '[a, b, "c"]' into a Python list."""
    return [
        x.strip().strip('"').strip("'")
        for x in value[1:-1].split(",")
        if x.strip()
    ]


def _parse_yaml_list_item(stripped: str, current_list: list | None) -> list:
    """Parse a YAML list item ('- ...') and append to current_list.

    Handles both plain string items and dict-style items ('- key: value').
    Returns the updated list.
    """
    if current_list is None:
        current_list = []

    item = stripped[2:].strip()

    # Dict-style list item: "- key: value"
    if ": " in item and not item.startswith('"'):
        k, v = item.split(": ", 1)
        k = k.strip()
        v = v.strip().strip('"')
        if (
            current_list
            and isinstance(current_list[-1], dict)
            and k not in current_list[-1]
        ):
            current_list[-1][k] = v
        else:
            current_list.append({k: v})
    else:
        current_list.append(item.strip('"').strip("'"))

    return current_list


def _parse_yaml_indented_kv(stripped: str, current_list: list) -> None:
    """Parse an indented key-value pair under a dict list item.

    Mutates the last dict in current_list in place.
    """
    if ": " not in stripped:
        return
    k, v = stripped.split(": ", 1)
    k = k.strip()
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        v = _parse_inline_list(v)
    else:
        v = v.strip('"').strip("'")
    current_list[-1][k] = v


def parse_simple_yaml(text: str) -> dict:
    """Parse simple YAML without pyyaml dependency.

    Handles the flat key-value and list structures used in handoff YAML files.
    Does NOT handle arbitrary nested YAML - only the handoff format.
    """
    result = {}
    current_key = None
    current_list = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item under a key
        if stripped.startswith("- ") and current_key is not None:
            current_list = _parse_yaml_list_item(stripped, current_list)
            result[current_key] = current_list
            continue

        # Indented key-value under a dict list item
        if (
            line.startswith("    ")
            and current_key
            and current_list
            and isinstance(current_list[-1], dict)
        ):
            _parse_yaml_indented_kv(stripped, current_list)
            continue

        # Top-level key
        if ":" in line and not line.startswith(" "):
            if current_key and current_list is not None:
                result[current_key] = current_list

            key, value = line.split(":", 1)
            current_key = key.strip()
            value = value.strip()
            current_list = None

            if value == "" or value == "[]":
                current_list = []
                result[current_key] = current_list
            elif value.startswith("[") and value.endswith("]"):
                result[current_key] = _parse_inline_list(value)
                current_list = None
                current_key = None
            else:
                result[current_key] = value.strip('"').strip("'")
                current_list = None

    return result


# =============================================================================
# SQL ADAPTATION (PostgreSQL compatibility)
# =============================================================================


def convert_pg_upsert(sql: str) -> str:
    """Convert SQLite INSERT OR REPLACE to PostgreSQL ON CONFLICT."""
    pattern = r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*\(([^)]+)\)"
    match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
    if not match:
        return sql

    columns = [c.strip() for c in match.group(2).split(",")]
    non_pk_cols = [c for c in columns if c != "id"]
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk_cols)

    sql = re.sub(r"INSERT\s+OR\s+REPLACE\s+INTO", "INSERT INTO", sql, flags=re.IGNORECASE)
    sql = sql.rstrip().rstrip(";")
    sql += f" ON CONFLICT (id) DO UPDATE SET {update_clause}"
    return sql


def adapt_for_postgres(sql: str, params: tuple, table_hint: str) -> tuple:
    """Adapt SQL and params for PostgreSQL's existing schema."""
    sql = sql.replace("?", "%s")

    if "INTO handoffs" in sql or table_hint == "handoffs":
        sql = """
            INSERT INTO handoffs
            (id, session_name, file_path, goal, what_worked, what_failed,
             key_decisions, outcome, root_span_id, session_id)
            VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path) DO UPDATE SET
                goal = EXCLUDED.goal,
                what_worked = EXCLUDED.what_worked,
                what_failed = EXCLUDED.what_failed,
                key_decisions = EXCLUDED.key_decisions,
                outcome = EXCLUDED.outcome,
                root_span_id = EXCLUDED.root_span_id,
                session_id = EXCLUDED.session_id,
                indexed_at = NOW()
        """
        if len(params) == 15:
            params = (
                params[1],   # session_name
                params[3],   # file_path
                params[4],   # task_summary -> goal
                params[5],   # what_worked
                params[6],   # what_failed
                params[7],   # key_decisions
                params[9],   # outcome
                params[10],  # root_span_id
                params[12],  # session_id
            )
        return sql, params

    if "INSERT OR REPLACE INTO" in sql:
        sql = convert_pg_upsert(sql)

    return sql, params


# =============================================================================
# FILE ID & CLASSIFICATION
# =============================================================================


def generate_file_id(file_path: str) -> str:
    """Generate a deterministic 12-char hex ID from a file path."""
    import hashlib

    return hashlib.md5(str(file_path).encode()).hexdigest()[:12]


def classify_file(file_path) -> str | None:
    """Classify a file path into an artifact type for indexing.

    Returns:
        "handoff", "handoff_yaml", "plan", "continuity", or None
    """
    path_str = str(file_path)
    suffix = file_path.suffix if hasattr(file_path, "suffix") else ""
    name = file_path.name if hasattr(file_path, "name") else ""

    if "handoffs" in path_str:
        if suffix in (".yaml", ".yml"):
            return "handoff_yaml"
        if suffix == ".md":
            return "handoff"
    if "plans" in path_str and suffix == ".md":
        return "plan"
    if name.startswith("CONTINUITY_CLAUDE-"):
        return "continuity"
    return None


# =============================================================================
# CONTENT PARSERS (pure — take content strings, not file paths)
# =============================================================================


def parse_handoff_content(raw_content: str, file_path) -> dict:
    """Parse handoff markdown content into structured data.

    Pure function — takes already-read content, no I/O.
    """
    import json

    frontmatter, content = parse_frontmatter(raw_content)

    sections = extract_sections(content, level=2)
    subsections = extract_sections(content, level=3)
    sections.update(subsections)

    file_id = generate_file_id(str(file_path))
    session_name, session_uuid = extract_session_info(file_path)

    task_match = re.match(r".*task-(\d+)", str(file_path.stem))
    task_number = int(task_match.group(1)) if task_match else None

    status = frontmatter.get("status", "UNKNOWN")
    outcome = normalize_outcome(status)

    return {
        "id": file_id,
        "session_name": session_name,
        "session_uuid": session_uuid,
        "task_number": task_number,
        "file_path": str(file_path),
        "task_summary": sections.get(
            "what_was_done", sections.get("summary", "")
        )[:500],
        "what_worked": sections.get("what_worked", ""),
        "what_failed": sections.get("what_failed", ""),
        "key_decisions": sections.get(
            "key_decisions", sections.get("decisions", "")
        ),
        "files_modified": json.dumps(
            extract_files(sections.get("files_modified", ""))
        ),
        "outcome": outcome,
        "root_span_id": frontmatter.get("root_span_id", ""),
        "turn_span_id": frontmatter.get("turn_span_id", ""),
        "session_id": frontmatter.get("session_id", ""),
        "braintrust_session_id": frontmatter.get("braintrust_session_id", ""),
        "created_at": frontmatter.get("date", ""),
    }


def parse_handoff_yaml_content(raw_content: str, file_path) -> dict:
    """Parse handoff YAML content into structured data.

    Pure function — takes already-read content, no I/O.
    """
    import json

    frontmatter, body = parse_frontmatter(raw_content)
    data = parse_simple_yaml(body)

    file_id = generate_file_id(str(file_path))

    session_name = frontmatter.get("session", "")
    if not session_name:
        session_name, _ = extract_session_info(file_path)

    # Build task summary from done_this_session
    done_items = data.get("done_this_session", [])
    if isinstance(done_items, list):
        task_lines = []
        for item in done_items:
            if isinstance(item, dict):
                task_lines.append(item.get("task", ""))
            elif isinstance(item, str):
                task_lines.append(item)
        task_summary = "; ".join(t for t in task_lines if t)[:500]
    else:
        task_summary = str(done_items)[:500]

    # Extract what_worked
    worked = data.get("worked", [])
    if isinstance(worked, list):
        what_worked = "\n".join(
            f"- {w}" if isinstance(w, str) else f"- {w}" for w in worked
        )
    else:
        what_worked = str(worked)

    # Extract what_failed
    failed = data.get("failed", [])
    if isinstance(failed, list):
        what_failed = "\n".join(
            f"- {f}" if isinstance(f, str) else f"- {f}" for f in failed
        )
    else:
        what_failed = str(failed)

    # Extract decisions
    decisions = data.get("decisions", [])
    if isinstance(decisions, list):
        decision_lines = []
        for d in decisions:
            if isinstance(d, dict):
                for k, v in d.items():
                    decision_lines.append(f"- {k}: {v}")
            elif isinstance(d, str):
                decision_lines.append(f"- {d}")
        key_decisions = "\n".join(decision_lines)
    else:
        key_decisions = str(decisions)

    # Extract files modified
    files_section = data.get("files", {})
    all_files = []
    if isinstance(files_section, dict):
        for file_list in files_section.values():
            if isinstance(file_list, list):
                all_files.extend(file_list)
    elif isinstance(files_section, list):
        all_files = files_section

    status = frontmatter.get("status", data.get("outcome", "UNKNOWN"))
    outcome = normalize_outcome(status)

    return {
        "id": file_id,
        "session_name": session_name,
        "session_uuid": None,
        "task_number": None,
        "file_path": str(file_path),
        "task_summary": task_summary,
        "what_worked": what_worked,
        "what_failed": what_failed,
        "key_decisions": key_decisions,
        "files_modified": json.dumps(all_files),
        "outcome": outcome,
        "root_span_id": frontmatter.get("root_span_id", ""),
        "turn_span_id": frontmatter.get("turn_span_id", ""),
        "session_id": frontmatter.get("session_id", ""),
        "braintrust_session_id": frontmatter.get("braintrust_session_id", ""),
        "created_at": frontmatter.get("date", ""),
    }


def parse_plan_content(content: str, file_path) -> dict:
    """Parse plan markdown content into structured data.

    Pure function — uses extract_sections() instead of duplicating the loop.
    """
    import json

    file_id = generate_file_id(str(file_path))

    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else file_path.stem

    sections = extract_sections(content, level=2)

    phases = []
    for key in sections:
        if re.match(r"^phase[_\-]?\d+", key):
            phases.append({"name": key, "content": sections[key][:500]})

    return {
        "id": file_id,
        "title": title,
        "file_path": str(file_path),
        "overview": sections.get("overview", "")[:1000],
        "approach": sections.get(
            "implementation_approach", sections.get("approach", "")
        )[:1000],
        "phases": json.dumps(phases),
        "constraints": sections.get(
            "what_we're_not_doing", sections.get("constraints", "")
        ),
    }


def parse_continuity_content(content: str, file_path) -> dict:
    """Parse continuity ledger content into structured data.

    Pure function — uses extract_sections() instead of duplicating the loop.
    """
    import json

    file_id = generate_file_id(str(file_path))

    session_match = re.search(r"CONTINUITY_CLAUDE-(.+)\.md", str(file_path.name))
    session_name = session_match.group(1) if session_match else file_path.stem

    sections = extract_sections(content, level=2)

    # Parse state section
    state = sections.get("state", "")
    state_done = []
    state_now = ""
    state_next = ""

    for line in state.split("\n"):
        if "[x]" in line.lower():
            state_done.append(line.strip())
        elif "[->]" in line or "now:" in line.lower():
            state_now = line.strip()
        elif "[ ]" in line or "next:" in line.lower():
            state_next = line.strip()

    return {
        "id": file_id,
        "session_name": session_name,
        "goal": sections.get("goal", "")[:500],
        "state_done": json.dumps(state_done),
        "state_now": state_now,
        "state_next": state_next,
        "key_learnings": sections.get(
            "key_learnings", sections.get("key_learnings_(this_session)", "")
        ),
        "key_decisions": sections.get("key_decisions", ""),
        "snapshot_reason": "manual",
    }
