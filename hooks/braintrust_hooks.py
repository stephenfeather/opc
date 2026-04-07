#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Braintrust tracing hooks - Cross-platform Python port.

Replaces the 5 bash hooks + common.sh:
- session_start
- session_end
- user_prompt_submit
- post_tool_use
- stop (creates LLM spans)

Usage:
    python3 braintrust_hooks.py <hook_name>

    # In settings.json:
    "command": "python3 $HOME/.claude/hooks/braintrust_hooks.py session_start"
"""

from __future__ import annotations

import json
import os
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

import faulthandler
faulthandler.enable(file=open(os.path.expanduser("~/.claude/logs/hooks_crash.log"), "a"), all_threads=True)

# Config from environment
STATE_DIR = Path.home() / ".claude" / "state" / "braintrust_sessions"
LOG_FILE = Path.home() / ".claude" / "state" / "braintrust_hook.log"
GLOBAL_STATE_FILE = Path.home() / ".claude" / "state" / "braintrust_global.json"

API_KEY = os.environ.get("BRAINTRUST_API_KEY", "")
PROJECT = os.environ.get("BRAINTRUST_CC_PROJECT", "claude-code")
API_URL = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev")
DEBUG = os.environ.get("BRAINTRUST_CC_DEBUG", "false").lower() == "true"
TRACE_ENABLED = os.environ.get("TRACE_TO_BRAINTRUST", "false").lower() == "true"


def ensure_dirs():
    """Ensure state directories exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(level: str, message: str):
    """Log to file."""
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} [{level}] {message}\n")


def debug(message: str):
    """Debug log if enabled."""
    if DEBUG:
        log("DEBUG", message)


def get_timestamp() -> str:
    """Get ISO timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_uuid() -> str:
    """Generate a UUID."""
    return str(uuid.uuid4())


def get_hostname() -> str:
    """Get hostname."""
    return socket.gethostname()


def get_username() -> str:
    """Get username."""
    return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


def get_os() -> str:
    """Get OS name."""
    return sys.platform


# State management
def load_session_state(session_id: str) -> dict:
    """Load session state from file."""
    ensure_dirs()
    state_file = STATE_DIR / f"{session_id}.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_session_state(session_id: str, state: dict):
    """Save session state atomically."""
    ensure_dirs()
    state_file = STATE_DIR / f"{session_id}.json"
    temp_file = state_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(state, indent=2))
    temp_file.rename(state_file)


def get_session_value(session_id: str, key: str) -> str | None:
    """Get a value from session state."""
    state = load_session_state(session_id)
    return state.get(key)


def set_session_value(session_id: str, key: str, value: Any):
    """Set a value in session state."""
    state = load_session_state(session_id)
    state[key] = value
    save_session_state(session_id, state)


def load_global_state() -> dict:
    """Load global state."""
    if GLOBAL_STATE_FILE.exists():
        try:
            return json.loads(GLOBAL_STATE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_global_state(state: dict):
    """Save global state."""
    ensure_dirs()
    GLOBAL_STATE_FILE.write_text(json.dumps(state, indent=2))


# Braintrust API
def get_project_id(project_name: str) -> str | None:
    """Get or cache project ID from Braintrust."""
    global_state = load_global_state()
    cached = global_state.get("project_ids", {}).get(project_name)
    if cached:
        return cached

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{API_URL}/v1/project",
                headers={"Authorization": f"Bearer {API_KEY}"},
                params={"project_name": project_name},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("objects"):
                    project_id = data["objects"][0]["id"]
                    global_state.setdefault("project_ids", {})[project_name] = project_id
                    save_global_state(global_state)
                    return project_id
    except Exception as e:
        log("ERROR", f"Failed to get project ID: {e}")
    return None


def insert_span(project_id: str, event: dict) -> str | None:
    """Insert a span into Braintrust."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{API_URL}/v1/project_logs/{project_id}/insert",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"events": [event]},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("row_ids", [None])[0]
    except Exception as e:
        log("ERROR", f"Failed to insert span: {e}")
    return None


# Hook implementations
def session_start(input_data: dict) -> dict:
    """Handle SessionStart hook."""
    debug("SessionStart hook triggered")

    if not TRACE_ENABLED:
        return {"result": "continue"}

    if not API_KEY:
        log("ERROR", "BRAINTRUST_API_KEY not set")
        return {"result": "continue"}

    session_id = input_data.get("session_id", generate_uuid())

    # Check if already has root span
    existing = get_session_value(session_id, "root_span_id")
    if existing:
        debug(f"Session already has root span: {existing}")
        return {"result": "continue"}

    project_id = get_project_id(PROJECT)
    if not project_id:
        log("ERROR", "Failed to get project")
        return {"result": "continue"}

    root_span_id = session_id
    timestamp = get_timestamp()
    workspace = input_data.get("cwd", "")
    workspace_name = Path(workspace).name if workspace else "Claude Code"

    event = {
        "id": root_span_id,
        "span_id": root_span_id,
        "root_span_id": root_span_id,
        "created": timestamp,
        "input": f"Session: {workspace_name}",
        "metadata": {
            "session_id": session_id,
            "workspace": workspace,
            "hostname": get_hostname(),
            "username": get_username(),
            "os": get_os(),
            "source": "claude-code",
        },
        "span_attributes": {
            "name": f"Claude Code: {workspace_name}",
            "type": "task",
        },
    }

    insert_span(project_id, event)

    set_session_value(session_id, "root_span_id", root_span_id)
    set_session_value(session_id, "project_id", project_id)
    set_session_value(session_id, "turn_count", 0)
    set_session_value(session_id, "started", timestamp)

    log("INFO", f"Created session root: {session_id} workspace={workspace_name}")
    return {"result": "continue"}


def compute_session_tags(session_id: str, input_data: dict | None = None) -> list[str]:
    """Compute tags for a session based on accumulated state."""
    tags: list[str] = []

    # --- Project tag ---
    state = load_session_state(session_id)
    workspace = state.get("workspace", "") or ""
    if not workspace and input_data:
        workspace = input_data.get("cwd", "") or ""
    if workspace:
        project_name = Path(workspace).name
        tags.append(f"project:{project_name}")

    # --- Scale tag ---
    # Python post_tool_use writes tool_counts dict; bash writes tool_count int
    tool_counts = state.get("tool_counts") or {}
    total_tools = sum(tool_counts.values()) if tool_counts else int(state.get("tool_count", 0) or 0)
    if total_tools < 20:
        tags.append("scale:quick")
    elif total_tools < 100:
        tags.append("scale:medium")
    else:
        tags.append("scale:large")

    # --- Agent usage tags ---
    agents_used = state.get("agents_used") or []
    if not agents_used:
        tags.append("agents:none")
    elif len(agents_used) <= 2:
        tags.append("agents:light")
    else:
        tags.append("agents:heavy")
    for agent_type in agents_used[:5]:
        tags.append(f"agent:{agent_type}")

    # --- Workflow tag (inferred from skills and tool mix) ---
    skills_used = state.get("skills_used") or []
    workflow_detected = False
    for skill in skills_used:
        tags.append(f"skill:{skill}")
        # Map skills to workflow types
        skill_workflow_map = {
            "fix": "workflow:fix",
            "debug": "workflow:debug",
            "build": "workflow:build",
            "tdd": "workflow:tdd",
            "test": "workflow:test",
            "refactor": "workflow:refactor",
            "explore": "workflow:explore",
            "research": "workflow:research",
            "review": "workflow:review",
            "commit": "workflow:commit",
            "release": "workflow:release",
            "migrate": "workflow:migrate",
        }
        if skill in skill_workflow_map:
            wf_tag = skill_workflow_map[skill]
            if wf_tag not in tags:
                tags.append(wf_tag)
                workflow_detected = True

    # Infer workflow from tool patterns if no skill detected
    if not workflow_detected:
        bash_count = tool_counts.get("Bash", 0)
        read_count = tool_counts.get("Read", 0)
        edit_count = tool_counts.get("Edit", 0)
        grep_count = tool_counts.get("Grep", 0)
        glob_count = tool_counts.get("Glob", 0)
        write_count = tool_counts.get("Write", 0)

        if edit_count + write_count > total_tools * 0.3 and total_tools > 10:
            tags.append("workflow:implement")
        elif grep_count + glob_count + read_count > total_tools * 0.6 and total_tools > 10:
            tags.append("workflow:explore")
        elif bash_count > total_tools * 0.5 and total_tools > 10:
            tags.append("workflow:shell-heavy")

    # --- Tool hygiene tag ---
    bash_count = tool_counts.get("Bash", 0)
    if total_tools > 10 and bash_count > 0:
        bash_ratio = bash_count / total_tools
        if bash_ratio < 0.15:
            tags.append("hygiene:clean")
        elif bash_ratio < 0.35:
            tags.append("hygiene:moderate")
        else:
            tags.append("hygiene:bash-heavy")

    # --- Outcome tags ---
    if state.get("has_pr"):
        tags.append("outcome:pr-created")
    elif state.get("has_push"):
        tags.append("outcome:pushed")
    elif state.get("has_commit"):
        tags.append("outcome:committed")
    else:
        tags.append("outcome:no-commit")

    turn_count = int(state.get("turn_count", 0) or 0)
    if turn_count <= 1 and total_tools < 5:
        tags.append("outcome:minimal")

    # --- Model tags ---
    models_used = state.get("models_used") or []
    for model in models_used:
        # Normalize model name: "claude-opus-4-6" -> "opus-4-6"
        short = model.replace("claude-", "") if model.startswith("claude-") else model
        tags.append(f"model:{short}")

    # --- Duration tag ---
    started = state.get("started", "")
    if started:
        try:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            duration_secs = (datetime.now(timezone.utc) - start_dt).total_seconds()
            if duration_secs < 120:
                tags.append("duration:short")
            elif duration_secs < 900:
                tags.append("duration:medium")
            else:
                tags.append("duration:long")
        except (ValueError, TypeError):
            pass

    # --- Token tags ---
    total_tokens = int(state.get("total_tokens", 0) or 0)
    if total_tokens > 0:
        if total_tokens < 5000:
            tags.append("tokens:low")
        elif total_tokens < 50000:
            tags.append("tokens:medium")
        else:
            tags.append("tokens:high")

    return tags


def session_end(input_data: dict) -> dict:
    """Handle SessionEnd hook - compute and apply session tags."""
    debug("SessionEnd hook triggered")

    if not TRACE_ENABLED:
        return {"result": "continue"}

    session_id = input_data.get("session_id", "")
    if not session_id:
        return {"result": "continue"}

    root_span_id = get_session_value(session_id, "root_span_id")
    project_id = get_session_value(session_id, "project_id")
    turn_count = get_session_value(session_id, "turn_count") or 0

    # Compute and apply tags
    if root_span_id and project_id:
        session_tags = compute_session_tags(session_id, input_data)

        if session_tags:
            tag_event = {
                "id": root_span_id,
                "_is_merge": True,
                "tags": session_tags,
                "metadata": {
                    "turn_count": turn_count,
                    "tool_counts": get_session_value(session_id, "tool_counts") or {},
                    "agents_used": get_session_value(session_id, "agents_used") or [],
                    "skills_used": get_session_value(session_id, "skills_used") or [],
                    "models_used": get_session_value(session_id, "models_used") or [],
                    "total_tokens": get_session_value(session_id, "total_tokens") or 0,
                },
            }
            insert_span(project_id, tag_event)
            log("INFO", f"Applied {len(session_tags)} tags: {session_tags}")

    log("INFO", f"Session ended: {session_id} (turns={turn_count})")
    return {"result": "continue"}


def user_prompt_submit(input_data: dict) -> dict:
    """Handle UserPromptSubmit hook."""
    debug("UserPromptSubmit hook triggered")

    if not TRACE_ENABLED:
        return {"result": "continue"}

    session_id = input_data.get("session_id", "")
    if not session_id:
        return {"result": "continue"}

    root_span_id = get_session_value(session_id, "root_span_id")
    project_id = get_session_value(session_id, "project_id")

    if not root_span_id or not project_id:
        # Create session root if missing
        session_start(input_data)
        root_span_id = get_session_value(session_id, "root_span_id")
        project_id = get_session_value(session_id, "project_id")
        if not root_span_id or not project_id:
            return {"result": "continue"}

    # Increment turn count
    turn_count = (get_session_value(session_id, "turn_count") or 0) + 1
    turn_span_id = generate_uuid()
    timestamp = get_timestamp()
    prompt = input_data.get("prompt", "")[:100]

    event = {
        "id": turn_span_id,
        "span_id": turn_span_id,
        "root_span_id": root_span_id,
        "span_parents": [root_span_id],
        "created": timestamp,
        "input": prompt,
        "span_attributes": {
            "name": f"Turn {turn_count}",
            "type": "task",
        },
    }

    insert_span(project_id, event)

    set_session_value(session_id, "turn_count", turn_count)
    set_session_value(session_id, "current_turn_span_id", turn_span_id)

    log("INFO", f"Turn {turn_count} started: {turn_span_id}")
    return {"result": "continue"}


def post_tool_use(input_data: dict) -> dict:
    """Handle PostToolUse hook."""
    debug("PostToolUse hook triggered")

    if not TRACE_ENABLED:
        return {"result": "continue"}

    session_id = input_data.get("session_id", "")
    tool_name = input_data.get("tool_name", "")

    if not session_id or not tool_name:
        return {"result": "continue"}

    turn_span_id = get_session_value(session_id, "current_turn_span_id")
    project_id = get_session_value(session_id, "project_id")
    root_span_id = get_session_value(session_id, "root_span_id")

    if not turn_span_id or not project_id:
        return {"result": "continue"}

    span_id = generate_uuid()
    timestamp = get_timestamp()
    tool_input = input_data.get("tool_input", {})
    tool_output = input_data.get("tool_response", input_data.get("output", {}))

    # Determine span name
    if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
        file_path = tool_input.get("file_path", tool_input.get("path", ""))
        span_name = f"{tool_name}: {Path(file_path).name}" if file_path else tool_name
    elif tool_name in ("Bash", "Terminal"):
        cmd = str(tool_input.get("command", ""))[:50]
        span_name = f"Terminal: {cmd}"
    else:
        span_name = tool_name

    event = {
        "id": span_id,
        "span_id": span_id,
        "root_span_id": root_span_id,
        "span_parents": [turn_span_id],
        "created": timestamp,
        "input": tool_input,
        "output": tool_output,
        "metadata": {"tool_name": tool_name},
        "span_attributes": {
            "name": span_name,
            "type": "tool",
        },
    }

    insert_span(project_id, event)

    # Accumulate tool counts for session-end tagging
    tool_counts = get_session_value(session_id, "tool_counts") or {}
    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    set_session_value(session_id, "tool_counts", tool_counts)

    # Track specific signals for tagging
    if tool_name == "Agent":
        agents_used = get_session_value(session_id, "agents_used") or []
        agent_type = tool_input.get("subagent_type", "unknown")
        if agent_type not in agents_used:
            agents_used.append(agent_type)
        set_session_value(session_id, "agents_used", agents_used)

    if tool_name == "Skill":
        skills_used = get_session_value(session_id, "skills_used") or []
        skill_name = tool_input.get("skill", "unknown")
        if skill_name not in skills_used:
            skills_used.append(skill_name)
        set_session_value(session_id, "skills_used", skills_used)

    # Track git commits
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        if cmd.startswith("git commit"):
            set_session_value(session_id, "has_commit", True)
        if "git push" in cmd:
            set_session_value(session_id, "has_push", True)
        if "gh pr create" in cmd:
            set_session_value(session_id, "has_pr", True)

    log("INFO", f"Tool: {span_name}")
    return {"result": "continue"}


def accumulate_tool_state(input_data: dict) -> dict:
    """Accumulate tool usage state for session-end tagging.

    Lightweight hook that only updates counters — no Braintrust spans created
    (the bash post_tool_use.sh handles span creation).
    """
    if not TRACE_ENABLED:
        return {"result": "continue"}

    session_id = input_data.get("session_id", "")
    tool_name = input_data.get("tool_name", "")
    if not session_id or not tool_name:
        return {"result": "continue"}

    tool_input = input_data.get("tool_input", {})

    # Accumulate tool counts for session-end tagging
    tool_counts = get_session_value(session_id, "tool_counts") or {}
    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    set_session_value(session_id, "tool_counts", tool_counts)

    # Track agent usage
    if tool_name == "Agent":
        agents_used = get_session_value(session_id, "agents_used") or []
        agent_type = tool_input.get("subagent_type", "unknown")
        if agent_type not in agents_used:
            agents_used.append(agent_type)
        set_session_value(session_id, "agents_used", agents_used)

    # Track skill usage
    if tool_name == "Skill":
        skills_used = get_session_value(session_id, "skills_used") or []
        skill_name = tool_input.get("skill", "unknown")
        if skill_name not in skills_used:
            skills_used.append(skill_name)
        set_session_value(session_id, "skills_used", skills_used)

    # Track git operations
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        if cmd.startswith("git commit"):
            set_session_value(session_id, "has_commit", True)
        if "git push" in cmd:
            set_session_value(session_id, "has_push", True)
        if "gh pr create" in cmd:
            set_session_value(session_id, "has_pr", True)

    return {"result": "continue"}


def stop(input_data: dict) -> dict:
    """Handle Stop hook - creates LLM spans for model calls within the turn.

    Parses the conversation JSONL to identify LLM calls (assistant messages
    after user/tool_result) and creates Braintrust spans for each.
    """
    log("INFO", "=== STOP HOOK CALLED ===")

    if not TRACE_ENABLED:
        return {"result": "continue"}

    # Get session ID
    session_id = input_data.get("session_id", "")
    if not session_id:
        transcript_path = input_data.get("transcript_path", "")
        if transcript_path:
            session_id = Path(transcript_path).stem

    if not session_id:
        debug("No session ID")
        return {"result": "continue"}

    # Get session state
    root_span_id = get_session_value(session_id, "root_span_id")
    project_id = get_session_value(session_id, "project_id")
    turn_span_id = get_session_value(session_id, "current_turn_span_id")

    if not turn_span_id or not project_id:
        log("WARN", f"No current turn to finalize (turn={turn_span_id}, project={project_id})")
        return {"result": "continue"}

    log("INFO", f"Stop hook processing turn: {turn_span_id} (session={session_id})")

    # Find conversation file
    conv_file = input_data.get("transcript_path", "")
    if not conv_file or not Path(conv_file).exists():
        sessions_dir = Path.home() / ".claude" / "projects"
        for jsonl in sessions_dir.rglob(f"{session_id}.jsonl"):
            conv_file = str(jsonl)
            break

    if not conv_file or not Path(conv_file).exists():
        debug("No conversation file")
        return {"result": "continue"}

    debug(f"Processing transcript: {conv_file}")

    # Get last processed line
    turn_last_line = get_session_value(session_id, "turn_last_line") or 0

    # Process transcript
    llm_calls_created = 0
    turn_models: set = set()
    turn_total_tokens = 0
    current_output_text = ""
    current_tool_calls: list = []
    current_model = ""
    current_prompt_tokens = 0
    current_completion_tokens = 0
    current_start_timestamp = ""
    current_end_timestamp = ""
    conversation_history: list = []
    line_num = 0

    def create_llm_span(output_text: str, model: str, prompt_tokens: int,
                        completion_tokens: int, start_ts: str, end_ts: str,
                        tool_calls: list, input_history: list) -> bool:
        """Create an LLM span in Braintrust."""
        nonlocal llm_calls_created, turn_total_tokens

        if not output_text and not tool_calls:
            return False

        span_id = generate_uuid()
        total_tokens = prompt_tokens + completion_tokens

        # Parse timestamps
        try:
            start_time = int(datetime.fromisoformat(start_ts.replace("Z", "+00:00")).timestamp()) if start_ts else int(datetime.now().timestamp())
            end_time = int(datetime.fromisoformat(end_ts.replace("Z", "+00:00")).timestamp()) if end_ts else int(datetime.now().timestamp())
        except (ValueError, AttributeError):
            start_time = end_time = int(datetime.now().timestamp())

        # Format output
        if tool_calls:
            output_json = {"role": "assistant", "content": output_text or "", "tool_calls": tool_calls}
        else:
            output_json = {"role": "assistant", "content": output_text}

        event = {
            "id": span_id,
            "span_id": span_id,
            "root_span_id": root_span_id,
            "span_parents": [turn_span_id],
            "created": start_ts or get_timestamp(),
            "input": input_history,
            "output": output_json,
            "metrics": {
                "start": start_time,
                "end": end_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens": total_tokens,
            },
            "metadata": {"model": model or "claude"},
            "span_attributes": {
                "name": model or "claude",
                "type": "llm",
            },
        }

        if insert_span(project_id, event):
            llm_calls_created += 1
            if model:
                turn_models.add(model)
            turn_total_tokens += total_tokens
            log("INFO", f"LLM span: {model} tokens={total_tokens} (turn={turn_span_id})")
            return True
        return False

    # Read and process JSONL
    with open(conv_file, "r") as f:
        for line in f:
            line_num += 1
            if line_num <= turn_last_line:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            msg_timestamp = msg.get("timestamp", "")

            if msg_type == "user":
                content = msg.get("message", {}).get("content", "")

                # Check if tool_result
                is_tool_result = False
                if isinstance(content, list) and content:
                    is_tool_result = content[0].get("type") == "tool_result"

                if is_tool_result:
                    # Save pending output first
                    if current_output_text or current_tool_calls:
                        create_llm_span(current_output_text, current_model,
                                       current_prompt_tokens, current_completion_tokens,
                                       current_start_timestamp, current_end_timestamp,
                                       current_tool_calls, conversation_history.copy())
                        conversation_history.append({
                            "role": "assistant",
                            "content": current_output_text,
                            "tool_calls": current_tool_calls if current_tool_calls else None,
                        })

                    # Add tool result to history
                    tool_content = content[0].get("content", "tool result")[:51200]
                    tool_id = content[0].get("tool_use_id", "")
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_content,
                    })

                    # Reset
                    current_output_text = ""
                    current_tool_calls = []
                    current_model = ""
                    current_prompt_tokens = 0
                    current_completion_tokens = 0
                    current_start_timestamp = ""
                    current_end_timestamp = ""
                else:
                    # Real user message
                    if current_output_text or current_tool_calls:
                        create_llm_span(current_output_text, current_model,
                                       current_prompt_tokens, current_completion_tokens,
                                       current_start_timestamp, current_end_timestamp,
                                       current_tool_calls, conversation_history.copy())
                        conversation_history.append({
                            "role": "assistant",
                            "content": current_output_text,
                            "tool_calls": current_tool_calls if current_tool_calls else None,
                        })

                    # Add user message
                    content_str = content if isinstance(content, str) else json.dumps(content)[:51200]
                    conversation_history.append({"role": "user", "content": content_str})

                    # Reset
                    current_output_text = ""
                    current_tool_calls = []
                    current_model = ""
                    current_prompt_tokens = 0
                    current_completion_tokens = 0
                    current_start_timestamp = msg_timestamp
                    current_end_timestamp = ""

            elif msg_type == "assistant":
                message = msg.get("message", {})
                content = message.get("content", "")

                # Extract text
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    text = "\n".join(texts)
                elif isinstance(content, str):
                    text = content
                else:
                    text = ""

                # Extract tool calls
                if isinstance(content, list):
                    tool_calls = [
                        {
                            "id": c.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": c.get("name", ""),
                                "arguments": json.dumps(c.get("input", {})),
                            },
                        }
                        for c in content if c.get("type") == "tool_use"
                    ]
                else:
                    tool_calls = []

                # Set start timestamp from first assistant message
                if not current_start_timestamp:
                    current_start_timestamp = msg_timestamp

                if text:
                    current_output_text = f"{current_output_text}\n{text}" if current_output_text else text
                    current_end_timestamp = msg_timestamp

                if tool_calls:
                    current_tool_calls = tool_calls
                    current_end_timestamp = msg_timestamp

                # Extract model
                model = message.get("model", "")
                if model:
                    current_model = model

                # Extract tokens
                usage = message.get("usage", {})
                if usage:
                    current_prompt_tokens += usage.get("input_tokens", 0) or 0
                    current_completion_tokens += usage.get("output_tokens", 0) or 0

    log("DEBUG", f"Finished processing transcript (lines={line_num}, llm_calls={llm_calls_created})")

    # Save final LLM call
    if current_output_text or current_tool_calls:
        log("DEBUG", "Saving final LLM call")
        create_llm_span(current_output_text, current_model,
                       current_prompt_tokens, current_completion_tokens,
                       current_start_timestamp, current_end_timestamp,
                       current_tool_calls, conversation_history.copy())

    # Update turn span with end time
    end_time = int(datetime.now().timestamp())
    turn_update = {
        "id": turn_span_id,
        "_is_merge": True,
        "metrics": {"end": end_time},
    }

    log("DEBUG", f"Attempting turn finalization: turn={turn_span_id} project={project_id}")
    insert_span(project_id, turn_update)

    # Accumulate model and token data for session-end tagging
    if turn_models:
        models_used = get_session_value(session_id, "models_used") or []
        for m in turn_models:
            if m not in models_used:
                models_used.append(m)
        set_session_value(session_id, "models_used", models_used)
    session_tokens = get_session_value(session_id, "total_tokens") or 0
    set_session_value(session_id, "total_tokens", session_tokens + turn_total_tokens)

    # Update state
    set_session_value(session_id, "turn_last_line", line_num)
    set_session_value(session_id, "current_turn_span_id", "")

    if llm_calls_created > 0:
        log("INFO", f"Created {llm_calls_created} LLM spans for turn")
    log("INFO", f"Turn finalized (end={end_time})")

    return {"result": "continue"}


HOOKS = {
    "session_start": session_start,
    "session_end": session_end,
    "user_prompt_submit": user_prompt_submit,
    "post_tool_use": post_tool_use,
    "accumulate_tool_state": accumulate_tool_state,
    "stop": stop,
}


def main():
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: braintrust_hooks.py <hook_name>", file=sys.stderr)
        sys.exit(1)

    hook_name = sys.argv[1]
    if hook_name not in HOOKS:
        print(f"Unknown hook: {hook_name}", file=sys.stderr)
        sys.exit(1)

    # Load .env if exists
    env_file = Path.home() / ".claude" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()

    # Reload config after .env
    global API_KEY, PROJECT, API_URL, DEBUG, TRACE_ENABLED
    API_KEY = os.environ.get("BRAINTRUST_API_KEY", "")
    PROJECT = os.environ.get("BRAINTRUST_CC_PROJECT", "claude-code")
    API_URL = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev")
    DEBUG = os.environ.get("BRAINTRUST_CC_DEBUG", "false").lower() == "true"
    TRACE_ENABLED = os.environ.get("TRACE_TO_BRAINTRUST", "false").lower() == "true"

    try:
        stdin_data = sys.stdin.read()
        input_data = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        input_data = {}

    result = HOOKS[hook_name](input_data)
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
