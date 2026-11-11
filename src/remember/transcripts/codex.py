"""Codex sessions: ~/.codex/sessions/YYYY/MM/DD/rollout-<time>-<id>.jsonl.

CODEX_HOME moves ~/.codex. Archived sessions in archived_sessions are included.
Each line is {"timestamp", "type", "payload"}. The conversation is in the
"response_item" lines. The first line, "session_meta", holds the system
prompt ("base_instructions"), and "compacted" lines hold compaction
summaries. "event_msg" lines repeat the conversation for the user interface,
and "turn_context" lines repeat settings and instructions, so both are left
out. Response items have no stable ID, so a message's ID is its line number,
which does not change because rollouts are only appended to.
"""

import os
import re
from collections.abc import Collection, Generator
from pathlib import Path

from .base import (
    ASSISTANT,
    SYSTEM,
    TOOL_RESULT,
    USER,
    Message,
    OnInvalid,
    Part,
    SessionInfo,
    ignore_invalid,
    json_or_text,
    modified_time,
    parse_timestamp,
    read_records,
    select_messages,
)

LABEL = "Codex"
CODEX_HOME = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
SESSION_FOLDERS = ("sessions", "archived_sessions")
ROLLOUT_NAME = re.compile(r"rollout-\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d-(.+)\.jsonl")
# Codex adds context to the conversation as user messages that start with one
# of these. Codex itself does not show such messages as the user's.
CONTEXT_PREFIXES = (
    "# agents.md instructions",
    "<environment_context>",
    "<user_instructions>",
    "<user_shell_command>",
    "<turn_aborted>",
    "<subagent_notification>",
    "<skill>",
)
TOOL_CALLS = {
    "function_call": "arguments",
    "custom_tool_call": "input",
    "local_shell_call": "action",
    "web_search_call": "action",
    "tool_search_call": "arguments",
}
TOOL_OUTPUTS = ("function_call_output", "custom_tool_call_output")
FIXED_TOOL_NAMES = {
    "local_shell_call": "local_shell",
    "web_search_call": "web_search",
    "tool_search_call": "tool_search",
}


def get_session_id(path: Path) -> str:
    """Return the thread ID from a rollout-<time>-<id>.jsonl filename."""
    match = ROLLOUT_NAME.fullmatch(path.name)
    return match.group(1) if match else path.stem


def current_session_id() -> str | None:
    """Codex 0.95 and later tell the shell commands they run which thread runs them."""
    return os.environ.get("CODEX_THREAD_ID") or None


def list_session_ids() -> list[str]:
    return [get_session_id(path) for path in _session_paths()]


def get_session(session_id: str) -> SessionInfo:
    for path in _session_paths():
        if get_session_id(path) == session_id:
            return _load_session(path)
    raise FileNotFoundError(f"no Codex session {session_id}")


def find_sessions(project: Path | None = None) -> list[SessionInfo]:
    sessions = []
    for path in _session_paths():
        try:
            session = _load_session(path)
        except OSError:
            continue  # Removed after it was listed.
        if project is None or session.cwd == str(project):
            sessions.append(session)
    return sessions


def read_messages(
    session: SessionInfo,
    *,
    roles: Collection[str] | None = None,
    head: int | None = None,
    on_invalid: OnInvalid | None = None,
) -> list[Message]:
    return select_messages(_messages(session.path, on_invalid), roles, head)


def _session_paths() -> list[Path]:
    return [
        path
        for folder in SESSION_FOLDERS
        if (CODEX_HOME / folder).is_dir()
        for path in (CODEX_HOME / folder).rglob("rollout-*.jsonl")
        if path.is_file()
    ]


def _load_session(path: Path) -> SessionInfo:
    end = modified_time(path)
    meta: dict = {}
    with path.open(encoding="utf-8") as source:
        for _, record in read_records(source, path, ignore_invalid):
            if record.get("type") == "session_meta":
                payload = record.get("payload")
                meta = payload if isinstance(payload, dict) else {}
                meta.setdefault("timestamp", record.get("timestamp"))
            break  # The session's metadata is its first line.
    return SessionInfo(
        id=get_session_id(path),
        path=path,
        cwd=str(meta.get("cwd", "")),
        start=parse_timestamp(meta.get("timestamp")) or end,
        end=end,
    )


def _messages(path: Path, on_invalid: OnInvalid | None) -> Generator[Message]:
    tool_names: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, record in read_records(source, path, on_invalid):
            item = record.get("payload")
            if not isinstance(item, dict):
                continue
            match record.get("type"):
                case "response_item":
                    message = _message(item, tool_names)
                case "session_meta":
                    message = _context(item.get("base_instructions"), "system prompt")
                case "compacted":
                    message = _context(item.get("message"), "compaction summary")
                case _:
                    message = None
            if message is not None:
                role, parts = message
                yield Message(
                    id=str(line_number),
                    role=role,
                    parts=parts,
                    time=parse_timestamp(record.get("timestamp")),
                )


def _message(
    item: dict, tool_names: dict[str, str]
) -> tuple[str, tuple[Part, ...]] | None:
    item_type = item.get("type")
    if item_type == "message":
        parts = _content_parts(item.get("content"))
        match item.get("role"):
            case "assistant":
                return ASSISTANT, parts
            case "user" if not _is_context(parts):
                return USER, parts
            case _:  # Context, and developer or system instructions.
                return SYSTEM, parts
    if item_type == "reasoning":
        texts = [
            str(entry.get("text", ""))
            for key in ("summary", "content")
            for entry in item.get(key) or []
            if isinstance(entry, dict)
        ]
        # Reasoning that is only saved encrypted still shows that it happened.
        text = "\n\n".join(text for text in texts if text.strip())
        return ASSISTANT, (Part("thinking", text),)
    if isinstance(item_type, str) and item_type in TOOL_CALLS:
        name = FIXED_TOOL_NAMES.get(item_type) or str(item.get("name"))
        tool_names[str(item.get("call_id"))] = name
        arguments = json_or_text(item.get(TOOL_CALLS[item_type]))
        return ASSISTANT, (Part("tool_call", name=name, arguments=arguments),)
    if item_type in TOOL_OUTPUTS:
        name = item.get("name") or tool_names.get(str(item.get("call_id")), "unknown")
        parts, is_error = _output_parts(item.get("output"))
        header = Part("tool_result", name=str(name), is_error=is_error)
        return TOOL_RESULT, (header, *parts)
    return None


def _context(text: object, name: str) -> tuple[str, tuple[Part, ...]] | None:
    # Base instructions are saved as {"text": ...}; older rollouts use a string.
    if isinstance(text, dict):
        text = text.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return SYSTEM, (Part("context", text, name=name),)


def _is_context(parts: tuple[Part, ...]) -> bool:
    return any(
        part.kind == "text"
        and part.text.lstrip().casefold().startswith(CONTEXT_PREFIXES)
        for part in parts
    )


def _content_parts(content: object) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (Part("text", content),)
    parts = []
    for item in content if isinstance(content, list) else []:
        if not isinstance(item, dict):
            continue
        match item.get("type"):
            case "input_text" | "output_text" | "text":
                parts.append(Part("text", str(item.get("text", ""))))
            case "input_image" | "output_image" | "image":
                parts.append(Part("image"))
            case item_type:
                parts.append(Part(str(item_type)))
    return tuple(parts)


def _output_parts(output: object) -> tuple[tuple[Part, ...], bool]:
    """Tool output parts, and whether the tool failed.

    Output is text or a list of content items. Older rollouts wrap it in an
    object with "content" and "success".
    """
    if isinstance(output, dict):
        parts, _ = _output_parts(output.get("content"))
        return parts, output.get("success") is False
    return _content_parts(output), False
