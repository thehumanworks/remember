"""Claude Code sessions: ~/.claude/projects/<project>/<id>.jsonl.

CLAUDE_CONFIG_DIR moves ~/.claude. Claude Code writes one record per content
block, so one assistant reply can span several records that share a message
ID; consecutive records of one reply are joined into one message.

Besides user and assistant records, "attachment" records hold context that
Claude Code adds: the system prompt ("prompt_snapshot"), files, reminders and
hook messages. A prompt that the user queued while the agent worked is only
saved as an attachment, so it becomes a user message. "system" records hold
notices such as command output. Attachments and records without text, such
as settings, token counts and API request copies, are left out. Subagent
records ("isSidechain") are left out too: they are not part of the main
conversation, and no saved session here has them to test against.
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
    modified_time,
    parse_timestamp,
    read_records,
    select_messages,
)

LABEL = "Claude Code"
CLAUDE_PROJECTS_DIR = (
    Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
)
# Records to read for the working directory and start time before giving up.
HEADER_RECORDS = 200
# Fields of an attachment that hold text for the conversation, in order.
ATTACHMENT_TEXT_FIELDS = (
    "filename",
    "text",
    "content",
    "snippet",
    "condition",
    "addedBlocks",
    "addedLines",
    "files",
)
PROMPT_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"


def get_project_from_path(path: str) -> str:
    """Claude Code names a project's directory after its path, with every
    character other than an ASCII letter or digit replaced by "-"."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def current_session_id() -> str | None:
    """Claude Code tells the commands it runs which session runs them."""
    return os.environ.get("CLAUDE_CODE_SESSION_ID") or None


def list_session_ids() -> list[str]:
    return [path.stem for path in _session_paths()]


def get_session(session_id: str) -> SessionInfo:
    for path in _session_paths():
        if path.stem == session_id:
            return _load_session(path)
    raise FileNotFoundError(f"no Claude Code session {session_id}")


def find_sessions(project: Path | None = None) -> list[SessionInfo]:
    # Directory names are lossy, so the recorded cwd decides.
    project_dir = None if project is None else get_project_from_path(str(project))
    sessions = []
    for path in _session_paths(project_dir):
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


def _session_paths(project_dir: str | None = None) -> list[Path]:
    if not CLAUDE_PROJECTS_DIR.is_dir():
        return []
    if project_dir is None:
        directories = [d for d in CLAUDE_PROJECTS_DIR.iterdir() if d.is_dir()]
    else:
        directories = [CLAUDE_PROJECTS_DIR / project_dir]
    return [
        path
        for directory in directories
        if directory.is_dir()
        for path in directory.iterdir()
        if path.suffix == ".jsonl" and path.is_file()
    ]


def _load_session(path: Path) -> SessionInfo:
    end = modified_time(path)
    cwd, start = "", None
    with path.open(encoding="utf-8") as source:
        records = read_records(source, path, ignore_invalid)
        for count, (_, record) in enumerate(records, start=1):
            if not cwd and isinstance(record.get("cwd"), str):
                cwd = record["cwd"]
            start = start or parse_timestamp(record.get("timestamp"))
            if (cwd and start) or count >= HEADER_RECORDS:
                break
    return SessionInfo(id=path.stem, path=path, cwd=cwd, start=start or end, end=end)


def _messages(path: Path, on_invalid: OnInvalid | None) -> Generator[Message]:
    tool_names: dict[str, str] = {}
    pending: Message | None = None
    pending_reply = None
    # Context that arrives while a reply is written waits until the reply ends.
    held: list[Message] = []
    with path.open(encoding="utf-8") as source:
        for line_number, record in read_records(source, path, on_invalid):
            if record.get("isSidechain"):
                continue
            message = record.get("message")
            if record.get("type") not in (USER, ASSISTANT) or not isinstance(
                message, dict
            ):
                added = _added_message(record, line_number, tool_names)
                if added is not None and pending_reply is not None:
                    held.append(added)
                elif added is not None:
                    if pending is not None:
                        yield pending
                        pending = None
                    yield added
                continue
            parts = _parts(message.get("content"), tool_names)
            reply = message.get("id") if record["type"] == ASSISTANT else None
            if pending is not None and reply is not None and reply == pending_reply:
                pending = Message(
                    pending.id, pending.role, pending.parts + parts, pending.time
                )
                continue
            if pending is not None:
                yield pending
            yield from held
            held = []
            pending = Message(
                id=_record_id(record, line_number),
                role=_role(record, parts),
                parts=parts,
                time=parse_timestamp(record.get("timestamp")),
            )
            pending_reply = reply
    if pending is not None:
        yield pending
    yield from held


def _record_id(record: dict, line_number: int) -> str:
    return str(record.get("uuid") or line_number)


def _added_message(
    record: dict, line_number: int, tool_names: dict[str, str]
) -> Message | None:
    """A message for an attachment or system record, or None without text."""
    role, parts = SYSTEM, tuple[Part, ...]()
    if record.get("type") == "system":
        content = record.get("content")
        if isinstance(content, str) and content.strip():
            name = str(record.get("subtype") or "notice").replace("_", " ")
            parts = (Part("context", content, name=name),)
    elif record.get("type") == "attachment" and isinstance(
        attachment := record.get("attachment"), dict
    ):
        kind = str(attachment.get("type") or "attachment")
        if kind == "queued_command":
            role, parts = USER, _parts(attachment.get("prompt"), tool_names)
        elif kind == "prompt_snapshot":
            prompt = _attachment_text(attachment.get("systemPrompt"))
            parts = (Part("context", prompt, name="system prompt"),) if prompt else ()
        else:
            text = "\n\n".join(
                text
                for field in ATTACHMENT_TEXT_FIELDS
                if (
                    text := _attachment_text(
                        attachment.get(field),
                        "\n" if field == "addedLines" else "\n\n",
                    )
                )
            )
            if text:
                parts = (Part("context", text, name=kind.replace("_", " ")),)
    if not parts:
        return None
    return Message(
        id=_record_id(record, line_number),
        role=role,
        parts=parts,
        time=parse_timestamp(record.get("timestamp")),
    )


def _attachment_text(value: object, separator: str = "\n\n") -> str:
    """The text in an attachment field: a string, file, block or list of them."""
    if isinstance(value, str):
        return "" if value == PROMPT_BOUNDARY else value.strip()
    if isinstance(value, list):
        return separator.join(
            text for item in value if (text := _attachment_text(item))
        )
    if isinstance(value, dict):
        # A file read: {"file": {"filePath", "content"}}; a text block:
        # {"type": "text", "text"}; an instruction file: {"path", "content"}.
        if isinstance(value.get("file"), dict):
            return _attachment_text(value["file"].get("content"))
        path = value.get("path")
        text = _attachment_text(value.get("text") or value.get("content"))
        return f"{path}\n{text}" if isinstance(path, str) and text else text
    return ""


def _role(record: dict, parts: tuple[Part, ...]) -> str:
    if record["type"] == ASSISTANT:
        return ASSISTANT
    if any(part.kind == "tool_result" for part in parts):
        return TOOL_RESULT
    # Text that Claude Code adds on its own, such as command caveats and
    # compaction summaries.
    if record.get("isMeta") or record.get("isCompactSummary"):
        return SYSTEM
    return USER


def _parts(content: object, tool_names: dict[str, str]) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (Part("text", content),)
    parts = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        match block.get("type"):
            case "text":
                parts.append(Part("text", str(block.get("text", ""))))
            case "image":
                parts.append(Part("image"))
            case "thinking":
                parts.append(Part("thinking", str(block.get("thinking", ""))))
            case "tool_use" | "server_tool_use":
                name = str(block.get("name"))
                tool_names[str(block.get("id"))] = name
                parts.append(Part("tool_call", name=name, arguments=block.get("input")))
            case "tool_result":
                name = tool_names.get(str(block.get("tool_use_id")), "unknown")
                is_error = bool(block.get("is_error"))
                parts.append(Part("tool_result", name=name, is_error=is_error))
                parts.extend(_parts(block.get("content"), tool_names))
            case block_type:
                parts.append(Part(str(block_type)))
    return tuple(parts)
