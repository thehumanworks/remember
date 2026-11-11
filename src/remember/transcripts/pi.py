"""Pi sessions: ~/.pi/agent/sessions/<project>/<timestamp>_<id>.jsonl."""

import os
import sys
from collections.abc import Collection, Generator, Sequence
from pathlib import Path
from typing import Final, Literal

from .base import (
    TOOL_RESULT,
    Message,
    OnInvalid,
    Part,
    SessionInfo,
    ignore_invalid,
    modified_time,
    parse_timestamp,
    read_records,
    render,
    select_messages,
)

LABEL = "Pi"
PI_SESSIONS_DIR = Path.home() / ".pi" / "agent" / "sessions"
PiEventType = Literal[
    "session",
    "model_change",
    "thinking_level_change",
    "custom",
    "message",
    "compaction",
]
PiMessageRole = Literal["system", "user", "assistant", "toolResult"]
DEFAULT_INCLUDE_ONLY: Final[list[PiEventType]] = ["message"]


def list_projects(order_by: Literal["asc", "desc"] = "desc") -> list[str]:
    """List projects by modification time, newest first by default."""
    return sorted(
        os.listdir(PI_SESSIONS_DIR),
        key=lambda project: (PI_SESSIONS_DIR / project).stat().st_mtime,
        reverse=order_by == "desc",
    )


def list_sessions(project: str, order_by: Literal["asc", "desc"] = "desc") -> list[str]:
    """List a project's JSONL session filenames by modification time, newest first."""
    project_dir = PI_SESSIONS_DIR / project
    return sorted(
        (
            session.name
            for session in project_dir.iterdir()
            if session.is_file() and session.suffix == ".jsonl"
        ),
        key=lambda session: (project_dir / session).stat().st_mtime,
        reverse=order_by == "desc",
    )


def get_project_from_path(path: str) -> str:
    return f"--{path.removeprefix('/').removesuffix('/').replace('/', '-')}--"


def get_project_from_cwd():
    return get_project_from_path(Path.cwd().__str__())


def get_session_id(session: str) -> str:
    """Return the session ID from a <timestamp>_<id>.jsonl filename."""
    return Path(session).stem.rpartition("_")[2]


def get_session_transcript(
    project: str,
    session: str,
    include_only: PiEventType | Sequence[PiEventType] | None = DEFAULT_INCLUDE_ONLY,
    head: int | None = None,
    *,
    message_roles: PiMessageRole | Sequence[PiMessageRole] | None = None,
    on_invalid: OnInvalid | None = None,
) -> list[dict]:
    """Read original event dictionaries, defaulting to all message events.

    Both filters accept a single value or a sequence. None or an empty sequence
    disables that filter. Message roles filter the nested ``message.role`` field
    and exclude non-message events, even when include_only allows them.
    The head limit applies after both filters. Positive limits stop reading early;
    negative limits exclude that many trailing matches. Blank lines are ignored.
    Invalid JSON or non-object records raise ValueError with the file and line,
    or are skipped after passing that error to on_invalid when it is given.
    """
    session_transcript_path = PI_SESSIONS_DIR / project / session
    event_types = [include_only] if isinstance(include_only, str) else include_only
    roles = [message_roles] if isinstance(message_roles, str) else message_roles
    events: list[dict] = []
    with session_transcript_path.open(encoding="utf-8") as source:
        if head == 0:
            return events
        for _, event in read_records(source, session_transcript_path, on_invalid):
            if event_types and event.get("type") not in event_types:
                continue
            if roles:
                message = event.get("message")
                if (
                    event.get("type") != "message"
                    or not isinstance(message, dict)
                    or message.get("role") not in roles
                ):
                    continue
            events.append(event)
            if head is not None and head > 0 and len(events) >= head:
                return events
    return events if head is None else events[:head]


def get_message_parts(message: dict) -> tuple[Part, ...]:
    """A Pi message's content as parts, led by the tool name for tool results."""
    parts = []
    if message.get("role") == TOOL_RESULT:
        parts.append(
            Part(
                "tool_result",
                name=str(message.get("toolName")),
                is_error=bool(message.get("isError")),
            )
        )
    content = message.get("content")
    if isinstance(content, str):
        parts.append(Part("text", content))
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
            case "toolCall":
                parts.append(
                    Part(
                        "tool_call",
                        name=str(block.get("name")),
                        arguments=block.get("arguments"),
                    )
                )
            case block_type:
                parts.append(Part(str(block_type)))
    return tuple(parts)


def get_message_text(message: dict, full: bool = False) -> str:
    """Render a message's content as readable text.

    By default only text and images are shown, so a message that only calls
    tools renders as an empty string. full also shows thinking, tool calls and
    which tool a result came from.
    """
    return render(get_message_parts(message), full)


# TranscriptSource


def current_session_id() -> str | None:
    """Pi tells the commands it runs which session runs them."""
    return os.environ.get("PI_SESSION_ID") or None


def list_session_ids() -> list[str]:
    return [get_session_id(path.name) for path in _session_paths()]


def get_session(session_id: str) -> SessionInfo:
    for path in _session_paths():
        if get_session_id(path.name) == session_id:
            return _load_session(path)
    raise FileNotFoundError(f"no Pi session {session_id}")


def find_sessions(project: Path | None = None) -> list[SessionInfo]:
    # Directory names are lossy, so the header's cwd decides.
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
    if not PI_SESSIONS_DIR.is_dir():
        return []
    projects = list_projects() if project_dir is None else [project_dir]
    return [
        PI_SESSIONS_DIR / project / filename
        for project in projects
        if (PI_SESSIONS_DIR / project).is_dir()
        for filename in list_sessions(project)
    ]


def _load_session(path: Path) -> SessionInfo:
    end = modified_time(path)
    headers = get_session_transcript(
        path.parent.name, path.name, "session", 1, on_invalid=ignore_invalid
    )
    header = headers[0] if headers else {}
    return SessionInfo(
        id=get_session_id(path.name),
        path=path,
        cwd=str(header.get("cwd", "")),
        start=parse_timestamp(header.get("timestamp")) or end,
        end=end,
    )


def _messages(path: Path, on_invalid: OnInvalid | None) -> Generator[Message]:
    with path.open(encoding="utf-8") as source:
        for _, event in read_records(source, path, on_invalid):
            message = event.get("message")
            if event.get("type") != "message" or not isinstance(message, dict):
                continue
            yield Message(
                id=str(event.get("id", "")),
                role=str(message.get("role")),
                parts=get_message_parts(message),
                time=parse_timestamp(event.get("timestamp")),
            )


if __name__ == "__main__":
    sessions = list_sessions(get_project_from_cwd())
    if len(sessions) == 0:
        print("No sessions")
        sys.exit(0)
    print(get_session_transcript(get_project_from_cwd(), sessions[0], head=10))
