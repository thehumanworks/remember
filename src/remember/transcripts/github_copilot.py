"""GitHub Copilot CLI sessions: ~/.copilot/session-state/<id>/events.jsonl.

Each line is an event {"type", "id", "timestamp", "data"}. The event types
are described by schemas/session-events.schema.json in the Copilot CLI
package. Sessions without an events.jsonl have no conversation and are skipped.

Copilot can save the same reasoning twice: as an "assistant.reasoning" event
and as the "reasoningText" of the message. It is shown once. Streaming
fragments ("*_delta", "tool.execution_partial_result"), which later events
repeat in full, and events without text, such as usage and model changes, are
left out. Events that Copilot marks "ephemeral" are not saved.
"""

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

LABEL = "Copilot"
COPILOT_SESSIONS_DIR = Path.home() / ".copilot" / "session-state"
EVENTS_FILE = "events.jsonl"
# Events that hold context or notices: (event type, data field, label).
CONTEXT_EVENTS = {
    "system.notification": ("content", "notification"),
    "session.compaction_complete": ("summaryContent", "compaction summary"),
    "skill.context_delivered": ("content", "skill context"),
    "instruction.update": ("content", "instructions"),
    "session.mode_notice_delivered": ("content", "mode notice"),
    "session.handoff": ("summary", "handoff summary"),
    "session.task_complete": ("summary", "task summary"),
    "session.error": ("message", "error"),
    "session.warning": ("message", "warning"),
    "session.info": ("message", "info"),
}


def current_session_id() -> str | None:
    """Copilot CLI does not tell the commands it runs which session runs them."""
    return None


def list_session_ids() -> list[str]:
    return [path.parent.name for path in _session_paths()]


def get_session(session_id: str) -> SessionInfo:
    path = COPILOT_SESSIONS_DIR / session_id / EVENTS_FILE
    if session_id in ("", ".", "..") or "/" in session_id or not path.is_file():
        raise FileNotFoundError(f"no Copilot session {session_id}")
    return _load_session(path)


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
    if not COPILOT_SESSIONS_DIR.is_dir():
        return []
    return [
        directory / EVENTS_FILE
        for directory in COPILOT_SESSIONS_DIR.iterdir()
        if (directory / EVENTS_FILE).is_file()
    ]


def _load_session(path: Path) -> SessionInfo:
    end = modified_time(path)
    data: dict = {}
    with path.open(encoding="utf-8") as source:
        for _, event in read_records(source, path, ignore_invalid):
            if event.get("type") == "session.start" and isinstance(
                event.get("data"), dict
            ):
                data = event["data"]
                data.setdefault("startTime", event.get("timestamp"))
            break  # A session starts with its session.start event.
    context = data.get("context")
    cwd = context.get("cwd", "") if isinstance(context, dict) else ""
    return SessionInfo(
        id=path.parent.name,
        path=path,
        cwd=str(cwd),
        start=parse_timestamp(data.get("startTime")) or end,
        end=end,
    )


def _messages(path: Path, on_invalid: OnInvalid | None) -> Generator[Message]:
    tool_names: dict[str, str] = {}
    reasoning: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for _, event in read_records(source, path, on_invalid):
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            message = _message(str(event.get("type")), data, tool_names, reasoning)
            if message is not None and message[1]:
                role, parts = message
                yield Message(
                    id=str(event.get("id", "")),
                    role=role,
                    parts=parts,
                    time=parse_timestamp(event.get("timestamp")),
                )


def _message(
    event_type: str, data: dict, tool_names: dict[str, str], reasoning: set[str]
) -> tuple[str, tuple[Part, ...]] | None:
    if event_type in CONTEXT_EVENTS:
        field, name = CONTEXT_EVENTS[event_type]
        text = data.get(field)
        if not isinstance(text, str) or not text.strip():
            return None
        role = ASSISTANT if event_type == "session.task_complete" else SYSTEM
        return role, (Part("context", text, name=name),)
    match event_type:
        case "user.message":
            # content is what the user typed; transformedContent is what the
            # model got, with context such as the time added around it.
            content = str(data.get("content", ""))
            parts = [Part("text", content)]
            sent = data.get("transformedContent")
            if isinstance(sent, str) and sent.strip() != content.strip():
                added = sent.replace(content, "", 1) if content.strip() else sent
                parts.append(Part("context", added.strip(), name="added context"))
            return USER, tuple(parts)
        case "skill.invoked":
            name = f"skill {data.get('name') or ''}".strip()
            return SYSTEM, (Part("context", str(data.get("content", "")), name=name),)
        case "assistant.message":
            parts = _reasoning(data.get("reasoningText"), reasoning)
            parts.append(Part("text", str(data.get("content", ""))))
            for request in data.get("toolRequests") or []:
                if isinstance(request, dict):
                    parts.append(
                        Part(
                            "tool_call",
                            name=str(request.get("name")),
                            arguments=request.get("arguments"),
                        )
                    )
            return ASSISTANT, tuple(parts)
        case "assistant.reasoning":
            return ASSISTANT, tuple(_reasoning(data.get("content"), reasoning))
        case "tool.execution_start":
            tool_names[str(data.get("toolCallId"))] = str(data.get("toolName"))
            return None
        case "tool.execution_complete":
            name = tool_names.get(str(data.get("toolCallId")), "unknown")
            result = data.get("result")
            error = data.get("error")
            if isinstance(result, dict):
                text = result.get("content", "")
            elif isinstance(error, dict):
                text = error.get("message", "")
            else:
                text = ""
            header = Part(
                "tool_result", name=name, is_error=data.get("success") is False
            )
            return TOOL_RESULT, (header, Part("text", str(text)))
        case "system.message":
            sender = data.get("role")
            name = (
                "system prompt" if sender in (None, "system") else f"{sender} message"
            )
            return SYSTEM, (Part("context", str(data.get("content", "")), name=name),)
    return None


def _reasoning(text: object, seen: set[str]) -> list[Part]:
    """A thinking part, unless this reasoning was already shown."""
    if not isinstance(text, str) or not text.strip() or text.strip() in seen:
        return []
    seen.add(text.strip())
    return [Part("thinking", text)]
