"""The interface that every transcript source implements, and its shared types.

A source is a module that reads the saved sessions of one coding agent, for
example ``pi`` or ``claude_code``. The module defines every member of
TranscriptSource at module level, so the module itself satisfies the protocol
(PEP 544). To add a source, write such a module and register it in
``remember.transcripts.SOURCES``.

Sources turn their own records into Message objects made of Part objects.
Rendering is shared, so every source prints thinking, tool calls and tool
results in the same way.

The full view must show everything a transcript records as text: messages,
thinking, tool calls and results, system prompts, and the context and notices
that the agent added. A source leaves out only records with no text, such as
token counts and settings, and streaming fragments that a later record repeats
in full. Each source lists what it leaves out in its module docstring.
"""

import json
from collections.abc import Callable, Collection, Generator, Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

type OnInvalid = Callable[[ValueError], None]

# Roles after normalisation. "system" covers context that the agent added on
# its own, such as system prompts, instructions and compaction summaries.
USER = "user"
ASSISTANT = "assistant"
TOOL_RESULT = "toolResult"
SYSTEM = "system"


@dataclass(frozen=True)
class SessionInfo:
    """A saved session: where its transcript is, and when and where it ran."""

    id: str
    path: Path
    # Working directory recorded when the session started; "" when unknown.
    cwd: str
    start: datetime
    # Time of the latest write to the transcript.
    end: datetime


@dataclass(frozen=True)
class Part:
    """One piece of a message.

    kind is "text", "image", "thinking", "tool_call", "tool_result",
    "context", or a source's own block type, which is shown by name in the
    full view. "context" is text that the agent added to the conversation,
    such as a system prompt or a compaction summary; name says what it is.
    """

    kind: str
    text: str = ""
    # Tool name for tool calls and tool results; what the text is, for context.
    name: str = ""
    # Tool call arguments: a JSON value, or text when the tool takes text.
    arguments: object = None
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    id: str
    role: str
    parts: tuple[Part, ...]
    time: datetime | None = None

    def text(self, full: bool = False) -> str:
        return render(self.parts, full)


@runtime_checkable
class TranscriptSource(Protocol):
    """What the CLI needs from the saved sessions of one coding agent."""

    LABEL: str
    """Name for people, for example "Claude Code"."""

    def list_session_ids(self) -> list[str]:
        """Every saved session ID, found without reading any transcript."""
        ...

    def get_session(self, session_id: str) -> SessionInfo:
        """The session with this exact ID.

        Raises FileNotFoundError when there is no such session.
        """
        ...

    def find_sessions(self, project: Path | None = None) -> list[SessionInfo]:
        """Sessions that started in project, or all sessions when None.

        project is an absolute, resolved path. Sessions that cannot be read
        are skipped. The order is not defined.
        """
        ...

    def read_messages(
        self,
        session: SessionInfo,
        *,
        roles: Collection[str] | None = None,
        head: int | None = None,
        on_invalid: OnInvalid | None = None,
    ) -> list[Message]:
        """The session's messages, in order.

        roles keeps only messages with these normalised roles. head stops
        reading after that many messages. Lines that are not JSON objects
        raise ValueError, or go to on_invalid and are skipped when it is given.
        """
        ...

    def current_session_id(self) -> str | None:
        """The ID of the session that runs this process, or None.

        Agents that tell their commands which session runs them do so through
        an environment variable. None means this agent does not say, or this
        process does not run inside it.
        """
        ...


def render(parts: Iterable[Part], full: bool = False) -> str:
    """Render parts as readable text.

    By default only text and images are shown. full also shows thinking,
    tool calls, which tool a result came from, added context, and other block
    types by name.
    """
    return "\n\n".join(text for part in parts if (text := render_part(part, full)))


def render_part(part: Part, full: bool) -> str:
    match part.kind:
        case "text":
            return part.text.strip()
        case "image":
            return "(image)"
        case _ if not full:
            return ""
        case "thinking" | "context":
            # Encrypted or redacted thinking has no text, but it happened.
            label = "thinking" if part.kind == "thinking" else part.name
            return f"({label}) {part.text.strip()}".rstrip()
        case "tool_call":
            arguments = part.arguments
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            return f"(tool call: {part.name}) {arguments}"
        case "tool_result":
            error = ", error" if part.is_error else ""
            return f"(tool result: {part.name}{error})"
        case kind:
            return f"({kind})"


def read_records(
    source: IO[str], path: Path, on_invalid: OnInvalid | None = None
) -> Iterator[tuple[int, dict]]:
    """Yield (line number, record) for each JSON object in a JSONL file.

    Blank lines are ignored. Invalid JSON or non-object records raise
    ValueError with the file and line, or are skipped after passing that
    error to on_invalid when it is given.
    """
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            error = ValueError(
                f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
            )
            error.__cause__ = exc
        else:
            if isinstance(record, dict):
                yield line_number, record
                continue
            error = ValueError(
                f"Expected a JSON object in {path} at line {line_number}"
            )
        if on_invalid is None:
            raise error
        on_invalid(error)


def select_messages(
    messages: Generator[Message],
    roles: Collection[str] | None = None,
    head: int | None = None,
) -> list[Message]:
    """Keep messages with the given roles, stopping after head of them.

    Closes messages afterwards, so a source that reads lazily stops reading
    and closes its file as soon as enough messages are found.
    """
    selected: list[Message] = []
    with closing(messages):
        if head is not None and head <= 0:
            return selected
        for message in messages:
            if roles and message.role not in roles:
                continue
            selected.append(message)
            if head is not None and len(selected) >= head:
                break
    return selected


def ignore_invalid(error: ValueError) -> None:
    pass


def parse_timestamp(value: object) -> datetime | None:
    """An ISO 8601 time as an aware local datetime, or None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).astimezone()
    except ValueError:
        return None


def modified_time(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def json_or_text(value: object) -> object:
    """Decode a JSON string, such as tool arguments; keep other text as it is."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
