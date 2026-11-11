"""
CLI entrypoint
"""

import argparse
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .transcripts import SOURCES as TRANSCRIPT_SOURCES
from .transcripts.base import Message as SourceMessage
from .transcripts.base import SessionInfo, TranscriptSource, ignore_invalid

SOURCES = tuple(TRANSCRIPT_SOURCES)
ROLES = ("user", "assistant")
REF_PATTERN = re.compile(
    rf"({'|'.join(map(re.escape, SOURCES))}):([^/\s]+)(?:/([^/\s]+))?"
)
RELATIVE_TIME_PATTERN = re.compile(r"(\d+)([hd])")
RELATIVE_TIME_UNITS = {"h": "hours", "d": "days"}
DEFAULT_MAX_CHARS = 2000
SHORT_ID_LENGTH = 8
PAGE_SIZE = 10
REQUEST_CHARS = 80
SNIPPET_CHARS = 160

REFS_HELP = """\
refs:
  pi:01a1ac0c             a session
  pi:01a1ac0c/9c3d2e10    a message in that session
  Copy refs from the output. Any unique ID prefix works."""

# Subparsers inherit color, but not abbreviation or suggestion behaviour.
# Abbreviations stay disabled so new options never break existing commands.
PARSER_OPTIONS = {
    "allow_abbrev": False,
    "suggest_on_error": True,
    "formatter_class": argparse.RawDescriptionHelpFormatter,
}


def parse_ref(value: str) -> tuple[str, str, str | None]:
    """Parse SOURCE:SESSION or SOURCE:SESSION/MESSAGE."""
    match = REF_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid ref {value!r}: expected SOURCE:SESSION or "
            f"SOURCE:SESSION/MESSAGE, where SOURCE is one of {', '.join(SOURCES)}"
        )
    return match.group(1), match.group(2), match.group(3)


def parse_when(value: str) -> datetime:
    """Parse an ISO date or time, or a relative time such as 7d or 12h."""
    if match := RELATIVE_TIME_PATTERN.fullmatch(value):
        amount, unit = match.groups()
        offset = timedelta(**{RELATIVE_TIME_UNITS[unit]: int(amount)})
        return datetime.now().astimezone() - offset
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid time {value!r}: use a date such as 2026-11-01, or 7d or 12h"
        ) from None
    # Times without an offset are local.
    return when.astimezone()


def parse_term(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("search terms must not be blank")
    return value


def parse_count(value: str, minimum: int) -> int:
    try:
        count = int(value)
    except ValueError:
        count = None
    if count is None or count < minimum:
        raise argparse.ArgumentTypeError(
            f"expected a whole number of at least {minimum}, got {value!r}"
        )
    return count


def positive_int(value: str) -> int:
    return parse_count(value, 1)


def non_negative_int(value: str) -> int:
    return parse_count(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remember",
        description="Find and read messages from past coding-agent sessions.",
        epilog=REFS_HELP,
        color=False,
        **PARSER_OPTIONS,
    )
    commands = parser.add_subparsers(
        title="commands", dest="command", metavar="COMMAND", required=True
    )

    scope = argparse.ArgumentParser(add_help=False)
    scope_options = scope.add_argument_group("scope")
    location = scope_options.add_mutually_exclusive_group()
    location.add_argument(
        "--project",
        type=Path,
        metavar="PATH",
        help="sessions started in PATH (default: current directory)",
    )
    location.add_argument(
        "--all-projects", action="store_true", help="sessions from every project"
    )
    scope_options.add_argument(
        "--source",
        action="append",
        choices=SOURCES,
        metavar="NAME",
        help=f"{', '.join(SOURCES)}; repeatable (default: all)",
    )
    scope_options.add_argument(
        "--since",
        type=parse_when,
        metavar="WHEN",
        help="active since WHEN: 2026-11-01, 7d or 12h",
    )
    scope_options.add_argument(
        "--until", type=parse_when, metavar="WHEN", help="active before WHEN"
    )
    scope_options.add_argument(
        "--include-current",
        action="store_true",
        help="include the session running this command (default: excluded)",
    )

    view = argparse.ArgumentParser(add_help=False)
    view.add_argument(
        "--role",
        action="append",
        choices=ROLES,
        metavar="ROLE",
        help="user or assistant; repeatable (default: both)",
    )
    view.add_argument(
        "--full",
        action="store_true",
        help="also show thinking, tool calls, tool results and system prompts",
    )

    search = commands.add_parser(
        "search",
        parents=[view, scope],
        help="find messages containing all terms, grouped by session",
        description=(
            "Find messages containing all terms (case-insensitive), "
            "grouped by session, newest first."
        ),
        epilog=REFS_HELP,
        **PARSER_OPTIONS,
    )
    search.add_argument(
        "terms",
        nargs="+",
        type=parse_term,
        metavar="TERM",
        help="text to find; quote a phrase to keep it together",
    )
    search.add_argument(
        "--limit",
        type=positive_int,
        default=20,
        metavar="N",
        help="matching messages to show (default: 20)",
    )

    sessions = commands.add_parser(
        "sessions",
        parents=[scope],
        help="list sessions by latest activity",
        description=(
            "List sessions by latest activity, newest first, "
            "each with its opening request."
        ),
        epilog=REFS_HELP,
        **PARSER_OPTIONS,
    )
    sessions.add_argument(
        "--limit",
        type=positive_int,
        default=10,
        metavar="N",
        help="sessions to show (default: 10)",
    )

    read = commands.add_parser(
        "read",
        parents=[view],
        help="read a session, or the messages around one message",
        description=(
            "Read a session from the start, or the messages around one message."
        ),
        epilog=REFS_HELP,
        **PARSER_OPTIONS,
    )
    read.add_argument(
        "ref", type=parse_ref, metavar="REF", help="session or message ref"
    )
    read.add_argument(
        "-B",
        dest="before",
        type=non_negative_int,
        default=3,
        metavar="N",
        help="messages before a message ref (default: 3)",
    )
    read.add_argument(
        "-A",
        dest="after",
        type=non_negative_int,
        default=10,
        metavar="N",
        help=(
            "messages after a message ref (default: 10); "
            "a session ref starts at its first message"
        ),
    )
    read.add_argument(
        "--max-chars",
        type=non_negative_int,
        default=DEFAULT_MAX_CHARS,
        metavar="N",
        help="characters per message before truncation (default: 2000, 0 = all)",
    )
    return parser


class CommandError(Exception):
    """A request that cannot be met, reported without a traceback."""


@dataclass(frozen=True)
class Session:
    ref: str
    source: str
    info: SessionInfo

    @property
    def cwd(self) -> str:
        return self.info.cwd

    @property
    def start(self) -> datetime:
        return self.info.start

    @property
    def end(self) -> datetime:
        return self.info.end


@dataclass(frozen=True)
class Message:
    # Index among all of the session's messages, so windows can anchor on
    # messages that the current view hides.
    position: int
    ref: str
    role: str
    time: datetime | None
    text: str


def warn_invalid(error: ValueError) -> None:
    print(f"remember: warning: {error} (line skipped)", file=sys.stderr)


def source_of(session: Session) -> TranscriptSource:
    return TRANSCRIPT_SOURCES[session.source]


def selected_sources(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys(args.source)) if args.source else list(SOURCES)


def short_ids(ids: Iterable[str]) -> dict[str, str]:
    """Map each ID to its shortest unique prefix of at least 8 characters."""
    ordered = sorted(set(ids))
    shorts = {}
    for index, full_id in enumerate(ordered):
        neighbours = ordered[max(index - 1, 0) : index] + ordered[index + 1 : index + 2]
        length = max(
            [SHORT_ID_LENGTH]
            + [len(os.path.commonprefix([full_id, other])) + 1 for other in neighbours]
        )
        # Pi session IDs are UUIDv7, so sessions started close together share
        # a prefix. Never end a ref on a UUID separator.
        if full_id[:length].endswith("-"):
            length += 1
        shorts[full_id] = full_id[:length]
    return shorts


def requested_project(args: argparse.Namespace) -> Path | None:
    if args.all_projects:
        return None
    return (args.project.expanduser() if args.project else Path.cwd()).resolve()


def find_sessions(args: argparse.Namespace) -> list[Session]:
    """Sessions in the requested scope, from every selected source, latest first."""
    project = requested_project(args)
    sessions = []
    for name in selected_sources(args):
        source = TRANSCRIPT_SOURCES[name]
        shorts = short_ids(source.list_session_ids())
        current = None if args.include_current else source.current_session_id()
        for info in source.find_sessions(project):
            if info.id == current:
                continue
            if args.since and info.end < args.since:
                continue
            if args.until and info.start >= args.until:
                continue
            sessions.append(
                Session(f"{name}:{shorts.get(info.id, info.id)}", name, info)
            )
    return sorted(sessions, key=lambda session: session.end, reverse=True)


def resolve_session(name: str, prefix: str) -> Session:
    source = TRANSCRIPT_SOURCES[name]
    session_ids = list(dict.fromkeys(source.list_session_ids()))
    shorts = short_ids(session_ids)
    matches = [
        session_id for session_id in session_ids if session_id.startswith(prefix)
    ]
    exact = [session_id for session_id in matches if session_id == prefix]
    sessions = []
    for session_id in exact or matches:
        try:
            info = source.get_session(session_id)
        except OSError:
            continue  # Removed after it was listed.
        sessions.append(Session(f"{name}:{shorts[session_id]}", name, info))
    sessions.sort(key=lambda session: session.end, reverse=True)
    if not sessions:
        raise CommandError(
            f"no {source.LABEL} session matches {name}:{prefix}; "
            "find refs with: remember sessions --all-projects"
        )
    if len(sessions) > 1:
        candidates = "\n".join(
            f"  {session.ref}  {format_span(session)}  {session.cwd}"
            for session in sessions[:PAGE_SIZE]
        )
        raise CommandError(
            f"{name}:{prefix} matches {len(sessions)} sessions; "
            f"use a longer ref:\n{candidates}"
        )
    return sessions[0]


def resolve_message(
    session: Session, messages: list[SourceMessage], prefix: str
) -> int:
    """Return the position of the message whose ID starts with prefix."""
    matches = [
        position
        for position, message in enumerate(messages)
        if message.id.startswith(prefix)
    ]
    exact = [position for position in matches if messages[position].id == prefix]
    matches = exact or matches
    if not matches:
        raise CommandError(
            f"no message {prefix} in {session.ref}; "
            f"read the session with: remember read {session.ref}"
        )
    if len(matches) > 1:
        raise CommandError(
            f"{session.ref}/{prefix} matches {len(matches)} messages; "
            "use the full message ID"
        )
    return matches[0]


def load_messages(session: Session) -> list[SourceMessage]:
    return source_of(session).read_messages(session.info, on_invalid=warn_invalid)


def view_messages(
    session: Session, messages: list[SourceMessage], args: argparse.Namespace
) -> list[Message]:
    """Messages in the requested view, skipping those with nothing to show."""
    if args.role:
        roles = set(args.role)
    else:
        roles = None if args.full else set(ROLES)
    # Refs use the shortest unique prefix among all of the session's messages,
    # so they stay the same in every view.
    shorts = short_ids(message.id for message in messages)
    shown = []
    for position, message in enumerate(messages):
        if roles is not None and message.role not in roles:
            continue
        text = message.text(args.full)
        if not text:
            continue
        shown.append(
            Message(
                position=position,
                ref=f"{session.ref}/{shorts[message.id]}",
                role=message.role,
                time=message.time,
                text=text,
            )
        )
    return shown


def opening_request(messages: Iterable[SourceMessage]) -> str | None:
    for message in messages:
        if message.role == "user":
            return message.text()
    return None


def one_line(text: str) -> str:
    return " ".join(text.split())


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    word = singular if count == 1 else plural_form or f"{singular}s"
    return f"{count} {word}"


def format_time(when: datetime | None, day: datetime) -> str:
    """HH:MM, with the date when it is not day's date."""
    if when is None:
        return "--:--"
    return when.strftime("%H:%M" if when.date() == day.date() else "%Y-%m-%d %H:%M")


def format_span(session: Session) -> str:
    return f"{session.start:%Y-%m-%d %H:%M}–{format_time(session.end, session.start)}"


def format_session(session: Session, request: str, show_project: bool) -> str:
    if len(request := one_line(request)) > REQUEST_CHARS:
        request = request[: REQUEST_CHARS - 1] + "…"
    fields = [session.ref, format_span(session)]
    if show_project:
        fields.append(session.cwd)
    fields.append(f'"{request}"')
    return "  ".join(fields)


def snippet(text: str, terms: Sequence[str]) -> str:
    """One line of text around the first matching term."""
    text = one_line(text)
    folded = text.casefold()
    found = [index for term in terms if (index := folded.find(term)) >= 0]
    # Start a little before the first match, but fill the snippet near the end.
    start = min(min(found, default=0) - SNIPPET_CHARS // 4, len(text) - SNIPPET_CHARS)
    start = max(0, start)
    end = start + SNIPPET_CHARS
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def suggest(options: list[str]) -> str:
    if len(options) < 2:
        return "".join(f" Try {option}." for option in options)
    return f" Try {', '.join(options[:-1])} or {options[-1]}."


def no_results(what: str, args: argparse.Namespace, options: list[str]) -> str:
    project = requested_project(args)
    where = "" if project is None else f" in {project}"
    if args.since or args.until:
        options.append("a wider --since or --until")
    if project is not None:
        options.append("--all-projects")
    return f"No {what}{where}.{suggest(options)}"


def view_flags(args: argparse.Namespace) -> str:
    """Options that keep a follow-up read in the same view."""
    flags = [f" --role {role}" for role in args.role or ()]
    if args.full:
        flags.append(" --full")
    if args.max_chars != DEFAULT_MAX_CHARS:
        flags.append(f" --max-chars {args.max_chars}")
    return "".join(flags)


def run_sessions(args: argparse.Namespace) -> int:
    listed = []
    for session in find_sessions(args):
        try:
            first_user_messages = source_of(session).read_messages(
                session.info, roles=("user",), head=1, on_invalid=ignore_invalid
            )
        except OSError:
            continue  # Removed after it was listed.
        # Sessions without a request have nothing to recall.
        if (request := opening_request(first_user_messages)) is not None:
            listed.append((session, request))
    if not listed:
        print(no_results("sessions", args, []))
        return 0
    for session, request in listed[: args.limit]:
        print(format_session(session, request, args.all_projects))
    if len(listed) > args.limit:
        print(
            f"\nShowing {args.limit} of {len(listed)} sessions. "
            f"Rerun with --limit {len(listed)} to see all."
        )
    return 0


def run_search(args: argparse.Namespace) -> int:
    terms = [one_line(term).casefold() for term in args.terms]
    results = []
    for session in find_sessions(args):
        try:
            messages = load_messages(session)
        except OSError:
            continue  # Removed after it was listed.
        matches = [
            message
            for message in view_messages(session, messages, args)
            if all(term in one_line(message.text).casefold() for term in terms)
        ]
        if matches:
            results.append((session, opening_request(messages) or "", matches))
    total = sum(len(matches) for _, _, matches in results)
    if not total:
        options = ["fewer terms"] + ([] if args.full else ["--full"])
        print(no_results("matches", args, options))
        return 0
    shown = 0
    for session, request, matches in results:
        if shown == args.limit:
            break
        print(format_session(session, request, args.all_projects))
        for message in matches[: args.limit - shown]:
            time = format_time(message.time, session.start)
            text = snippet(message.text, terms)
            print(f"  {message.ref}  {time}  {message.role:<10} {text}")
            shown += 1
        print()
    summary = (
        f"{plural(total, 'match', 'matches')} in {plural(len(results), 'session')}"
    )
    if shown < total:
        summary = (
            f"Showing {shown} of {summary}. "
            f"Narrow the search or rerun with --limit {total}"
        )
    first = results[0][2][0]
    full = " --full" if args.full else ""
    print(f"{summary} · context: remember read {first.ref}{full}")
    return 0


def run_read(args: argparse.Namespace) -> int:
    source_name, session_prefix, message_prefix = args.ref
    session = resolve_session(source_name, session_prefix)
    all_messages = load_messages(session)
    messages = view_messages(session, all_messages, args)
    if message_prefix is None:
        shown = messages[: args.after + 1]
    else:
        anchor = resolve_message(session, all_messages, message_prefix)
        earlier = [message for message in messages if message.position < anchor]
        later = [message for message in messages if message.position >= anchor]
        shown = earlier[-args.before :] if args.before else []
        shown += later[: args.after + 1]

    count = plural(len(messages), "message")
    print(" · ".join([session.ref, session.cwd, format_span(session), count]))
    for message in shown:
        time = format_time(message.time, session.start)
        print(f"\n[{message.role} · {message.ref} · {time}]")
        text = message.text
        if args.max_chars and len(text) > args.max_chars:
            more = len(text) - args.max_chars
            full = " --full" if args.full else ""
            text = (
                f"{text[: args.max_chars].rstrip()}\n[{more:,} more chars · "
                f"remember read {message.ref} -B 0 -A 0 --max-chars 0{full}]"
            )
        print(text)
    print()
    if not messages:
        print("No messages in this view." + ("" if args.full else " Try --full."))
        return 0
    if not shown:
        print("No messages from this position in this view.")
        return 0
    first, last = messages.index(shown[0]), messages.index(shown[-1])
    flags = view_flags(args)
    if first == last:
        parts = [f"message {first + 1} of {len(messages)}"]
    else:
        parts = [f"messages {first + 1}–{last + 1} of {len(messages)}"]
    if first > 0:
        earlier_ref = messages[first - 1].ref
        parts.append(f"earlier: remember read {earlier_ref} -B {PAGE_SIZE} -A 0{flags}")
    if last < len(messages) - 1:
        parts.append(f"later: remember read {messages[last + 1].ref} -B 0{flags}")
    print(" · ".join(parts))
    return 0


COMMANDS = {"search": run_search, "sessions": run_sessions, "read": run_read}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except CommandError as error:
        print(f"remember: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
