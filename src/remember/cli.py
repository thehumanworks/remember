"""
CLI entrypoint
"""

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .transcripts import SOURCES as TRANSCRIPT_SOURCES
from .transcripts.base import Message as SourceMessage
from .transcripts.base import OnInvalid, SessionInfo, TranscriptSource, ignore_invalid

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


def add_output_options(
    parser: argparse.ArgumentParser, *, default: str = argparse.SUPPRESS
) -> None:
    output = parser.add_argument_group("output")
    output.add_argument(
        "--format",
        choices=("text", "json"),
        default=default,
        help="output format (default: text)",
    )
    output.add_argument(
        "--json",
        dest="format",
        action="store_const",
        const="json",
        default=argparse.SUPPRESS,
        help="machine-readable JSON (same as --format json)",
    )


def build_parser(*, exit_on_error: bool = True) -> argparse.ArgumentParser:
    parser_options = {**PARSER_OPTIONS, "exit_on_error": exit_on_error}
    parser = argparse.ArgumentParser(
        prog="remember",
        description="Find and read messages from past coding-agent sessions.",
        epilog=REFS_HELP,
        color=False,
        **parser_options,
    )
    add_output_options(parser, default="text")
    output = argparse.ArgumentParser(add_help=False)
    add_output_options(output)
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
        help=(
            "show everything the transcript holds as text: also thinking, "
            "tool calls, tool results, system prompts, and context and "
            "notices that the agent added"
        ),
    )

    search = commands.add_parser(
        "search",
        parents=[view, scope, output],
        help="find messages containing all terms, grouped by session",
        description=(
            "Find messages containing all terms (case-insensitive), "
            "grouped by session, newest first."
        ),
        epilog=REFS_HELP,
        **parser_options,
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
        parents=[scope, output],
        help="list sessions by latest activity",
        description=(
            "List sessions by latest activity, newest first, "
            "each with its opening request."
        ),
        epilog=REFS_HELP,
        **parser_options,
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
        parents=[view, output],
        help="read a session, or the messages around one message",
        description=(
            "Read a session from the start, or the messages around one message."
        ),
        epilog=REFS_HELP,
        **parser_options,
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
    id: str
    ref: str
    role: str
    time: datetime | None
    text: str


@dataclass
class Output:
    json_enabled: bool = False
    warnings: list[str] = field(default_factory=list)

    def write_json(self, data: dict) -> None:
        print(
            json.dumps(
                {**data, "warnings": self.warnings}, ensure_ascii=False, indent=2
            )
        )

    def warn_invalid(self, error: ValueError) -> None:
        warning = f"{error} (line skipped)"
        if self.json_enabled:
            self.warnings.append(warning)
        else:
            print(f"remember: warning: {warning}", file=sys.stderr)

    def error(self, kind: str, error: Exception) -> None:
        if self.json_enabled:
            self.write_json({"error": {"type": kind, "message": str(error)}})
        else:
            print(f"remember: {error}", file=sys.stderr)


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


def load_messages(session: Session, on_invalid: OnInvalid) -> list[SourceMessage]:
    return source_of(session).read_messages(session.info, on_invalid=on_invalid)


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
                id=message.id,
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


def session_json(session: Session) -> dict:
    return {
        "ref": session.ref,
        "id": session.info.id,
        "source": session.source,
        "cwd": session.cwd,
        "start": session.start.isoformat(),
        "end": session.end.isoformat(),
    }


def message_json(message: Message, max_chars: int = 0) -> dict:
    return {
        "ref": message.ref,
        "id": message.id,
        "role": message.role,
        "time": message.time.isoformat() if message.time is not None else None,
        "text": message.text[:max_chars] if max_chars else message.text,
        "truncated": bool(max_chars and len(message.text) > max_chars),
        "total_chars": len(message.text),
    }


def run_sessions(args: argparse.Namespace, output: Output) -> int:
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
    if output.json_enabled:
        output.write_json(
            {
                "sessions": [
                    {**session_json(session), "request": request}
                    for session, request in listed[: args.limit]
                ],
                "total": len(listed),
                "limit": args.limit,
            }
        )
        return 0
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


def run_search(args: argparse.Namespace, output: Output) -> int:
    terms = [one_line(term).casefold() for term in args.terms]
    results = []
    for session in find_sessions(args):
        try:
            messages = load_messages(session, output.warn_invalid)
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
    if output.json_enabled:
        sessions = []
        remaining = args.limit
        for session, request, matches in results:
            if not remaining:
                break
            selected = matches[:remaining]
            sessions.append(
                {
                    **session_json(session),
                    "request": request,
                    "matches": [message_json(message) for message in selected],
                }
            )
            remaining -= len(selected)
        output.write_json(
            {
                "sessions": sessions,
                "total_matches": total,
                "total_sessions": len(results),
                "limit": args.limit,
            }
        )
        return 0
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


def run_read(args: argparse.Namespace, output: Output) -> int:
    source_name, session_prefix, message_prefix = args.ref
    session = resolve_session(source_name, session_prefix)
    all_messages = load_messages(session, output.warn_invalid)
    messages = view_messages(session, all_messages, args)
    if message_prefix is None:
        shown = messages[: args.after + 1]
    else:
        anchor = resolve_message(session, all_messages, message_prefix)
        earlier = [message for message in messages if message.position < anchor]
        later = [message for message in messages if message.position >= anchor]
        shown = earlier[-args.before :] if args.before else []
        shown += later[: args.after + 1]

    if output.json_enabled:
        first = messages.index(shown[0]) if shown else None
        last = messages.index(shown[-1]) if shown else None
        output.write_json(
            {
                "session": session_json(session),
                "messages": [
                    message_json(message, args.max_chars) for message in shown
                ],
                "pagination": {
                    "total": len(messages),
                    "start": first + 1 if first is not None else None,
                    "end": last + 1 if last is not None else None,
                    "earlier_ref": messages[first - 1].ref if first else None,
                    "later_ref": (
                        messages[last + 1].ref
                        if last is not None and last < len(messages) - 1
                        else None
                    ),
                },
            }
        )
        return 0
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


def wants_json(argv: Sequence[str]) -> bool:
    """Find the output mode even when another argument fails validation.

    The last valid output option wins; -- ends option parsing. Invalid
    formats do not hide a JSON request when reporting a usage error.
    """
    output_format = "text"
    for index, option in enumerate(argv):
        if option == "--":
            break
        if option == "--json":
            output_format = "json"
        elif option == "--format" and index + 1 < len(argv):
            if argv[index + 1] in ("text", "json"):
                output_format = argv[index + 1]
        elif option in ("--format=text", "--format=json"):
            output_format = option.partition("=")[2]
    return output_format == "json"


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output = Output(json_enabled=wants_json(argv))
    try:
        args = build_parser(exit_on_error=not output.json_enabled).parse_args(argv)
    except argparse.ArgumentError as error:
        output.error("usage_error", error)
        return 2
    output.json_enabled = args.format == "json"
    try:
        return COMMANDS[args.command](args, output)
    except CommandError as error:
        output.error("command_error", error)
        return 2
    except OSError as error:
        output.error("io_error", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
