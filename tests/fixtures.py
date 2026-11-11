"""Small sessions in each agent's on-disk format, for tests.

Each writer saves the same conversation: the user asks why the flake check
fails, the agent thinks, reads flake.nix with a tool, and answers. Records
follow the formats seen in real transcripts, including records that are not
part of the conversation and must be skipped.
"""

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

QUESTION = "Why does the flake check fail?"
THINKING = "Look at flake.nix."
CHECKING = "Checking flake.nix."
FILE_TEXT = "{ outputs = { ... }; }"
ANSWER = "The flake check fails because nixpkgs is not pinned."


def stamp(start: datetime, minutes: int) -> str:
    when = start.replace(tzinfo=UTC) + timedelta(minutes=minutes)
    return f"{when:%Y-%m-%dT%H:%M:%S.000Z}"


def write_jsonl(path: Path, records: list, end: str) -> Path:
    """Write records, one JSON value per line, and set the file's time to end."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    when = datetime.fromisoformat(end).timestamp()
    os.utime(path, (when, when))
    return path


def uuid(prefix: str, number: int) -> str:
    return f"{prefix}{number:04d}-0000-4000-8000-000000000000"


def claude_session(root: Path, cwd: str, session_id: str, start: datetime) -> Path:
    """A Claude Code session; the first record has no working directory."""
    prefix = session_id[:4]

    def record(kind: str, number: int, minute: int, message: dict, **extra) -> dict:
        return {
            "parentUuid": uuid(prefix, number - 1) if number > 1 else None,
            "isSidechain": False,
            "userType": "external",
            "cwd": cwd,
            "sessionId": session_id,
            "version": "2.1.318",
            "entrypoint": "cli",
            "type": kind,
            "message": message,
            "uuid": uuid(prefix, number),
            "timestamp": stamp(start, minute),
            **extra,
        }

    def reply(message_id: str, block: dict) -> dict:
        return {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [block],
        }

    records = [
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": stamp(start, 0),
        },
        record("user", 1, 1, {"role": "user", "content": QUESTION}),
        # One reply, saved as one record per content block.
        record(
            "assistant",
            2,
            2,
            reply("msg_1", {"type": "thinking", "thinking": THINKING}),
        ),
        record("assistant", 3, 2, reply("msg_1", {"type": "text", "text": CHECKING})),
        record(
            "assistant",
            4,
            2,
            reply(
                "msg_1",
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "flake.nix"},
                },
            ),
        ),
        record(
            "user",
            5,
            3,
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": FILE_TEXT}],
                    }
                ],
            },
            toolUseResult={"type": "text", "file": {"filePath": "flake.nix"}},
        ),
        {"type": "attachment", "timestamp": stamp(start, 3), "cwd": cwd},
        record(
            "user",
            6,
            4,
            {
                "role": "user",
                "content": "<local-command-caveat>Caveat</local-command-caveat>",
            },
            isMeta=True,
        ),
        record("assistant", 7, 5, reply("msg_2", {"type": "text", "text": ANSWER})),
        {"type": "system", "subtype": "turn_duration", "timestamp": stamp(start, 5)},
    ]
    path = root / re.sub(r"[^A-Za-z0-9]", "-", cwd) / f"{session_id}.jsonl"
    return write_jsonl(path, records, stamp(start, 5))


def codex_session(
    root: Path, cwd: str, session_id: str, start: datetime, folder: str = "sessions"
) -> Path:
    """A Codex rollout, with the context Codex adds as user messages."""

    def line(kind: str, minute: int, payload: dict) -> dict:
        return {"timestamp": stamp(start, minute), "type": kind, "payload": payload}

    def message(role: str, text: str, kind: str = "input_text") -> dict:
        return {
            "type": "message",
            "role": role,
            "content": [{"type": kind, "text": text}],
        }

    records = [
        line(
            "session_meta",
            0,
            {
                "id": session_id,
                "timestamp": stamp(start, 0),
                "cwd": cwd,
                "originator": "codex_cli_rs",
                "cli_version": "0.95.0",
                "source": "cli",
            },
        ),
        line("response_item", 0, message("developer", "<permissions instructions>")),
        line(
            "response_item",
            0,
            message("user", f"# AGENTS.md instructions for {cwd}\n\n<INSTRUCTIONS>"),
        ),
        line(
            "response_item",
            0,
            message(
                "user",
                f"<environment_context>\n  <cwd>{cwd}</cwd>\n</environment_context>",
            ),
        ),
        line("turn_context", 1, {"cwd": cwd, "model": "gpt-5.5"}),
        line("response_item", 1, message("user", QUESTION)),
        line(
            "event_msg",
            1,
            {"type": "user_message", "message": QUESTION, "kind": "plain"},
        ),
        line(
            "response_item",
            2,
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": THINKING}],
                "content": None,
                "encrypted_content": "gAAAA",
            },
        ),
        line(
            "response_item",
            2,
            {
                "type": "function_call",
                "name": "shell",
                "arguments": '{"command":["cat","flake.nix"]}',
                "call_id": "call_1",
            },
        ),
        line(
            "response_item",
            3,
            {"type": "function_call_output", "call_id": "call_1", "output": FILE_TEXT},
        ),
        line("response_item", 4, message("assistant", ANSWER, "output_text")),
        line("event_msg", 4, {"type": "agent_message", "message": ANSWER}),
    ]
    when = start.replace(tzinfo=UTC)
    path = (
        root
        / folder
        / f"{when:%Y/%m/%d}"
        / f"rollout-{when:%Y-%m-%dT%H-%M-%S}-{session_id}.jsonl"
    )
    return write_jsonl(path, records, stamp(start, 4))


def copilot_session(root: Path, cwd: str, session_id: str, start: datetime) -> Path:
    """A GitHub Copilot CLI session's events.jsonl."""
    prefix = session_id[:4]
    numbers = iter(range(1, 100))

    def event(kind: str, minute: int, data: dict) -> dict:
        return {
            "type": kind,
            "data": data,
            "id": uuid(prefix, next(numbers)),
            "timestamp": stamp(start, minute),
            "parentId": None,
        }

    records = [
        event(
            "session.start",
            0,
            {
                "sessionId": session_id,
                "version": 1,
                "producer": "copilot-agent",
                "copilotVersion": "1.0.105",
                "startTime": stamp(start, 0),
                "context": {"cwd": cwd, "gitRoot": cwd, "branch": "main"},
            },
        ),
        event("session.model_change", 0, {"newModel": "claude-sonnet-5.5"}),
        event(
            "user.message",
            1,
            {
                "content": QUESTION,
                "transformedContent": f"<current_datetime>now</current_datetime>\n\n{QUESTION}",
            },
        ),
        event(
            "system.message",
            1,
            {"role": "system", "content": "You are the GitHub Copilot CLI."},
        ),
        event("assistant.turn_start", 1, {"turnId": "0"}),
        event("assistant.reasoning", 2, {"reasoningId": "r1", "content": THINKING}),
        event(
            "assistant.message",
            2,
            {
                "messageId": "m1",
                "content": CHECKING,
                "toolRequests": [
                    {
                        "toolCallId": "tc_1",
                        "name": "view",
                        "arguments": {"path": "flake.nix"},
                    }
                ],
            },
        ),
        event(
            "tool.execution_start",
            2,
            {
                "toolCallId": "tc_1",
                "toolName": "view",
                "arguments": {"path": "flake.nix"},
            },
        ),
        event(
            "tool.execution_complete",
            3,
            {"toolCallId": "tc_1", "success": True, "result": {"content": FILE_TEXT}},
        ),
        event(
            "assistant.message",
            4,
            {"messageId": "m2", "content": ANSWER, "toolRequests": []},
        ),
        event("session.shutdown", 4, {"shutdownType": "routine"}),
    ]
    directory = root / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workspace.yaml").write_text(
        f"id: {session_id}\ncwd: {cwd}\n", encoding="utf-8"
    )
    return write_jsonl(directory / "events.jsonl", records, stamp(start, 4))
