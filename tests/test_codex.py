import json
import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fixtures import (
    ANSWER,
    FILE_TEXT,
    QUESTION,
    THINKING,
    codex_session,
    stamp,
    write_jsonl,
)

from remember.transcripts import codex

PROJECT = "/work/remember"
SESSION_ID = "019a4f2e-7c31-7b20-9f4d-3e8a1c6b2d50"
ARCHIVED_ID = "019a3b10-1111-7222-8333-444455556666"
START = datetime(2026, 11, 9, 10, 0)


class CodexTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        codex_home = patch.object(codex, "CODEX_HOME", self.root)
        codex_home.start()
        self.addCleanup(codex_home.stop)
        self.path = codex_session(self.root, PROJECT, SESSION_ID, START)
        codex_session(
            self.root,
            "/work/other",
            ARCHIVED_ID,
            datetime(2026, 11, 2, 9, 0),
            folder="archived_sessions",
        )

    def messages(self, **kwargs):
        return codex.read_messages(codex.get_session(SESSION_ID), **kwargs)

    def test_session_ids_come_from_rollout_file_names(self):
        self.assertEqual(
            codex.get_session_id(
                Path(f"rollout-2026-11-09T10-00-00-{SESSION_ID}.jsonl")
            ),
            SESSION_ID,
        )
        self.assertCountEqual(codex.list_session_ids(), [SESSION_ID, ARCHIVED_ID])

    def test_session_metadata_gives_directory_and_start(self):
        session = codex.get_session(SESSION_ID)
        self.assertEqual(session.path, self.path)
        self.assertEqual(session.cwd, PROJECT)
        self.assertEqual(session.start, START.replace(tzinfo=UTC))
        self.assertEqual(session.end, datetime(2026, 11, 9, 10, 4, tzinfo=UTC))
        with self.assertRaises(FileNotFoundError):
            codex.get_session("ffffffff")

    def test_find_sessions_includes_archived_sessions(self):
        self.assertEqual(
            [session.id for session in codex.find_sessions(Path(PROJECT))], [SESSION_ID]
        )
        self.assertCountEqual(
            [session.id for session in codex.find_sessions()], [SESSION_ID, ARCHIVED_ID]
        )
        with patch.object(codex, "CODEX_HOME", self.root / "missing"):
            self.assertEqual(codex.find_sessions(), [])

    def test_context_is_not_the_users_and_ui_events_are_skipped(self):
        messages = self.messages()
        self.assertEqual(
            [(message.id, message.role) for message in messages],
            [
                ("2", "system"),
                ("3", "system"),
                ("4", "system"),
                ("6", "user"),
                ("8", "assistant"),
                ("9", "assistant"),
                ("10", "toolResult"),
                ("11", "assistant"),
            ],
        )
        self.assertEqual(
            [message.text() for message in messages if message.role == "user"],
            [QUESTION],
        )
        self.assertEqual(
            [message.text(full=True) for message in messages[4:]],
            [
                f"(thinking) {THINKING}",
                '(tool call: shell) {"command": ["cat", "flake.nix"]}',
                f"(tool result: shell)\n\n{FILE_TEXT}",
                ANSWER,
            ],
        )
        self.assertEqual(messages[3].time, datetime(2026, 11, 9, 10, 1, tzinfo=UTC))

    def test_tool_output_forms(self):
        for output, expected in (
            (
                [{"type": "input_text", "text": "listed"}],
                "(tool result: apply_patch)\n\nlisted",
            ),
            (
                {"content": "denied", "success": False},
                "(tool result: apply_patch, error)\n\ndenied",
            ),
        ):
            with self.subTest(output=output):
                records = [
                    {
                        "type": "session_meta",
                        "timestamp": stamp(START, 0),
                        "payload": {"cwd": PROJECT},
                    },
                    {
                        "type": "response_item",
                        "timestamp": stamp(START, 1),
                        "payload": {
                            "type": "custom_tool_call",
                            "name": "apply_patch",
                            "input": "*** Begin Patch",
                            "call_id": "call_2",
                        },
                    },
                    {
                        "type": "response_item",
                        "timestamp": stamp(START, 1),
                        "payload": {
                            "type": "custom_tool_call_output",
                            "call_id": "call_2",
                            "output": output,
                        },
                    },
                ]
                write_jsonl(self.path, records, stamp(START, 1))
                call, result = self.messages()
                # Free-form tool input is shown as it is, not as JSON.
                self.assertEqual(
                    call.text(full=True), "(tool call: apply_patch) *** Begin Patch"
                )
                self.assertEqual(result.text(full=True), expected)

    def test_roles_and_head_stop_reading_early(self):
        with self.path.open("a", encoding="utf-8") as rollout:
            rollout.write(json.dumps([]) + "\n")
        first = self.messages(roles={"user"}, head=1)
        self.assertEqual([message.text() for message in first], [QUESTION])
        with self.assertRaises(ValueError):
            self.messages()

    def test_current_session_comes_from_the_environment(self):
        with patch.dict(os.environ, {"CODEX_THREAD_ID": SESSION_ID}):
            self.assertEqual(codex.current_session_id(), SESSION_ID)
        with patch.dict(os.environ):
            os.environ.pop("CODEX_THREAD_ID", None)
            self.assertIsNone(codex.current_session_id())


if __name__ == "__main__":
    unittest.main()
