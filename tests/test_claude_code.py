import json
import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fixtures import (
    ANSWER,
    CHECKING,
    FILE_TEXT,
    QUESTION,
    QUEUED,
    SYSTEM_PROMPT,
    THINKING,
    claude_session,
)

from remember.transcripts import claude_code

PROJECT = "/work/remember"
SESSION_ID = "7f3a91c2-0d4e-4b7a-9c61-2f5e8a1b3c4d"
OTHER_ID = "2b6c0e15-3f7a-4d2e-8b19-6a4c2d8e9f01"
START = datetime(2026, 11, 9, 10, 0)


class ClaudeCodeTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        projects_dir = patch.object(claude_code, "CLAUDE_PROJECTS_DIR", self.root)
        projects_dir.start()
        self.addCleanup(projects_dir.stop)
        self.path = claude_session(self.root, PROJECT, SESSION_ID, START)
        claude_session(self.root, "/work/other", OTHER_ID, datetime(2026, 11, 8, 9, 0))

    def messages(self, **kwargs):
        return claude_code.read_messages(claude_code.get_session(SESSION_ID), **kwargs)

    def test_project_directory_names_replace_every_other_character(self):
        self.assertEqual(
            claude_code.get_project_from_path("/home/nixos/.config/nixos"),
            "-home-nixos--config-nixos",
        )

    def test_sessions_are_found_by_file_name(self):
        self.assertCountEqual(claude_code.list_session_ids(), [SESSION_ID, OTHER_ID])
        session = claude_code.get_session(SESSION_ID)
        self.assertEqual(session.path, self.path)
        # The first record has no working directory, so a later one gives it.
        self.assertEqual(session.cwd, PROJECT)
        self.assertEqual(session.start, START.replace(tzinfo=UTC))
        self.assertEqual(session.end, datetime(2026, 11, 9, 10, 5, tzinfo=UTC))
        with self.assertRaises(FileNotFoundError):
            claude_code.get_session("ffffffff")

    def test_find_sessions_uses_the_recorded_directory(self):
        # /work-remember and /work/remember share a directory name.
        claude_session(self.root, "/work-remember", "0c9d", datetime(2026, 11, 7, 8, 0))
        self.assertEqual(
            [session.id for session in claude_code.find_sessions(Path(PROJECT))],
            [SESSION_ID],
        )
        self.assertCountEqual(
            [session.id for session in claude_code.find_sessions()],
            [SESSION_ID, OTHER_ID, "0c9d"],
        )
        self.assertEqual(claude_code.find_sessions(Path("/work/none")), [])

    def test_without_a_projects_directory(self):
        with patch.object(claude_code, "CLAUDE_PROJECTS_DIR", self.root / "missing"):
            self.assertEqual(claude_code.list_session_ids(), [])
            self.assertEqual(claude_code.find_sessions(), [])

    def test_full_view_shows_every_record_with_text(self):
        self.assertEqual(
            [(message.role, message.text(full=True)) for message in self.messages()],
            [
                ("system", f"(system prompt) {SYSTEM_PROMPT}"),
                ("user", QUESTION),
                (
                    "assistant",
                    (
                        f"(thinking) {THINKING}\n\n{CHECKING}\n\n"
                        '(tool call: Read) {"file_path": "flake.nix"}'
                    ),
                ),
                # Added while the reply was written, so shown after it.
                ("system", "(total tokens reminder) <total_tokens>9000</total_tokens>"),
                ("toolResult", f"(tool result: Read)\n\n{FILE_TEXT}"),
                (
                    "system",
                    "(local command) <local-command-stdout>Reloaded</local-command-stdout>",
                ),
                ("system", "<local-command-caveat>Caveat</local-command-caveat>"),
                ("user", QUEUED),
                ("assistant", ANSWER),
            ],
        )

    def test_default_view_shows_what_was_said(self):
        self.assertEqual(
            [
                (message.role, text)
                for message in self.messages(roles={"user", "assistant"})
                if (text := message.text())
            ],
            [
                ("user", QUESTION),
                ("assistant", CHECKING),
                # Queued prompts are only saved as attachments.
                ("user", QUEUED),
                ("assistant", ANSWER),
            ],
        )

    def test_records_of_one_reply_become_one_message(self):
        reply = self.messages()[2]
        # A joined message keeps the ID and time of its first record.
        self.assertEqual(reply.id, "7f3a0002-0000-4000-8000-000000000000")
        self.assertEqual(reply.time, datetime(2026, 11, 9, 10, 2, tzinfo=UTC))
        self.assertEqual(
            [part.kind for part in reply.parts], ["thinking", "text", "tool_call"]
        )

    def test_attachment_text_forms(self):
        for attachment, expected in (
            (
                {
                    "type": "file",
                    "filename": "/work/flake.nix",
                    "content": {"type": "text", "file": {"content": FILE_TEXT}},
                },
                f"(file) /work/flake.nix\n\n{FILE_TEXT}",
            ),
            (
                {
                    "type": "instructions",
                    "files": [{"path": "/work/AGENTS.md", "content": "Use uv."}],
                },
                "(instructions) /work/AGENTS.md\nUse uv.",
            ),
            (
                {"type": "deferred_tools_delta", "addedLines": ["Read", "Edit"]},
                "(deferred tools delta) Read\nEdit",
            ),
            ({"type": "date", "date": "2026-11-09"}, None),
        ):
            with self.subTest(kind=attachment["type"]):
                record = {"type": "attachment", "uuid": "a1", "attachment": attachment}
                self.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                texts = [message.text(full=True) for message in self.messages()]
                self.assertEqual(texts, [expected] if expected else [])

    def test_roles_and_head_stop_reading_early(self):
        with self.path.open("a", encoding="utf-8") as transcript:
            transcript.write("{invalid json}\n")
        first = self.messages(roles={"user"}, head=1)
        self.assertEqual([message.text() for message in first], [QUESTION])
        with self.assertRaises(ValueError):
            self.messages()
        errors = []
        self.assertEqual(len(self.messages(on_invalid=errors.append)), 9)
        self.assertEqual(len(errors), 1)

    def test_subagent_records_are_skipped(self):
        text = self.path.read_text(encoding="utf-8")
        question = json.dumps(QUESTION)
        lines = [
            line.replace('"isSidechain": false', '"isSidechain": true')
            if question in line
            else line
            for line in text.splitlines(keepends=True)
        ]
        self.path.write_text("".join(lines), encoding="utf-8")
        texts = [message.text() for message in self.messages()]
        self.assertNotIn(QUESTION, texts)
        self.assertIn(ANSWER, texts)

    def test_current_session_comes_from_the_environment(self):
        with patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": SESSION_ID}):
            self.assertEqual(claude_code.current_session_id(), SESSION_ID)
        with patch.dict(os.environ):
            os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
            self.assertIsNone(claude_code.current_session_id())


if __name__ == "__main__":
    unittest.main()
