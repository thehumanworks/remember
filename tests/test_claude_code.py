import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fixtures import ANSWER, CHECKING, FILE_TEXT, QUESTION, THINKING, claude_session

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

    def test_records_of_one_reply_become_one_message(self):
        messages = self.messages()
        self.assertEqual(
            [(message.role, message.text()) for message in messages],
            [
                ("user", QUESTION),
                ("assistant", CHECKING),
                ("toolResult", FILE_TEXT),
                ("system", "<local-command-caveat>Caveat</local-command-caveat>"),
                ("assistant", ANSWER),
            ],
        )
        self.assertEqual(
            messages[1].text(full=True),
            f"(thinking) {THINKING}\n\n{CHECKING}\n\n"
            '(tool call: Read) {"file_path": "flake.nix"}',
        )
        self.assertEqual(
            messages[2].text(full=True), f"(tool result: Read)\n\n{FILE_TEXT}"
        )
        # A joined message keeps the ID and time of its first record.
        self.assertEqual(messages[1].id, "7f3a0002-0000-4000-8000-000000000000")
        self.assertEqual(messages[1].time, datetime(2026, 11, 9, 10, 2, tzinfo=UTC))

    def test_roles_and_head_stop_reading_early(self):
        with self.path.open("a", encoding="utf-8") as transcript:
            transcript.write("{invalid json}\n")
        first = self.messages(roles={"user"}, head=1)
        self.assertEqual([message.text() for message in first], [QUESTION])
        with self.assertRaises(ValueError):
            self.messages()
        errors = []
        self.assertEqual(len(self.messages(on_invalid=errors.append)), 5)
        self.assertEqual(len(errors), 1)

    def test_subagent_records_are_skipped(self):
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[1] = lines[1].replace('"isSidechain": false', '"isSidechain": true')
        self.path.write_text("".join(lines), encoding="utf-8")
        self.assertNotIn(QUESTION, [message.text() for message in self.messages()])

    def test_current_session_comes_from_the_environment(self):
        with patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": SESSION_ID}):
            self.assertEqual(claude_code.current_session_id(), SESSION_ID)
        with patch.dict(os.environ):
            os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
            self.assertIsNone(claude_code.current_session_id())


if __name__ == "__main__":
    unittest.main()
