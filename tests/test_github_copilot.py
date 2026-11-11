import json
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
    SUMMARY,
    SYSTEM_PROMPT,
    THINKING,
    copilot_session,
)

from remember.transcripts import github_copilot

PROJECT = "/work/remember"
SESSION_ID = "07b4f288-3402-461c-9749-dd356b0e8cf2"
OTHER_ID = "5c1e9a70-8d2b-4f6e-a913-0b7c4e2d6f18"
START = datetime(2026, 11, 9, 10, 0)


class CopilotTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        sessions_dir = patch.object(github_copilot, "COPILOT_SESSIONS_DIR", self.root)
        sessions_dir.start()
        self.addCleanup(sessions_dir.stop)
        self.path = copilot_session(self.root, PROJECT, SESSION_ID, START)
        copilot_session(self.root, "/work/other", OTHER_ID, datetime(2026, 11, 8, 9, 0))
        # Copilot creates a directory before a session has any events.
        (self.root / "2bfa70fa-37a0-40df-b784-00699ee1b937").mkdir()

    def messages(self, **kwargs):
        return github_copilot.read_messages(
            github_copilot.get_session(SESSION_ID), **kwargs
        )

    def test_sessions_need_an_events_file(self):
        self.assertCountEqual(github_copilot.list_session_ids(), [SESSION_ID, OTHER_ID])
        session = github_copilot.get_session(SESSION_ID)
        self.assertEqual(session.path, self.path)
        self.assertEqual(session.cwd, PROJECT)
        self.assertEqual(session.start, START.replace(tzinfo=UTC))
        self.assertEqual(session.end, datetime(2026, 11, 9, 10, 4, tzinfo=UTC))
        for missing in ("2bfa70fa-37a0-40df-b784-00699ee1b937", "ffffffff", "..", ""):
            with self.subTest(session_id=missing), self.assertRaises(FileNotFoundError):
                github_copilot.get_session(missing)

    def test_find_sessions_by_directory(self):
        self.assertEqual(
            [session.id for session in github_copilot.find_sessions(Path(PROJECT))],
            [SESSION_ID],
        )
        self.assertCountEqual(
            [session.id for session in github_copilot.find_sessions()],
            [SESSION_ID, OTHER_ID],
        )
        with patch.object(
            github_copilot, "COPILOT_SESSIONS_DIR", self.root / "missing"
        ):
            self.assertEqual(github_copilot.find_sessions(), [])

    def test_events_become_messages(self):
        messages = self.messages()
        self.assertEqual(
            [(message.role, message.text()) for message in messages],
            [
                # What the user typed, without the context Copilot adds.
                ("user", QUESTION),
                ("system", ""),
                ("assistant", ""),
                ("assistant", CHECKING),
                ("toolResult", FILE_TEXT),
                ("assistant", ANSWER),
                ("system", ""),
                ("system", ""),
            ],
        )
        self.assertEqual(messages[0].id, "07b40003-0000-4000-8000-000000000000")

    def test_full_view_shows_every_event_with_text(self):
        self.assertEqual(
            [message.text(full=True) for message in self.messages()],
            [
                f"{QUESTION}\n\n(added context) <current_datetime>now</current_datetime>",
                f"(system prompt) {SYSTEM_PROMPT}",
                # The message repeats this reasoning; it is shown once.
                f"(thinking) {THINKING}",
                f'{CHECKING}\n\n(tool call: view) {{"path": "flake.nix"}}',
                f"(tool result: view)\n\n{FILE_TEXT}",
                ANSWER,
                f"(compaction summary) {SUMMARY}",
                "(skill nix) Use nix flake check.",
            ],
        )

    def test_reasoning_only_in_the_message_is_shown(self):
        text = self.path.read_text(encoding="utf-8")
        lines = [
            line
            for line in text.splitlines(keepends=True)
            if '"assistant.reasoning"' not in line
        ]
        self.path.write_text("".join(lines), encoding="utf-8")
        self.assertEqual(
            self.messages()[2].text(full=True),
            f"(thinking) {THINKING}\n\n{CHECKING}\n\n"
            '(tool call: view) {"path": "flake.nix"}',
        )

    def test_failed_tools_show_their_error(self):
        with self.path.open("a", encoding="utf-8") as events:
            events.write(
                json.dumps(
                    {
                        "type": "tool.execution_complete",
                        "id": "e1",
                        "timestamp": "2026-11-09T10:05:00.000Z",
                        "data": {
                            "toolCallId": "tc_1",
                            "success": False,
                            "error": {"message": "file not found"},
                        },
                    }
                )
                + "\n"
            )
        self.assertEqual(
            self.messages()[-1].text(full=True),
            "(tool result: view, error)\n\nfile not found",
        )

    def test_head_stops_reading_early(self):
        with self.path.open("a", encoding="utf-8") as events:
            events.write("{invalid json}\n")
        self.assertEqual(
            [m.text() for m in self.messages(roles={"user"}, head=1)], [QUESTION]
        )
        with self.assertRaises(ValueError):
            self.messages()

    def test_there_is_no_current_session(self):
        self.assertIsNone(github_copilot.current_session_id())


if __name__ == "__main__":
    unittest.main()
