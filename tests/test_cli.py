from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import fixtures
from remember import cli
from remember.transcripts import claude_code, codex, github_copilot, pi

PROJECT = "/work/remember"
OTHER_PROJECT = "/work/other"
OTHER_ID = "019f0000-0000-7000-8000-000000000000"
OLD_ID = "01a1ac0c-68a7-71a1-a88e-b1e509f41b15"
EMPTY_ID = "01a1b000-0000-7000-8000-000000000000"
NEW_ID = "01a1ca88-0c00-77d3-948e-db0ee653f166"
CURRENT_ID = "01a1ca88-1d00-77d3-948e-db0ee653f166"
QUESTION = "Is there value in removing the Polars dependency?"
CLAUDE_ID = "7f3a91c2-0d4e-4b7a-9c61-2f5e8a1b3c4d"
CODEX_ID = "019a4f2e-7c31-7b20-9f4d-3e8a1c6b2d50"
COPILOT_ID = "07b4f288-3402-461c-9749-dd356b0e8cf2"


def text(value):
    return [{"type": "text", "text": value}]


class ParserTests(unittest.TestCase):
    def parse(self, *argv):
        return cli.build_parser().parse_args(argv)

    def assert_usage_error(self, *argv, message):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.parse(*argv)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_command_is_required(self):
        self.assert_usage_error(message="required: COMMAND")

    def test_search_defaults(self):
        args = self.parse("search", "polars", "dependency")
        self.assertEqual(args.command, "search")
        self.assertEqual(args.terms, ["polars", "dependency"])
        self.assertIsNone(args.role)
        self.assertFalse(args.full)
        self.assertEqual(args.limit, 20)
        self.assertIsNone(args.project)
        self.assertFalse(args.all_projects)
        self.assertIsNone(args.source)
        self.assertIsNone(args.since)
        self.assertIsNone(args.until)
        self.assertFalse(args.include_current)

    def test_search_options(self):
        args = self.parse(
            "search",
            "polars dependency",
            "--role",
            "user",
            "--role",
            "assistant",
            "--full",
            "--limit",
            "5",
            "--project",
            "/home/nixos/ai/remember",
            "--source",
            "pi",
            "--source",
            "claude-code",
            "--include-current",
        )
        self.assertEqual(args.terms, ["polars dependency"])
        self.assertEqual(args.role, ["user", "assistant"])
        self.assertTrue(args.full)
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.project, Path("/home/nixos/ai/remember"))
        self.assertEqual(args.source, ["pi", "claude-code"])
        self.assertTrue(args.include_current)

    def test_search_requires_non_blank_terms(self):
        self.assert_usage_error("search", message="required: TERM")
        for term in ("", " ", "\t"):
            with self.subTest(term=term):
                self.assert_usage_error(
                    "search", "polars", term, message="must not be blank"
                )

    def test_sessions_defaults_and_scope(self):
        args = self.parse("sessions")
        self.assertEqual(args.command, "sessions")
        self.assertEqual(args.limit, 10)
        self.assertFalse(args.all_projects)
        args = self.parse("sessions", "--all-projects", "--limit", "3")
        self.assertTrue(args.all_projects)
        self.assertEqual(args.limit, 3)

    def test_sessions_has_no_view_options(self):
        for option in ("--role", "--full"):
            with self.subTest(option=option):
                self.assert_usage_error(
                    "sessions", option, "user", message="unrecognized arguments"
                )

    def test_project_and_all_projects_are_exclusive(self):
        for command in (["search", "polars"], ["sessions"]):
            with self.subTest(command=command[0]):
                self.assert_usage_error(
                    *command,
                    "--project",
                    ".",
                    "--all-projects",
                    message="not allowed with argument --project",
                )

    def test_unknown_values_suggest_the_closest_choice(self):
        self.assert_usage_error("serch", "polars", message="maybe you meant 'search'")
        self.assert_usage_error(
            "sessions", "--source", "claude", message="maybe you meant 'claude-code'"
        )
        self.assert_usage_error(
            "read", "pi:01a1ac0c", "--role", "toolResult", message="invalid choice"
        )

    def test_option_abbreviations_are_rejected(self):
        self.assert_usage_error("sessions", "--all", message="unrecognized arguments")
        self.assert_usage_error(
            "read", "pi:01a1ac0c", "--max", "0", message="unrecognized arguments"
        )

    def test_read_session_ref_defaults(self):
        args = self.parse("read", "pi:01a1ac0c")
        self.assertEqual(args.command, "read")
        self.assertEqual(args.ref, ("pi", "01a1ac0c", None))
        self.assertEqual((args.before, args.after, args.max_chars), (3, 10, 2000))
        self.assertIsNone(args.role)
        self.assertFalse(args.full)

    def test_read_message_ref_and_window(self):
        args = self.parse(
            "read",
            "claude-code:6f2a9b01-4c1d/9c3d2e10",
            "-B",
            "0",
            "-A",
            "0",
            "--max-chars",
            "0",
            "--role",
            "user",
            "--full",
        )
        self.assertEqual(args.ref, ("claude-code", "6f2a9b01-4c1d", "9c3d2e10"))
        self.assertEqual((args.before, args.after, args.max_chars), (0, 0, 0))
        self.assertEqual(args.role, ["user"])
        self.assertTrue(args.full)

    def test_read_requires_an_explicit_ref(self):
        """read never chooses a session itself; see CommandTests for why."""
        self.assert_usage_error("read", message="required: REF")
        self.assert_usage_error("read", "--full", message="required: REF")
        for ref in ("last", "latest"):
            with self.subTest(ref=ref):
                self.assert_usage_error("read", ref, message="invalid ref")

    def test_read_has_no_scope_options(self):
        self.assert_usage_error(
            "read", "pi:01a1ac0c", "--all-projects", message="unrecognized arguments"
        )

    def test_invalid_refs(self):
        for ref in (
            "pi",
            "pi:",
            "01a1ac0c",
            "claude:01a1ac0c",
            "pi:01a1ac0c/",
            "pi:/9c3d2e10",
            "pi:01a1ac0c/9c3d2e10/extra",
            "pi:01a1 ac0c",
        ):
            with self.subTest(ref=ref):
                self.assert_usage_error("read", ref, message="invalid ref")

    def test_counts_are_validated(self):
        # Values are attached to their options so negative numbers parse as values.
        for argv, option, minimum in (
            (["search", "polars"], "--limit=", 1),
            (["sessions"], "--limit=", 1),
            (["read", "pi:01a1ac0c"], "-B", 0),
            (["read", "pi:01a1ac0c"], "-A", 0),
            (["read", "pi:01a1ac0c"], "--max-chars=", 0),
        ):
            for value in (str(minimum - 1), "1.5", "many"):
                with self.subTest(option=option, value=value):
                    self.assert_usage_error(
                        *argv,
                        option + value,
                        message=f"expected a whole number of at least {minimum}",
                    )

    def test_relative_times(self):
        for value, offset in (("7d", timedelta(days=7)), ("12h", timedelta(hours=12))):
            with self.subTest(value=value):
                expected = datetime.now().astimezone() - offset
                args = self.parse("sessions", "--since", value, "--until", value)
                for when in (args.since, args.until):
                    self.assertIsNotNone(when.tzinfo)
                    self.assertLess(abs(when - expected), timedelta(seconds=5))

    def test_absolute_times(self):
        args = self.parse("sessions", "--since", "2026-11-01")
        self.assertEqual(args.since, datetime(2026, 11, 1).astimezone())
        self.assertIsNotNone(args.since.tzinfo)
        args = self.parse("sessions", "--until", "2026-11-11T12:15:11.755Z")
        self.assertEqual(
            args.until, datetime(2026, 11, 11, 12, 15, 11, 755000, tzinfo=UTC)
        )

    def test_invalid_times(self):
        for value in ("yesterday", "7", "7w", "-7d", "2026-13-01"):
            with self.subTest(value=value):
                self.assert_usage_error(
                    "sessions", f"--since={value}", message="invalid time"
                )

    def test_help_documents_refs_for_every_command(self):
        parser = cli.build_parser()
        self.assertIn("pi:01a1ac0c/9c3d2e10", parser.format_help())
        for command in ("search", "sessions", "read"):
            with self.subTest(command=command):
                stdout = StringIO()
                with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    parser.parse_args([command, "--help"])
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("pi:01a1ac0c/9c3d2e10", stdout.getvalue())
                self.assertNotIn("\x1b[", stdout.getvalue())


class ShortIdTests(unittest.TestCase):
    def test_ids_use_at_least_eight_characters(self):
        shorts = cli.short_ids([OLD_ID, OTHER_ID, "01a1ac0d-0000"])
        self.assertEqual(shorts[OTHER_ID], "019f0000")
        self.assertEqual(shorts[OLD_ID], "01a1ac0c")

    def test_ids_sharing_a_prefix_are_extended_past_the_separator(self):
        shorts = cli.short_ids([NEW_ID, CURRENT_ID, OLD_ID])
        self.assertEqual(shorts[NEW_ID], "01a1ca88-0")
        self.assertEqual(shorts[CURRENT_ID], "01a1ca88-1")
        self.assertEqual(shorts[OLD_ID], "01a1ac0c")


class SnippetTests(unittest.TestCase):
    def test_short_text_is_one_line(self):
        self.assertEqual(
            cli.snippet("Removed\n  the Polars\tdependency.", ["polars"]),
            "Removed the Polars dependency.",
        )

    def test_long_text_is_cut_around_the_first_match(self):
        value = "a " * 200 + "Polars matters here. " + "b " * 200
        snippet = cli.snippet(value, ["polars"])
        self.assertTrue(snippet.startswith("…"))
        self.assertTrue(snippet.endswith("…"))
        self.assertIn("Polars matters here.", snippet)
        self.assertEqual(len(snippet), cli.SNIPPET_CHARS + 2)


class CommandTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        sessions_dir = patch.object(pi, "PI_SESSIONS_DIR", self.root)
        sessions_dir.start()
        self.addCleanup(sessions_dir.stop)
        # Other agents read from their own empty directories, never the real ones.
        other_agents = TemporaryDirectory()
        self.addCleanup(other_agents.cleanup)
        self.agents = Path(other_agents.name)
        for module, attribute in (
            (claude_code, "CLAUDE_PROJECTS_DIR"),
            (codex, "CODEX_HOME"),
            (github_copilot, "COPILOT_SESSIONS_DIR"),
        ):
            directory = patch.object(module, attribute, self.agents / module.__name__)
            directory.start()
            self.addCleanup(directory.stop)
        # Restore the local time zone after the environment is restored.
        self.addCleanup(time.tzset)
        environment = patch.dict(os.environ, {"TZ": "UTC", "PI_SESSION_ID": CURRENT_ID})
        environment.start()
        self.addCleanup(environment.stop)
        for name in ("CLAUDE_CODE_SESSION_ID", "CODEX_THREAD_ID"):
            os.environ.pop(name, None)
        time.tzset()

        self.old_path = self.write_session(
            PROJECT,
            OLD_ID,
            datetime(2026, 11, 5, 14, 10),
            [
                {"role": "system", "content": "You are pi."},
                {"role": "user", "content": text(QUESTION)},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Check where polars is used"},
                        *text("Yes. Polars is only used to read JSONL."),
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "read",
                            "arguments": {"path": "pi.py"},
                        },
                    ],
                },
                {
                    "role": "toolResult",
                    "toolName": "read",
                    "isError": False,
                    "content": text("import polars as pl"),
                },
                {
                    "role": "assistant",
                    "content": text("Removed the Polars dependency."),
                },
                {"role": "user", "content": text("Thanks.")},
            ],
        )
        self.write_session(
            PROJECT,
            NEW_ID,
            datetime(2026, 11, 11, 12, 14),
            [
                {"role": "user", "content": text("Define the CLI usage.")},
                {
                    "role": "assistant",
                    "content": text("Three commands: search, sessions and read."),
                },
            ],
        )
        self.write_session(
            PROJECT,
            CURRENT_ID,
            datetime(2026, 11, 11, 15, 0),
            [
                {
                    "role": "user",
                    "content": text("Polars question from the current session."),
                },
            ],
        )
        self.write_session(PROJECT, EMPTY_ID, datetime(2026, 11, 10, 9, 0), [])
        self.write_session(
            OTHER_PROJECT,
            OTHER_ID,
            datetime(2026, 11, 8, 10, 0),
            [
                {"role": "user", "content": text("Fix the flake check.")},
                {"role": "assistant", "content": text("Polars is not used here.")},
            ],
        )

    def write_session(self, cwd, session_id, start, messages):
        """Write a session with a message every minute after start (UTC)."""
        start = start.replace(tzinfo=UTC)
        project_dir = self.root / pi.get_project_from_path(cwd)
        project_dir.mkdir(exist_ok=True)
        path = project_dir / f"{start:%Y-%m-%dT%H-%M-%S-000Z}_{session_id}.jsonl"
        events = [
            {
                "type": "session",
                "version": 3,
                "id": session_id,
                "timestamp": f"{start:%Y-%m-%dT%H:%M:%S.000Z}",
                "cwd": cwd,
            },
            {"type": "model_change", "id": f"{session_id[4:8]}0000", "parentId": None},
        ]
        for index, message in enumerate(messages, start=1):
            when = start + timedelta(minutes=index)
            events.append(
                {
                    "type": "message",
                    "id": f"{session_id[4:8]}{index:04d}",
                    "parentId": events[-1]["id"],
                    "timestamp": f"{when:%Y-%m-%dT%H:%M:%S.000Z}",
                    "message": message,
                }
            )
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        end = (start + timedelta(minutes=len(messages))).timestamp()
        os.utime(path, (end, end))
        return path

    def run_cli(self, *argv):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def assert_output(self, argv, expected):
        code, stdout, stderr = self.run_cli(*argv)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(stdout, expected)

    def assert_error(self, argv, message):
        code, stdout, stderr = self.run_cli(*argv)
        self.assertEqual((code, stdout), (2, ""))
        self.assertIn(message, stderr)
        return stderr

    # sessions

    def test_sessions_lists_requests_newest_first(self):
        self.assert_output(
            ["sessions", "--project", PROJECT],
            'pi:01a1ca88-0  2026-11-11 12:14–12:16  "Define the CLI usage."\n'
            f'pi:01a1ac0c  2026-11-05 14:10–14:17  "{QUESTION}"\n',
        )

    def test_sessions_defaults_to_the_current_directory(self):
        with patch.object(Path, "cwd", return_value=Path(PROJECT)):
            code, stdout, _ = self.run_cli("sessions")
        self.assertEqual(code, 0)
        self.assertEqual(stdout.splitlines()[0].split()[0], "pi:01a1ca88-0")

    def test_sessions_can_include_the_current_session(self):
        code, stdout, _ = self.run_cli(
            "sessions", "--project", PROJECT, "--include-current"
        )
        self.assertEqual(
            stdout.splitlines()[0],
            'pi:01a1ca88-1  2026-11-11 15:00–15:01  "Polars question from the current session."',
        )

    def test_sessions_across_projects_show_the_project(self):
        self.assert_output(
            ["sessions", "--all-projects"],
            'pi:01a1ca88-0  2026-11-11 12:14–12:16  /work/remember  "Define the CLI usage."\n'
            'pi:019f0000  2026-11-08 10:00–10:02  /work/other  "Fix the flake check."\n'
            f'pi:01a1ac0c  2026-11-05 14:10–14:17  /work/remember  "{QUESTION}"\n',
        )

    def test_sessions_use_the_recorded_directory_not_the_storage_name(self):
        # /work-remember and /work/remember share a storage directory name.
        self.write_session(
            "/work-remember",
            "01a1d000-0000-7000-8000-000000000000",
            datetime(2026, 11, 9, 8, 0),
            [{"role": "user", "content": text("Hi")}],
        )
        code, stdout, _ = self.run_cli("sessions", "--project", PROJECT)
        self.assertNotIn("pi:01a1d000", stdout)
        code, stdout, _ = self.run_cli("sessions", "--project", "/work-remember")
        self.assertEqual(stdout, 'pi:01a1d000  2026-11-09 08:00–08:01  "Hi"\n')

    def test_sessions_limit_reports_the_total(self):
        self.assert_output(
            ["sessions", "--project", PROJECT, "--limit", "1"],
            'pi:01a1ca88-0  2026-11-11 12:14–12:16  "Define the CLI usage."\n'
            "\nShowing 1 of 2 sessions. Rerun with --limit 2 to see all.\n",
        )

    def test_sessions_filter_by_activity(self):
        for argv, expected in (
            (["--since", "2026-11-10"], ["pi:01a1ca88-0"]),
            (["--until", "2026-11-10"], ["pi:01a1ac0c"]),
            # A session counts as active through its last message.
            (["--since", "2026-11-05T14:17", "--until", "2026-11-06"], ["pi:01a1ac0c"]),
        ):
            with self.subTest(argv=argv):
                code, stdout, _ = self.run_cli("sessions", "--project", PROJECT, *argv)
                self.assertEqual(
                    [line.split()[0] for line in stdout.splitlines()], expected
                )

    def test_sessions_without_results_suggest_a_wider_scope(self):
        self.assert_output(
            ["sessions", "--project", "/work/none"],
            "No sessions in /work/none. Try --all-projects.\n",
        )
        self.assert_output(
            ["sessions", "--project", PROJECT, "--since", "2026-12-01"],
            "No sessions in /work/remember. "
            "Try a wider --since or --until or --all-projects.\n",
        )

    def test_sessions_without_a_sessions_directory(self):
        with patch.object(pi, "PI_SESSIONS_DIR", self.root / "missing"):
            self.assert_output(["sessions", "--all-projects"], "No sessions.\n")

    # other agents

    def write_other_agents(self):
        """The flake question in Claude Code, Codex and Copilot, an hour apart."""
        fixtures.claude_session(
            self.agents / claude_code.__name__,
            PROJECT,
            CLAUDE_ID,
            datetime(2026, 11, 9, 10, 0),
        )
        fixtures.codex_session(
            self.agents / codex.__name__,
            PROJECT,
            CODEX_ID,
            datetime(2026, 11, 9, 11, 0),
        )
        fixtures.copilot_session(
            self.agents / github_copilot.__name__,
            PROJECT,
            COPILOT_ID,
            datetime(2026, 11, 9, 12, 0),
        )

    def test_sessions_from_every_agent_newest_first(self):
        self.write_other_agents()
        flake = f'"{fixtures.QUESTION}"'
        self.assert_output(
            ["sessions", "--project", PROJECT],
            'pi:01a1ca88-0  2026-11-11 12:14–12:16  "Define the CLI usage."\n'
            f"copilot:07b4f288  2026-11-09 12:00–12:04  {flake}\n"
            f"codex:019a4f2e  2026-11-09 11:00–11:04  {flake}\n"
            f"claude-code:7f3a91c2  2026-11-09 10:00–10:05  {flake}\n"
            f'pi:01a1ac0c  2026-11-05 14:10–14:17  "{QUESTION}"\n',
        )

    def test_sessions_by_source(self):
        self.write_other_agents()
        _, stdout, _ = self.run_cli(
            "sessions", "--project", PROJECT, "--source", "codex", "--source", "copilot"
        )
        self.assertEqual(
            [line.split()[0] for line in stdout.splitlines()],
            ["copilot:07b4f288", "codex:019a4f2e"],
        )

    def test_the_current_session_of_each_agent_is_excluded(self):
        self.write_other_agents()
        os.environ["CLAUDE_CODE_SESSION_ID"] = CLAUDE_ID
        os.environ["CODEX_THREAD_ID"] = CODEX_ID
        _, stdout, _ = self.run_cli("sessions", "--project", PROJECT)
        refs = [line.split()[0] for line in stdout.splitlines()]
        self.assertNotIn("claude-code:7f3a91c2", refs)
        self.assertNotIn("codex:019a4f2e", refs)
        self.assertIn("copilot:07b4f288", refs)
        _, stdout, _ = self.run_cli(
            "sessions", "--project", PROJECT, "--include-current"
        )
        refs = [line.split()[0] for line in stdout.splitlines()]
        self.assertIn("claude-code:7f3a91c2", refs)
        self.assertIn("codex:019a4f2e", refs)

    def test_read_a_claude_code_session(self):
        self.write_other_agents()
        self.assert_output(
            ["read", "claude-code:7f3a"],
            "claude-code:7f3a91c2 · /work/remember · 2026-11-09 10:00–10:05 · 3 messages\n"
            "\n[user · claude-code:7f3a91c2/7f3a0001 · 10:01]\n"
            f"{fixtures.QUESTION}\n"
            "\n[assistant · claude-code:7f3a91c2/7f3a0002 · 10:02]\n"
            f"{fixtures.CHECKING}\n"
            "\n[assistant · claude-code:7f3a91c2/7f3a0007 · 10:05]\n"
            f"{fixtures.ANSWER}\n"
            "\nmessages 1–3 of 3\n",
        )

    def test_read_around_a_codex_message(self):
        self.write_other_agents()
        code, stdout, _ = self.run_cli("read", "codex:019a4f2e/6", "-B", "0", "-A", "1")
        self.assertEqual(code, 0)
        self.assertEqual(
            [line for line in stdout.splitlines() if line.startswith("[")],
            [
                "[user · codex:019a4f2e/6 · 11:01]",
                "[assistant · codex:019a4f2e/11 · 11:04]",
            ],
        )

    def test_read_a_copilot_session_in_full(self):
        self.write_other_agents()
        code, stdout, _ = self.run_cli("read", "copilot:07b4f288", "--full")
        self.assertEqual(code, 0)
        self.assertIn(" · 6 messages\n", stdout)
        self.assertIn(
            "\n[toolResult · copilot:07b4f288/07b40009 · 12:03]\n"
            f"(tool result: view)\n\n{fixtures.FILE_TEXT}\n",
            stdout,
        )

    def test_search_across_agents(self):
        self.write_other_agents()
        _, stdout, _ = self.run_cli("search", "flake check", "--project", PROJECT)
        self.assertEqual(
            [line.split()[0] for line in stdout.splitlines() if line[:1].isalpha()],
            ["copilot:07b4f288", "codex:019a4f2e", "claude-code:7f3a91c2"],
        )
        self.assertTrue(
            stdout.endswith(
                "6 matches in 3 sessions · "
                "context: remember read copilot:07b4f288/07b40003\n"
            )
        )

    def test_unknown_sessions_name_the_agent(self):
        self.assert_error(
            ["read", "claude-code:ffffffff"],
            "remember: no Claude Code session matches claude-code:ffffffff; "
            "find refs with: remember sessions --all-projects\n",
        )

    # read

    def test_read_session_shows_the_conversation(self):
        self.assert_output(
            ["read", "pi:01a1ac0c"],
            "pi:01a1ac0c · /work/remember · 2026-11-05 14:10–14:17 · 4 messages\n"
            "\n[user · pi:01a1ac0c/ac0c0002 · 14:12]\n"
            f"{QUESTION}\n"
            "\n[assistant · pi:01a1ac0c/ac0c0003 · 14:13]\n"
            "Yes. Polars is only used to read JSONL.\n"
            "\n[assistant · pi:01a1ac0c/ac0c0006 · 14:16]\n"
            "Removed the Polars dependency.\n"
            "\n[user · pi:01a1ac0c/ac0c0007 · 14:17]\n"
            "Thanks.\n"
            "\nmessages 1–4 of 4\n",
        )

    def test_read_window_around_a_message_links_to_both_sides(self):
        code, stdout, _ = self.run_cli(
            "read", "pi:01a1ac0c/ac0c0006", "-B", "1", "-A", "0"
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            [line for line in stdout.splitlines() if line.startswith("[")],
            [
                "[assistant · pi:01a1ac0c/ac0c0003 · 14:13]",
                "[assistant · pi:01a1ac0c/ac0c0006 · 14:16]",
            ],
        )
        self.assertEqual(
            stdout.splitlines()[-1],
            "messages 2–3 of 4 · earlier: remember read pi:01a1ac0c/ac0c0002 -B 10 -A 0"
            " · later: remember read pi:01a1ac0c/ac0c0007 -B 0",
        )

    def test_read_session_pages_from_the_start(self):
        code, stdout, _ = self.run_cli("read", "pi:01a1ac0c", "-A", "1")
        self.assertEqual(
            stdout.splitlines()[-1],
            "messages 1–2 of 4 · later: remember read pi:01a1ac0c/ac0c0006 -B 0",
        )

    def test_read_anchors_hidden_messages_at_their_position(self):
        # ac0c0005 is a tool result, which the default view hides.
        code, stdout, _ = self.run_cli(
            "read", "pi:01a1ac0c/ac0c0005", "-B", "1", "-A", "0"
        )
        self.assertEqual(code, 0)
        self.assertIn("[assistant · pi:01a1ac0c/ac0c0003 · 14:13]", stdout)
        self.assertIn("[assistant · pi:01a1ac0c/ac0c0006 · 14:16]", stdout)
        self.assertNotIn("ac0c0005 ·", stdout)

    def test_read_full_shows_everything(self):
        code, stdout, _ = self.run_cli(
            "read", "pi:01a1ac0c/ac0c0003", "--full", "-B", "1", "-A", "2"
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            stdout,
            "pi:01a1ac0c · /work/remember · 2026-11-05 14:10–14:17 · 7 messages\n"
            "\n[user · pi:01a1ac0c/ac0c0002 · 14:12]\n"
            f"{QUESTION}\n"
            "\n[assistant · pi:01a1ac0c/ac0c0003 · 14:13]\n"
            "(thinking) Check where polars is used\n\nYes. Polars is only used to read JSONL.\n"
            "\n[assistant · pi:01a1ac0c/ac0c0004 · 14:14]\n"
            '(tool call: read) {"path": "pi.py"}\n'
            "\n[toolResult · pi:01a1ac0c/ac0c0005 · 14:15]\n"
            "(tool result: read)\n\nimport polars as pl\n"
            "\nmessages 2–5 of 7 · earlier: remember read pi:01a1ac0c/ac0c0001 -B 10 -A 0 --full"
            " · later: remember read pi:01a1ac0c/ac0c0006 -B 0 --full\n",
        )

    def test_read_role_filter_is_kept_when_paging(self):
        code, stdout, _ = self.run_cli(
            "read", "pi:01a1ac0c", "--role", "user", "-A", "0"
        )
        self.assertIn(" · 2 messages\n", stdout)
        self.assertEqual(
            stdout.splitlines()[-1],
            "message 1 of 2 · later: remember read pi:01a1ac0c/ac0c0007 -B 0 --role user",
        )

    def test_read_truncates_long_messages_with_a_way_to_see_more(self):
        code, stdout, _ = self.run_cli(
            "read", "pi:01a1ac0c/ac0c0002", "-B", "0", "-A", "0", "--max-chars", "12"
        )
        self.assertIn(
            "\nIs there val\n"
            f"[{len(QUESTION) - 12} more chars · "
            "remember read pi:01a1ac0c/ac0c0002 -B 0 -A 0 --max-chars 0]\n",
            stdout,
        )
        self.assertEqual(
            stdout.splitlines()[-1],
            "message 1 of 4 · later: remember read pi:01a1ac0c/ac0c0003 -B 0 --max-chars 12",
        )

    def test_read_accepts_prefixes_and_the_current_session(self):
        code, stdout, _ = self.run_cli(
            "read", "pi:01a1ac/ac0c0006", "-B", "0", "-A", "0"
        )
        self.assertIn("[assistant · pi:01a1ac0c/ac0c0006 · 14:16]", stdout)
        code, stdout, _ = self.run_cli("read", "pi:01a1ca88-1")
        self.assertIn("Polars question from the current session.", stdout)

    def test_read_has_no_latest_session_shortcut(self):
        """Decision (2026-11-11): read takes an explicit session ref only.

        A stateless CLI cannot reliably tell which session is running it.
        One conversation can have transcripts in two sources: the Pi
        claude-bridge provider also writes a Claude Code transcript. Several
        agents can also run in one project at once. A shortcut such as "last"
        could therefore return the caller's own conversation. Agents find
        refs with `remember sessions` and choose one themselves.
        """
        for ref in ("pi:last", "pi:latest"):
            with self.subTest(ref=ref):
                stderr = self.assert_error(
                    ["read", ref], f"no Pi session matches {ref}"
                )
                self.assertIn("remember sessions", stderr)

    def test_read_session_without_messages(self):
        self.assert_output(
            ["read", "pi:01a1b000"],
            "pi:01a1b000 · /work/remember · 2026-11-10 09:00–09:00 · 0 messages\n"
            "\nNo messages in this view. Try --full.\n",
        )

    def test_read_reference_errors(self):
        self.assert_error(
            ["read", "pi:ffffffff"],
            "remember: no Pi session matches pi:ffffffff; "
            "find refs with: remember sessions --all-projects\n",
        )
        stderr = self.assert_error(
            ["read", "pi:01a1ca88"], "pi:01a1ca88 matches 2 sessions"
        )
        self.assertIn(
            "  pi:01a1ca88-1  2026-11-11 15:00–15:01  /work/remember\n", stderr
        )
        self.assertIn(
            "  pi:01a1ca88-0  2026-11-11 12:14–12:16  /work/remember\n", stderr
        )
        self.assert_error(
            ["read", "pi:01a1ac0c/ffff"],
            "remember: no message ffff in pi:01a1ac0c; "
            "read the session with: remember read pi:01a1ac0c\n",
        )
        self.assert_error(
            ["read", "pi:01a1ac0c/ac0c000"], "pi:01a1ac0c/ac0c000 matches 7 messages"
        )

    def test_read_skips_unreadable_lines_with_a_warning(self):
        lines = self.old_path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines.insert(4, "\x00" * 20 + "\n")
        self.old_path.write_text("".join(lines), encoding="utf-8")
        code, stdout, stderr = self.run_cli("read", "pi:01a1ac0c")
        self.assertEqual(code, 0)
        self.assertIn(" · 4 messages\n", stdout)
        self.assertRegex(
            stderr,
            r"^remember: warning: Invalid JSON in .* at line 5: .* \(line skipped\)\n$",
        )

    # search

    def test_search_groups_matches_by_session(self):
        self.assert_output(
            ["search", "polars", "--project", PROJECT],
            f'pi:01a1ac0c  2026-11-05 14:10–14:17  "{QUESTION}"\n'
            f"  pi:01a1ac0c/ac0c0002  14:12  user       {QUESTION}\n"
            "  pi:01a1ac0c/ac0c0003  14:13  assistant  Yes. Polars is only used to read JSONL.\n"
            "  pi:01a1ac0c/ac0c0006  14:16  assistant  Removed the Polars dependency.\n"
            "\n3 matches in 1 session · context: remember read pi:01a1ac0c/ac0c0002\n",
        )

    def test_search_needs_every_term_ignoring_case_and_spacing(self):
        for terms, expected in (
            (["POLARS   dependency"], ["ac0c0002", "ac0c0006"]),
            (["polars", "removed"], ["ac0c0006"]),
            (["polars", "nowhere"], []),
        ):
            with self.subTest(terms=terms):
                code, stdout, _ = self.run_cli("search", *terms, "--project", PROJECT)
                found = [
                    line.split()[0].rsplit("/", 1)[1]
                    for line in stdout.splitlines()
                    if line.startswith("  pi:")
                ]
                self.assertEqual(found, expected)

    def test_search_full_includes_tool_output(self):
        code, stdout, _ = self.run_cli(
            "search", "import polars", "--project", PROJECT, "--full"
        )
        self.assertIn(
            "  pi:01a1ac0c/ac0c0005  14:15  toolResult (tool result: read) import polars as pl\n",
            stdout,
        )
        self.assertTrue(
            stdout.endswith("context: remember read pi:01a1ac0c/ac0c0005 --full\n")
        )

    def test_search_by_role_and_including_the_current_session(self):
        code, stdout, _ = self.run_cli(
            "search",
            "polars",
            "--project",
            PROJECT,
            "--role",
            "user",
            "--include-current",
        )
        self.assertEqual(
            [
                line.split()[0]
                for line in stdout.splitlines()
                if line.startswith("  pi:")
            ],
            ["pi:01a1ca88-1/ca880001", "pi:01a1ac0c/ac0c0002"],
        )

    def test_search_across_projects(self):
        code, stdout, _ = self.run_cli(
            "search", "polars", "--all-projects", "--limit", "2"
        )
        self.assertEqual(
            stdout,
            'pi:019f0000  2026-11-08 10:00–10:02  /work/other  "Fix the flake check."\n'
            "  pi:019f0000/00000002  10:02  assistant  Polars is not used here.\n"
            "\n"
            f'pi:01a1ac0c  2026-11-05 14:10–14:17  /work/remember  "{QUESTION}"\n'
            f"  pi:01a1ac0c/ac0c0002  14:12  user       {QUESTION}\n"
            "\nShowing 2 of 4 matches in 2 sessions. Narrow the search or rerun with --limit 4"
            " · context: remember read pi:019f0000/00000002\n",
        )

    def test_search_without_matches_suggests_next_steps(self):
        self.assert_output(
            ["search", "nowhere", "--project", PROJECT],
            "No matches in /work/remember. Try fewer terms, --full or --all-projects.\n",
        )
        self.assert_output(
            ["search", "nowhere", "--all-projects", "--full"],
            "No matches. Try fewer terms.\n",
        )


if __name__ == "__main__":
    unittest.main()
