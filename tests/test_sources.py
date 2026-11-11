import inspect
import unittest

from remember import cli
from remember.transcripts import SOURCES
from remember.transcripts.base import Part, TranscriptSource, render


class SourceRegistryTests(unittest.TestCase):
    def test_every_source_implements_the_protocol(self):
        members = [
            name
            for name in dir(TranscriptSource)
            if not name.startswith("_") and callable(getattr(TranscriptSource, name))
        ]
        self.assertEqual(
            sorted(members),
            [
                "current_session_id",
                "find_sessions",
                "get_session",
                "list_session_ids",
                "read_messages",
            ],
        )
        for source_name, source in SOURCES.items():
            with self.subTest(source=source_name):
                self.assertIsInstance(source, TranscriptSource)
                self.assertIsInstance(source.LABEL, str)
                for member in members:
                    expected = inspect.signature(getattr(TranscriptSource, member))
                    parameters = list(expected.parameters.values())[1:]  # Drop self.
                    actual = inspect.signature(getattr(source, member))
                    self.assertEqual(
                        [
                            (p.name, p.kind, p.default)
                            for p in actual.parameters.values()
                        ],
                        [(p.name, p.kind, p.default) for p in parameters],
                        member,
                    )

    def test_the_cli_offers_every_source(self):
        self.assertEqual(cli.SOURCES, ("pi", "codex", "claude-code", "copilot"))
        for source_name in SOURCES:
            with self.subTest(source=source_name):
                self.assertEqual(
                    cli.parse_ref(f"{source_name}:01a1ac0c/9c3d2e10"),
                    (source_name, "01a1ac0c", "9c3d2e10"),
                )


class RenderTests(unittest.TestCase):
    parts = (
        Part("tool_result", name="bash", is_error=True),
        Part("thinking", " Plan it. "),
        Part("text", " Done. "),
        Part("tool_call", name="read", arguments={"path": "café.py"}),
        Part("tool_call", name="apply_patch", arguments="*** Begin Patch"),
        Part("image"),
        Part("redacted_thinking"),
        Part("text", "   "),
    )

    def test_default_shows_text_and_images_only(self):
        self.assertEqual(render(self.parts), "Done.\n\n(image)")

    def test_full_shows_every_part(self):
        self.assertEqual(
            render(self.parts, full=True),
            "(tool result: bash, error)\n\n(thinking) Plan it.\n\nDone.\n\n"
            '(tool call: read) {"path": "café.py"}\n\n'
            "(tool call: apply_patch) *** Begin Patch\n\n(image)\n\n(redacted_thinking)",
        )


if __name__ == "__main__":
    unittest.main()
