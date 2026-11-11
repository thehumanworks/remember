import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from remember.transcripts import pi


class GetSessionTranscriptTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        sessions_dir = Path(temporary_directory.name)
        self.project = '--test-project--'
        self.session = 'session.jsonl'
        project_dir = sessions_dir / self.project
        project_dir.mkdir()
        self.session_path = project_dir / self.session
        patcher = patch.object(pi, 'PI_SESSIONS_DIR', sessions_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events = [
            {'type': 'session', 'id': 'session'},
            {'type': 'model_change', 'id': 'model'},
            *[
                {
                    'type': 'message',
                    'id': role,
                    'message': {'role': role, 'content': f'{role} content'},
                }
                for role in ('system', 'user', 'assistant', 'toolResult')
            ],
        ]
        self.write_events(self.events)

    def write_events(self, events):
        self.session_path.write_text(
            ''.join(json.dumps(event) + '\n' for event in events),
            encoding='utf-8',
        )

    def read_transcript(self, **kwargs):
        return pi.get_session_transcript(self.project, self.session, **kwargs)

    def test_default_includes_all_message_roles(self):
        self.assertEqual(self.read_transcript(), self.events[2:])

    def test_single_message_role(self):
        for event in self.events[2:]:
            role = event['message']['role']
            with self.subTest(role=role):
                self.assertEqual(self.read_transcript(message_roles=role), [event])

    def test_multiple_message_roles(self):
        for roles in (['user', 'assistant'], ('user', 'assistant')):
            with self.subTest(roles=roles):
                transcript = self.read_transcript(message_roles=roles)
                self.assertEqual(transcript, self.events[3:5])

    def test_event_filter_accepts_single_value_or_sequence(self):
        for event_types in ('model_change', ['model_change'], ('model_change',)):
            with self.subTest(event_types=event_types):
                transcript = self.read_transcript(include_only=event_types)
                self.assertEqual(transcript, [self.events[1]])
        transcript = self.read_transcript(include_only=['session', 'model_change'])
        self.assertEqual(transcript, self.events[:2])

    def test_disabled_event_filter_includes_all_events(self):
        for event_types in (None, [], ()):
            with self.subTest(event_types=event_types):
                transcript = self.read_transcript(include_only=event_types)
                self.assertEqual(transcript, self.events)

    def test_disabled_role_filter_preserves_default(self):
        for roles in (None, [], ()):
            with self.subTest(roles=roles):
                transcript = self.read_transcript(message_roles=roles)
                self.assertEqual(transcript, self.events[2:])

    def test_role_filter_excludes_non_message_events(self):
        self.write_events([
            {'type': 'custom', 'id': 'custom', 'message': {'role': 'user'}},
            *self.events,
        ])
        for event_types in (None, [], ['custom', 'message']):
            with self.subTest(event_types=event_types):
                transcript = self.read_transcript(
                    include_only=event_types, message_roles='user'
                )
                self.assertEqual(transcript, [self.events[3]])

    def test_event_and_role_filters_are_combined(self):
        transcript = self.read_transcript(
            include_only='model_change', message_roles='user'
        )
        self.assertEqual(transcript, [])

    def test_head_applies_after_both_filters_and_remains_positional(self):
        transcript = pi.get_session_transcript(
            self.project, self.session, ['message'], 1,
            message_roles=['user', 'assistant'],
        )
        self.assertEqual(transcript, [self.events[3]])
        self.assertEqual(self.read_transcript(message_roles='user', head=0), [])

    def test_head_limits_preserve_slicing_semantics(self):
        for head in (-1, -4, -10, 0, 1, 10):
            with self.subTest(head=head):
                transcript = self.read_transcript(head=head)
                self.assertEqual(transcript, self.events[2:][:head])

    def test_positive_head_stops_before_parsing_remaining_lines(self):
        with self.session_path.open('a', encoding='utf-8') as source:
            source.write('{invalid json}\n')
        transcript = self.read_transcript(message_roles='user', head=1)
        self.assertEqual(transcript, [self.events[3]])
        with self.assertRaises(ValueError):
            self.read_transcript(message_roles='user')

    def test_zero_head_does_not_parse_any_records(self):
        self.session_path.write_text('{invalid json}\n', encoding='utf-8')
        self.assertEqual(self.read_transcript(head=0), [])

    def test_unmatched_role_returns_empty_list(self):
        self.write_events(self.events[:4])
        self.assertEqual(self.read_transcript(message_roles='toolResult'), [])

    def test_role_filter_handles_absent_null_or_invalid_message_fields(self):
        for message in ('absent', None, {'content': 'no role'}, {'role': None}, 'text', []):
            with self.subTest(message=message):
                event = {'type': 'message', 'id': 'message'}
                if message != 'absent':
                    event['message'] = message
                self.write_events([event, self.events[3]])
                transcript = self.read_transcript(message_roles='user')
                self.assertEqual(transcript, [self.events[3]])

    def test_session_without_messages_returns_empty_list(self):
        self.write_events(self.events[:2])
        self.assertEqual(self.read_transcript(), [])
        self.assertEqual(self.read_transcript(message_roles='user'), [])

    def test_preserves_heterogeneous_nested_records(self):
        events = [
            *self.events[:2],
            {
                'type': 'message',
                'id': 'user',
                'message': {'role': 'user', 'content': 'Hello, café ☕'},
                'extra': {'flags': [True, None], 'count': 1},
            },
            {
                'type': 'message',
                'id': 'assistant',
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': 'Checking now.'},
                        {'type': 'toolCall', 'name': 'read', 'arguments': {'path': 'pi.py'}},
                    ],
                },
            },
        ]
        self.session_path.write_text(
            ''.join(json.dumps(event, ensure_ascii=False) + '\n' for event in events),
            encoding='utf-8',
        )
        self.assertEqual(self.read_transcript(include_only=None), events)
        self.assertEqual(self.read_transcript(message_roles='assistant'), [events[-1]])

    def test_message_fields_after_long_metadata_prefix_are_not_lost(self):
        self.write_events([self.events[0]] * 150 + [self.events[3]])
        self.assertEqual(self.read_transcript(message_roles='user'), [self.events[3]])

    def test_empty_or_whitespace_only_file_returns_empty_list(self):
        for content in ('', '\n  \n\t\n'):
            with self.subTest(content=content):
                self.session_path.write_text(content, encoding='utf-8')
                self.assertEqual(self.read_transcript(), [])

    def test_blank_lines_are_ignored(self):
        content = '\n \n' + json.dumps(self.events[3]) + '\n\t\n'
        self.session_path.write_text(content, encoding='utf-8')
        self.assertEqual(self.read_transcript(), [self.events[3]])

    def test_invalid_json_error_identifies_file_and_physical_line(self):
        content = json.dumps(self.events[0]) + '\n\n{invalid json}\n'
        self.session_path.write_text(content, encoding='utf-8')
        with self.assertRaises(ValueError) as raised:
            self.read_transcript()
        self.assertIn('Invalid JSON', str(raised.exception))
        self.assertIn(str(self.session_path), str(raised.exception))
        self.assertIn('line 3', str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, json.JSONDecodeError)

    def test_non_object_records_report_file_and_line(self):
        for record in (None, [], 'text', 1, True):
            with self.subTest(record=record):
                self.write_events([self.events[0], record])
                with self.assertRaises(ValueError) as raised:
                    self.read_transcript()
                self.assertIn('Expected a JSON object', str(raised.exception))
                self.assertIn(str(self.session_path), str(raised.exception))
                self.assertIn('line 2', str(raised.exception))

    def test_missing_session_raises_even_with_zero_head(self):
        self.session_path.unlink()
        for head in (None, 0):
            with self.subTest(head=head):
                with self.assertRaises(FileNotFoundError):
                    self.read_transcript(head=head)

    def test_on_invalid_skips_unreadable_lines(self):
        content = (
            json.dumps(self.events[0]) + '\n'
            + '\x00' * 10 + '\n'
            + '[]\n'
            + json.dumps(self.events[3]) + '\n'
            + '{"type": "message", "id": "partial'
        )
        self.session_path.write_text(content, encoding='utf-8')
        errors = []
        transcript = self.read_transcript(include_only=None, on_invalid=errors.append)
        self.assertEqual(transcript, [self.events[0], self.events[3]])
        self.assertEqual(
            [str(error).split(' at ')[1].split(':')[0] for error in errors],
            ['line 2', 'line 3', 'line 5'],
        )
        self.assertIsInstance(errors[0].__cause__, json.JSONDecodeError)


class GetSessionIdTests(unittest.TestCase):
    def test_returns_the_id_after_the_timestamp(self):
        self.assertEqual(
            pi.get_session_id(
                '2026-11-05T14-10-56-807Z_01a1ac0c-68a7-71a1-a88e-b1e509f41b15.jsonl'
            ),
            '01a1ac0c-68a7-71a1-a88e-b1e509f41b15',
        )
        self.assertEqual(pi.get_session_id('plain.jsonl'), 'plain')


class GetMessageTextTests(unittest.TestCase):
    assistant = {
        'role': 'assistant',
        'content': [
            {'type': 'thinking', 'thinking': ' Plan it. '},
            {'type': 'text', 'text': ' First part. '},
            {'type': 'toolCall', 'id': 'c1', 'name': 'read', 'arguments': {'path': 'café.py'}},
            {'type': 'text', 'text': 'Second part.'},
            {'type': 'image', 'data': 'iVBOR', 'mimeType': 'image/png'},
            {'type': 'unknown'},
            'not a block',
        ],
    }

    def test_default_shows_text_and_images_only(self):
        self.assertEqual(
            pi.get_message_text(self.assistant),
            'First part.\n\nSecond part.\n\n(image)',
        )

    def test_full_shows_thinking_tool_calls_and_other_blocks(self):
        self.assertEqual(
            pi.get_message_text(self.assistant, full=True),
            '(thinking) Plan it.\n\nFirst part.\n\n'
            '(tool call: read) {"path": "café.py"}\n\nSecond part.\n\n(image)\n\n(unknown)',
        )

    def test_tool_only_messages_are_empty_by_default(self):
        message = {
            'role': 'assistant',
            'content': [{'type': 'toolCall', 'name': 'read', 'arguments': {}}],
        }
        self.assertEqual(pi.get_message_text(message), '')

    def test_tool_results_name_the_tool_in_full(self):
        message = {
            'role': 'toolResult',
            'toolName': 'bash',
            'isError': True,
            'content': [{'type': 'text', 'text': 'exit 1'}],
        }
        self.assertEqual(pi.get_message_text(message), 'exit 1')
        self.assertEqual(
            pi.get_message_text(message, full=True),
            '(tool result: bash, error)\n\nexit 1',
        )

    def test_string_and_missing_content(self):
        self.assertEqual(
            pi.get_message_text({'role': 'system', 'content': ' You are pi. '}),
            'You are pi.',
        )
        self.assertEqual(pi.get_message_text({'role': 'user'}), '')


if __name__ == '__main__':
    unittest.main()
