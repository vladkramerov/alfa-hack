import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import report_sprint


class ReportPublicationTests(unittest.TestCase):
    def publish(self, paths):
        helper = getattr(report_sprint, 'publish_pair', None)
        self.assertIsNotNone(helper, 'Report publication needs an isolated, recoverable helper')
        helper(paths, ['# Synthetic report\n', json.dumps({'synthetic': True}) + '\n'])

    def outputs(self, directory):
        return [Path(directory) / 'REPORT.md', Path(directory) / 'summary.json']

    def assert_complete(self, paths):
        self.assertEqual(paths[0].read_text(encoding='utf8'), '# Synthetic report\n')
        self.assertEqual(json.loads(paths[1].read_text(encoding='utf8')), {'synthetic': True})
        self.assertEqual(set(paths[0].parent.iterdir()), set(paths))

    def test_second_create_failure_cleans_outputs_and_allows_retry(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            paths = self.outputs(directory)
            original = report_sprint.os.open
            creations = 0

            def failing_open(*args, **kwargs):
                nonlocal creations
                creations += 1
                if creations == 2:
                    raise OSError('injected second create failure')
                return original(*args, **kwargs)

            with patch.object(report_sprint.os, 'open', side_effect=failing_open):
                with self.assertRaisesRegex(OSError, 'injected second create'):
                    self.publish(paths)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.publish(paths)
            self.assert_complete(paths)

    def test_second_write_failure_closes_fds_cleans_outputs_and_allows_retry(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            paths = self.outputs(directory)
            original = report_sprint.os.fdopen
            streams = []

            class WriteFailure:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    self.stream.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def write(self, value):
                    raise OSError('injected second write failure')

            def failing_fdopen(*args, **kwargs):
                stream = original(*args, **kwargs)
                streams.append(stream)
                return WriteFailure(stream) if len(streams) == 2 else stream

            with patch.object(report_sprint.os, 'fdopen', side_effect=failing_fdopen):
                with self.assertRaisesRegex(OSError, 'injected second write'):
                    self.publish(paths)
            self.assertTrue(all(stream.closed for stream in streams))
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.publish(paths)
            self.assert_complete(paths)

    def test_success_only_publishes_complete_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            paths = self.outputs(directory)
            original = os.link
            snapshots = []

            def inspect_publish(source, destination):
                snapshots.append(Path(source).read_bytes())
                return original(source, destination)

            with patch.object(report_sprint.os, 'link', side_effect=inspect_publish):
                self.publish(paths)
            self.assertEqual(snapshots, [b'# Synthetic report\n', b'{"synthetic": true}\n'])
            self.assert_complete(paths)

    def test_second_publish_collision_preserves_existing_file_and_rolls_back_own(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            paths = self.outputs(directory)
            original = os.link

            def collide(source, destination):
                if Path(destination) == paths[1]:
                    paths[1].write_bytes(b'existing artifact')
                return original(source, destination)

            with patch.object(report_sprint.os, 'link', side_effect=collide):
                with self.assertRaises(FileExistsError):
                    self.publish(paths)
            self.assertFalse(paths[0].exists())
            self.assertEqual(paths[1].read_bytes(), b'existing artifact')
            self.assertEqual(list(Path(directory).iterdir()), [paths[1]])

    def test_existing_pair_never_changes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            paths = self.outputs(directory)
            self.publish(paths)
            before = [path.read_bytes() for path in paths]
            with self.assertRaises(FileExistsError):
                self.publish(paths)
            self.assertEqual([path.read_bytes() for path in paths], before)
            self.assert_complete(paths)

    def test_encoding_failure_happens_before_any_file_creation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            helper = getattr(report_sprint, 'publish_pair', None)
            self.assertIsNotNone(helper)
            with self.assertRaises(UnicodeEncodeError):
                helper(self.outputs(directory), ['valid', '\ud800'])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
