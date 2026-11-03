import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import general_bridge


class GeneralBridgeTests(unittest.TestCase):
    def test_missing_runtime_is_explicit(self):
        with patch('general_bridge.available',return_value=False):
            with self.assertRaises(FileNotFoundError):
                general_bridge.recognize_general('video.mp4')

    def setUp(self):
        general_bridge._stop_worker()
        self.addCleanup(general_bridge._stop_worker)

    def fake_start(self, responses):
        import queue
        from unittest.mock import MagicMock
        process = MagicMock()
        process.poll.return_value = None
        general_bridge._worker = process
        general_bridge._responses = queue.Queue()
        for response in responses:
            general_bridge._responses.put(response)
        return process

    def test_worker_reused_and_requests_keep_their_paths(self):
        reply = json.dumps({'result': {'engine':'auto-avsr','audio_used':False,'text':'hello'}})
        def start():
            self.fake_start([reply, reply])
        with patch('general_bridge.available', return_value=True), patch('general_bridge._start_worker', side_effect=start) as launch:
            for name in ['file with spaces.mp4', 'second.mp4']:
                self.assertEqual(general_bridge.recognize_general(name)['text'], 'hello')
            self.assertEqual(launch.call_count, 1)
            calls = general_bridge._worker.stdin.write.call_args_list
            self.assertTrue(json.loads(calls[0].args[0])['path'].endswith('file with spaces.mp4'))
            self.assertTrue(json.loads(calls[1].args[0])['path'].endswith('second.mp4'))

    def test_timeout_kills_worker_and_next_request_restarts(self):
        process = self.fake_start([])
        with patch('general_bridge.available', return_value=True), patch('general_bridge.TIMEOUT', 0.01):
            with self.assertRaisesRegex(ValueError, 'timed out'):
                general_bridge.recognize_general('video.mp4')
            process.kill.assert_called_once()
            self.assertIsNone(general_bridge._worker)
            reply = json.dumps({'result': {'engine':'auto-avsr','audio_used':False}})
            with patch('general_bridge._start_worker', side_effect=lambda: self.fake_start([reply])) as launch:
                general_bridge.recognize_general('next.mp4')
                launch.assert_called_once()

    def test_failure_and_invalid_responses_do_not_expose_logs(self):
        for reply in [None, 'private path', json.dumps({'error':'private path'}),
                      json.dumps({'result':{'engine':'auto-avsr','audio_used':True}})]:
            process = self.fake_start([reply])
            with patch('general_bridge.available', return_value=True):
                with self.assertRaisesRegex(ValueError, 'short clip') as error:
                    general_bridge.recognize_general('video.mp4')
            self.assertNotIn('private path', str(error.exception))
            process.kill.assert_called_once()

    def test_sentence_engine_rejects_prepared_grid_arrays(self):
        from recognition import recognize_file
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.npz'; path.touch()
            with self.assertRaisesRegex(ValueError,'video'):
                recognize_file(path,engine='auto-avsr')


if __name__=='__main__':
    unittest.main()
