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

    def test_worker_is_bounded_visual_only_and_uses_argument_list(self):
        def run(command,**options):
            self.assertIsInstance(command,list)
            self.assertEqual(options['timeout'],180)
            self.assertNotIn('shell',options)
            output=Path(command[command.index('--output')+1])
            output.write_text(json.dumps({'engine':'auto-avsr','audio_used':False,'text':'hello'}))
            return SimpleNamespace(returncode=0)
        with patch('general_bridge.available',return_value=True), patch('general_bridge.subprocess.run',side_effect=run):
            self.assertEqual(general_bridge.recognize_general('file with spaces.mp4')['text'],'hello')

    def test_worker_failure_does_not_expose_logs(self):
        with patch('general_bridge.available',return_value=True), patch('general_bridge.subprocess.run',return_value=SimpleNamespace(returncode=1,stderr='private path')):
            with self.assertRaisesRegex(ValueError,'short clip'):
                general_bridge.recognize_general('video.mp4')

    def test_sentence_engine_rejects_prepared_grid_arrays(self):
        from recognition import recognize_file
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.npz'; path.touch()
            with self.assertRaisesRegex(ValueError,'video'):
                recognize_file(path,engine='auto-avsr')


if __name__=='__main__':
    unittest.main()
