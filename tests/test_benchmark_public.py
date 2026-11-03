import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_public import benchmark


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / 'weights'
        self.weights.write_bytes(b'weights')
        self.rows = []
        for index in range(2):
            content = str(index).encode()
            (self.root / f'{index}.mp4').write_bytes(content)
            self.rows.append(dict(video=f'{index}.mp4', text='hello world', mode='grid',
                                  speaker=f'speaker-{index}', label_source='human',
                                  sha256=hashlib.sha256(content).hexdigest()))
        self.manifest = self.root / 'manifest.json'

    def save(self):
        self.manifest.write_text(json.dumps(dict(samples=self.rows, purpose='test',
            training_overlap='unknown', speaker_identity_verified=True)))

    def test_english_inference_never_receives_labels_and_failures_count(self):
        self.save()
        with patch('general_bridge.WEIGHTS', self.weights), patch('benchmark_public.recognize_file',
                side_effect=[{'text':'hello world', 'status':'decoded'}, ValueError('bad clip')]) as infer:
            report = benchmark(self.manifest, engine='auto-avsr')
        self.assertEqual(report['wer'], .5)
        self.assertEqual(report['input_failures'], 1)
        self.assertEqual(report['exact_match_rate'], .5)
        self.assertEqual(report['decoder'], 'hybrid_ctc_attention')
        for call in infer.call_args_list:
            self.assertEqual(call.kwargs['mode'], 'mediapipe')
            self.assertIsNone(call.kwargs['weights'])
            self.assertNotIn('text', call.kwargs)

    def test_duplicate_content_rejected(self):
        self.rows[1]['sha256'] = self.rows[0]['sha256']
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            benchmark(self.manifest, self.weights)

    def test_modified_clip_rejected_before_inference(self):
        self.save()
        (self.root / '0.mp4').write_bytes(b'changed')
        with patch('benchmark_public.recognize_file') as infer:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                benchmark(self.manifest, self.weights)
            infer.assert_not_called()
