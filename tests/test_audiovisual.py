import io
import tempfile
import unittest
import wave
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from audiovisual import lip_expression, recognize_audiovisual, read_audio, attach_lip_expressions
from service import create_app


class AudioVisualTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('av'), 'Optional audio dependencies not installed')
    def test_real_audio_decode_selects_interval_and_flushes_resampler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.wav'
            values = np.concatenate([np.zeros(44100), np.full(44100, 8192)]).astype('<i2')
            with wave.open(str(path), 'wb') as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(44100)
                file.writeframes(values.tobytes())
            audio = read_audio(path, 1, 1)
            self.assertEqual(len(audio), 16000)
            self.assertAlmostEqual(float(audio[8000]), .25, places=3)
            self.assertGreater(float(audio[-1]), .2)
            self.assertIsNone(read_audio(path, 3, 1))
            # A short clip must not be padded to the default 16-second limit.
            full_audio = read_audio(path, 0, 16)
            self.assertEqual(len(full_audio), 32000)
            np.testing.assert_allclose(full_audio[:15000], 0, atol=1e-6)
            self.assertAlmostEqual(float(full_audio[-1]), .25, places=3)

    def test_expression_uses_aspect_corrected_geometry(self):
        points = [NS(x=.5, y=.5) for _ in range(478)]
        points[61] = NS(x=.3, y=.5)
        points[291] = NS(x=.7, y=.5)
        self.assertEqual(lip_expression(points, 640, 480)['label'], 'closed lips')
        points[14] = NS(x=.5, y=.7)
        self.assertEqual(lip_expression(points, 640, 480)['label'], 'open mouth')

    def test_invalid_intervals_never_decode(self):
        for start, duration in [(float('nan'), 3), (0, float('inf')), (-1, 3), (0, 301), (0, 0)]:
            with self.subTest(start=start, duration=duration), patch('audiovisual.read_audio') as decode:
                with self.assertRaises(ValueError):
                    recognize_audiovisual('video.mp4', start, duration)
                decode.assert_not_called()

    def test_no_audio_and_silence_do_not_invent_transcripts(self):
        for audio, status in [(None, 'no_audio'), (np.zeros(16000), 'no_speech')]:
            with patch('audiovisual.read_audio', return_value=audio), patch('audiovisual.expression_timeline', return_value={'samples': []}), patch('audiovisual.speech_model') as model:
                result = recognize_audiovisual('video.mp4')
                self.assertEqual(result['status'], status)
                self.assertIsNone(result['usable_text'])
                model.assert_not_called()

    def test_word_times_include_selected_offset_and_face_failure_preserves_text(self):
        segment = NS(text=' Hello world.', words=[NS(word=' Hello', start=.2, end=.6)])
        model = NS(transcribe=lambda *a, **k: (iter([segment]), NS(language='en')))
        with patch('audiovisual.read_audio', return_value=np.ones(16000)), patch('audiovisual.speech_model', return_value=model), patch('audiovisual.expression_timeline', side_effect=ValueError()):
            result = recognize_audiovisual('video.mp4', 5, 3)
        self.assertEqual(result['usable_text'], 'Hello world.')
        self.assertEqual(result['words'][0]['start'], 5.2)
        self.assertEqual(result['lip_expressions']['samples'], [])

    def test_combined_visual_result_uses_same_interval_and_preserves_words(self):
        result = {'engine': 'original', 'usable_text': 'hello', 'audio_used': False}
        samples = [{'time': 2, 'label': 'smiling lips', 'opening_ratio': .2,
                    'gesture_model_available': True}]
        with patch('audiovisual.expression_timeline', return_value={'samples': samples}) as timeline:
            attach_lip_expressions(result, 'same-clip.mp4', 2, 3)
        timeline.assert_called_once_with('same-clip.mp4', 2, 3)
        self.assertEqual(result['usable_text'], 'hello')
        self.assertFalse(result['audio_used'])
        self.assertEqual(result['dominant_expression'], 'smiling lips')
        self.assertEqual(result['lip_expressions']['summary'], {'smiling lips': 1})

    def test_expression_failure_does_not_discard_visual_words(self):
        result = {'usable_text': 'hello'}
        with patch('audiovisual.expression_timeline', side_effect=ValueError('No visible face')):
            attach_lip_expressions(result, 'same-clip.mp4')
        self.assertEqual(result['usable_text'], 'hello')
        self.assertIsNone(result['dominant_expression'])
        self.assertEqual(result['lip_expressions']['samples'], [])

    def test_manual_recording_keeps_words_and_expressions_after_sixteen_seconds(self):
        segment = NS(text='last words', words=[NS(word='last', start=18., end=19.)])
        model = NS(transcribe=lambda *a, **k: (iter([segment]), NS(language='en')))
        shapes = {'samples': [{'time': 18., 'label': 'smiling lips', 'opening_ratio': .2}]}
        with patch('audiovisual.read_audio', return_value=np.ones(320000)) as audio, patch('audiovisual.speech_model', return_value=model), patch('audiovisual.expression_timeline', return_value=shapes) as expressions:
            result = recognize_audiovisual('camera.webm', 0, 20)
        audio.assert_called_once_with('camera.webm', 0, 20)
        expressions.assert_called_once_with('camera.webm', 0, 20)
        self.assertEqual(result['words'][0]['start'], 18.)
        self.assertEqual(result['lip_expressions']['samples'][0]['time'], 18.)

    def test_audio_route_bypasses_visual_weights_and_cleans_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app({'TESTING': True, 'ENVIRONMENT': 'development', 'API_TOKEN': '', 'UPLOAD_ROOT': directory})
            self.addCleanup(app.extensions['store'].engine.dispose)
            with patch('audiovisual.recognize_audiovisual', return_value={'text': 'hello'}) as recognize:
                response = app.test_client().post('/api/v1/recognize', data={'video': (io.BytesIO(b'test'), 'camera.webm'), 'engine': 'audio-video'})
                self.assertEqual(response.status_code, 200)
                recognize.assert_called_once()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unavailable_model_returns_actionable_error(self):
        app = create_app({'TESTING': True, 'ENVIRONMENT': 'development', 'API_TOKEN': ''})
        self.addCleanup(app.extensions['store'].engine.dispose)
        with patch('audiovisual.recognize_audiovisual', side_effect=RuntimeError('Run setup_audio.py')):
            response = app.test_client().post('/api/v1/recognize', data={'video': (io.BytesIO(b'test'), 'video.mp4'), 'engine': 'audio-video'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('setup_audio.py', response.json['message'])


if __name__ == '__main__':
    unittest.main()
