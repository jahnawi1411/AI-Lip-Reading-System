import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS

from audiovisual import lip_expression, recognize_expressions
from service import create_app


class ExpressionTests(unittest.TestCase):
    def shape(self, opening=0, **scores):
        points = [NS(x=.5, y=.5) for _ in range(478)]
        points[61] = NS(x=.3, y=.5)
        points[291] = NS(x=.7, y=.5)
        points[14] = NS(x=.5, y=.5 + opening * .4)
        coefficients = dict(mouthSmileLeft=0., mouthSmileRight=0., mouthPucker=0., mouthFunnel=0.)
        coefficients.update(scores)
        return lip_expression(points, 640, 640, [NS(category_name=k, score=v) for k,v in coefficients.items()])

    def test_smile_pout_purse_and_opening_are_distinct(self):
        self.assertEqual(self.shape(mouthSmileLeft=.8, mouthSmileRight=.7)['label'], 'smiling lips')
        smile = self.shape(.4, mouthSmileLeft=.8, mouthSmileRight=.7)
        self.assertEqual(smile['label'], 'smiling lips')
        self.assertEqual(smile['opening_label'], 'open mouth')
        self.assertEqual(self.shape(mouthPucker=.8)['label'], 'pursed lips')
        self.assertEqual(self.shape(.2, mouthPucker=.8)['label'], 'pouting lips')
        self.assertEqual(self.shape(.2, mouthFunnel=.8)['label'], 'pouting lips')
        for opening, label in [(0., 'closed lips'), (.15, 'slightly open lips'), (.4, 'open mouth')]:
            self.assertEqual(self.shape(opening)['label'], label)

    def test_ambiguous_and_one_sided_signals_do_not_force_a_gesture(self):
        self.assertEqual(self.shape(mouthSmileLeft=.9, mouthSmileRight=.1)['label'], 'closed lips')
        self.assertEqual(self.shape(mouthSmileLeft=.8, mouthSmileRight=.8, mouthPucker=.75)['label'], 'closed lips')
        self.assertEqual(self.shape(mouthPucker=.3)['label'], 'closed lips')

    def test_invalid_scores_fall_back_without_nonfinite_json(self):
        import json
        for value in [float('nan'), float('inf'), -1, 2]:
            result = self.shape(mouthPucker=value)
            self.assertFalse(result['gesture_model_available'])
            self.assertEqual(result['label'], 'closed lips')
            json.dumps(result, allow_nan=False)

    def test_no_blendshapes_keeps_geometry_and_reports_fallback(self):
        points = [NS(x=.5, y=.5) for _ in range(478)]
        points[61].x, points[291].x = .3, .7
        result = lip_expression(points, 640, 480)
        self.assertFalse(result['gesture_model_available'])
        self.assertEqual(result['label'], 'closed lips')
        points[61].x = float('nan')
        with self.assertRaises(ValueError):
            lip_expression(points, 640, 480)

    @unittest.skipUnless(Path('models/face_landmarker.task').exists() and Path('data/grid_check/id2_vcd_swwp2s.mpg').exists(), 'Local face model and sample required')
    def test_real_video_produces_mouth_gesture_coefficients(self):
        result = recognize_expressions('data/grid_check/id2_vcd_swwp2s.mpg', 0, 1)
        expressions = result['lip_expressions']
        self.assertEqual(result['status'], 'measured')
        self.assertGreater(expressions['gesture_samples'], 0)
        self.assertFalse(result['audio_used'])
        self.assertEqual(sum(expressions['summary'].values()), expressions['detected_samples'])
        valid = {'smiling lips', 'pouting lips', 'pursed lips', 'closed lips', 'slightly open lips', 'open mouth', 'face unavailable'}
        for sample in expressions['samples']:
            self.assertIn(sample['label'], valid)
            if sample.get('gesture_model_available'):
                self.assertIn('mouthPucker', sample['mouth_scores'])
                self.assertIn('opening_label', sample)

    def test_summary_excludes_missing_faces_and_does_not_use_audio(self):
        samples = [
            {'time': 2, 'label': 'closed lips', 'opening_ratio': 0.02},
            {'time': 2.2, 'label': 'face unavailable', 'opening_ratio': None},
            {'time': 2.4, 'label': 'open mouth', 'opening_ratio': 0.4},
            {'time': 2.6, 'label': 'closed lips', 'opening_ratio': 0.01},
        ]
        with patch('audiovisual.expression_timeline', return_value={'samples': samples}) as timeline, patch('audiovisual.read_audio') as audio, patch('audiovisual.speech_model') as model:
            result = recognize_expressions('silent.mp4', 2, 3)
        timeline.assert_called_once_with('silent.mp4', 2, 3)
        audio.assert_not_called()
        model.assert_not_called()
        self.assertFalse(result['audio_used'])
        self.assertIsNone(result['usable_text'])
        self.assertEqual(result['dominant_expression'], 'closed lips')
        self.assertEqual(result['lip_expressions']['detected_samples'], 3)
        self.assertEqual(result['lip_expressions']['total_samples'], 4)

    def test_missing_face_is_not_a_recognized_expression(self):
        with patch('audiovisual.expression_timeline', return_value={'samples': [{'label': 'face unavailable', 'opening_ratio': None}]}):
            result = recognize_expressions('video.mp4')
        self.assertEqual(result['status'], 'face_unavailable')
        self.assertIsNone(result['dominant_expression'])

    def test_empty_interval_and_invalid_times(self):
        with patch('audiovisual.expression_timeline', return_value={'samples': []}):
            with self.assertRaises(ValueError):
                recognize_expressions('video.mp4')
        for start, duration in [(-1, 3), (61, 3), (float('nan'), 3), (0, 0), (0, 301), (0, float('inf'))]:
            with self.subTest(start=start, duration=duration), patch('audiovisual.expression_timeline') as timeline:
                with self.assertRaises(ValueError):
                    recognize_expressions('video.mp4', start, duration)
                timeline.assert_not_called()

    def test_route_history_and_cleanup_without_speech_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app({'TESTING': True, 'ENVIRONMENT': 'development', 'API_TOKEN': '', 'UPLOAD_ROOT': directory})
            self.addCleanup(app.extensions['store'].engine.dispose)
            client = app.test_client()
            csrf = client.get('/api/auth/session').json['csrf']
            auth = client.post('/api/auth/register', json={'username': 'expressions', 'password': 'test-password-123'}, headers={'X-CSRF-Token': csrf})
            headers = {'X-CSRF-Token': auth.json['csrf']}
            samples = [{'time': 0, 'label': 'closed lips', 'opening_ratio': 0.01}]
            with patch('audiovisual.expression_timeline', return_value={'samples': samples}), patch('lip_reader.get_model') as model:
                response = client.post('/api/v1/recognize', headers=headers, data={'video': (io.BytesIO(b'test'), 'silent.webm'), 'engine': 'lip-expression'})
            self.assertEqual(response.status_code, 200)
            model.assert_not_called()
            self.assertEqual(response.json['dominant_expression'], 'closed lips')
            self.assertEqual(client.get('/api/v1/history').json['items'][0]['result']['engine'], 'lip-expression')
            self.assertEqual(list(Path(directory).iterdir()), [])
            for error, status in [(ValueError('Invalid video'), 422), (RuntimeError('Face model missing'), 503)]:
                with patch('audiovisual.recognize_expressions', side_effect=error):
                    response = client.post('/api/v1/recognize', headers=headers, data={'video': (io.BytesIO(b'test'), 'silent.mp4'), 'engine': 'lip-expression'})
                self.assertEqual(response.status_code, status)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
