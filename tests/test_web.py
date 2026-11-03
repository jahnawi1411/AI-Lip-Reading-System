import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from service import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.app = create_app({"TESTING": True, "ENVIRONMENT": "development", "API_TOKEN": "", "UPLOAD_ROOT": self.directory.name})
        self.addCleanup(self.app.extensions["store"].engine.dispose)
        self.client = self.app.test_client()

    def register(self, client, username):
        csrf = client.get('/api/auth/session').json['csrf']
        response = client.post('/api/auth/register', json={'username': username, 'password': 'test-password-123'}, headers={'X-CSRF-Token': csrf})
        self.assertEqual(response.status_code, 200)
        return {'X-CSRF-Token': response.json['csrf']}

    def test_home_and_security_headers(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'VisionSpeak', response.data)
        self.assertIn("script-src 'self'", response.headers['Content-Security-Policy'])
        self.assertIn('microphone=(self)', response.headers['Permissions-Policy'])
        self.assertIn(b'value="audio-video"', response.data)
        with self.client.get('/static/app.js') as asset:
            self.assertEqual(asset.status_code, 200)

    def test_csrf_login_and_logout(self):
        self.assertEqual(self.client.post('/api/auth/register', json={}).status_code, 403)
        headers = self.register(self.client, 'alice')
        self.assertEqual(self.client.get('/api/auth/session').json['user']['username'], 'alice')
        self.assertEqual(self.client.delete('/api/v1/history').status_code, 403)
        self.assertEqual(self.client.post('/api/auth/logout', headers=headers).status_code, 200)
        self.assertEqual(self.client.get('/api/v1/history').status_code, 401)
        csrf = self.client.get('/api/auth/session').json['csrf']
        response = self.client.post('/api/auth/login', headers={'X-CSRF-Token': csrf}, json={'username':'alice','password':'test-password-123'})
        self.assertEqual(response.status_code, 200)

    def test_history_is_private_and_persistent(self):
        headers = self.register(self.client, 'alice')
        result = {'text': 'hi', 'usable_text': 'hi', 'confidence': None, 'review_required': True}
        with patch('lip_reader.get_model'), patch('preprocessing.LipDetector'), patch('recognition.recognize_file', return_value=result):
            response = self.client.post('/api/v1/recognize', headers=headers,
                data={'video': (io.BytesIO(b'test'), 'sample.mp4')})
        self.assertEqual(response.status_code, 200)
        identifier = response.json['history_id']
        self.assertEqual(len(self.client.get('/api/v1/history').json['items']), 1)
        bob = self.app.test_client(); bob_headers = self.register(bob, 'bob')
        self.assertEqual(bob.get('/api/v1/history').json['items'], [])
        self.assertEqual(bob.delete('/api/v1/history/'+identifier, headers=bob_headers).status_code, 404)
        self.assertEqual(self.client.delete('/api/v1/history/'+identifier, headers=headers).status_code, 200)
        self.assertEqual(self.client.get('/api/v1/history').json['items'], [])

    def test_words_and_expressions_saved_together_from_one_upload(self):
        headers = self.register(self.client, 'combined')
        result = {'text': 'hello', 'usable_text': 'hello', 'engine': 'original', 'audio_used': False}
        shapes = {'samples': [{'time': 1, 'label': 'smiling lips', 'opening_ratio': .2}]}
        with patch('lip_reader.get_model'), patch('preprocessing.LipDetector'), patch('recognition.recognize_file', return_value=result) as words, patch('audiovisual.expression_timeline', return_value=shapes) as expressions:
            response = self.client.post('/api/v1/recognize', headers=headers,
                data={'video': (io.BytesIO(b'test'), 'clip.mp4'), 'engine': 'original', 'start': '1', 'duration': '3'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(words.call_args.args[0], expressions.call_args.args[0])
        self.assertEqual(expressions.call_args.args[1:], (1., 3.))
        self.assertEqual(response.json['usable_text'], 'hello')
        self.assertEqual(response.json['dominant_expression'], 'smiling lips')
        saved = self.client.get('/api/v1/history').json['items'][0]['result']
        self.assertEqual(saved['lip_expressions']['summary'], {'smiling lips': 1})
        self.assertEqual(saved['usable_text'], 'hello')

    def test_webcam_validation_and_saved_result(self):
        headers = self.register(self.client, 'alice')
        self.assertEqual(self.client.post('/api/v1/recognize-frames', json={'frames':[]}, headers=headers).status_code, 400)
        import cv2, base64
        image = np.random.default_rng(2).integers(0, 255, (120,160,3), dtype=np.uint8)
        encoded = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
        data = {'frames':[encoded]*75, 'timestamps_ms':list(range(0,3000,40))}
        result = {'text': 'hi', 'usable_text': 'hi', 'confidence': None, 'review_required': True}
        with patch('lip_reader.get_model'), patch('preprocessing.LipDetector') as detector, patch('recognition.recognize_frames', return_value=result):
            detector.return_value.__enter__.return_value.detect.return_value = (np.ones((46,140),np.uint8),(1,1,2,2))
            response = self.client.post('/api/v1/recognize-frames', json=data, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn('history_id', response.json)
        self.assertEqual(self.client.get('/api/v1/history').json['items'][0]['result']['source_label'], 'Webcam capture')
        data['timestamps_ms'] = [0]*75
        self.assertEqual(self.client.post('/api/v1/recognize-frames', json=data, headers=headers).status_code, 422)

    def test_database_reopens_and_limits_requests(self):
        from history_store import Store
        url = 'sqlite:///' + (Path(self.directory.name)/'persist.db').as_posix()
        first = Store(url); user=first.register('persist', 'hash'); first.save(user['id'], {'text':'hello'})
        first.engine.dispose()
        second=Store(url)
        try:
            self.assertEqual(second.list(user['id'])[0]['result']['text'], 'hello')
            self.assertTrue(second.allow('test', maximum=1))
            self.assertFalse(second.allow('test', maximum=1))
        finally:
            second.engine.dispose()


if __name__ == '__main__':
    unittest.main()
