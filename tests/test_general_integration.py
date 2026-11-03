"""Real local sentence model tests; skipped when optional assets are absent."""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from general_bridge import available, _stop_worker

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT/'data/public_regression/sbia1a.mpg'


@unittest.skipUnless(available() and VIDEO.exists(), 'Install optional sentence model and public samples')
class GeneralIntegrationTests(unittest.TestCase):
    def test_real_upload_camera_and_history(self):
        self.addCleanup(_stop_worker, True)
        import cv2
        from service import create_app
        with tempfile.TemporaryDirectory() as directory:
            app = create_app({'TESTING':True,'ENVIRONMENT':'development','API_TOKEN':'','UPLOAD_ROOT':directory})
            self.addCleanup(app.extensions['store'].engine.dispose)
            client = app.test_client()
            csrf = client.get('/api/auth/session').json['csrf']
            login = client.post('/api/auth/register',json={'username':'sentence_test','password':'test-password-123'},headers={'X-CSRF-Token':csrf})
            headers = {'X-CSRF-Token':login.json['csrf']}
            self.assertIn('English sentences (experimental)',client.get('/').text)
            upload = client.post('/api/v1/recognize',headers=headers,data={'video':(io.BytesIO(VIDEO.read_bytes()),'clip.mpg'),'engine':'auto-avsr'})
            self.assertEqual(upload.status_code,200,upload.json)
            capture = cv2.VideoCapture(str(VIDEO)); frames = []
            try:
                while True:
                    ok,frame = capture.read()
                    if not ok: break
                    frames.append(base64.b64encode(cv2.imencode('.jpg',frame)[1]).decode())
            finally:
                capture.release()
            camera = client.post('/api/v1/recognize-frames',headers=headers,json={'engine':'auto-avsr','frames':frames,'timestamps_ms':list(range(0,3000,40))})
            self.assertEqual(camera.status_code,200,camera.json)
            for result in (upload.json,camera.json):
                self.assertEqual(result['engine'],'auto-avsr')
                self.assertFalse(result['audio_used'])
                self.assertTrue(result['review_required'])
                self.assertIsNone(result['model_score'])
                self.assertEqual(len(result['weights_sha256']),64)
                self.assertNotIn('input_report',result)
                self.assertTrue(result['text'])
            self.assertEqual(len(client.get('/api/v1/history').json['items']),2)
            self.assertEqual(list(Path(directory).iterdir()),[])
            # Record raw outputs; reference is only attached after inference.
            report = {'reference':'set blue in a one again','upload':upload.json,'camera_frames':camera.json,
                      'scope':'Single public GRID smoke check, not a general-speech benchmark or physical camera test.'}
            (ROOT/'reports/general-integration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
