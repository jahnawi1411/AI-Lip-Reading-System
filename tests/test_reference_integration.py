"""Local-asset integration checks; no network access, training or saved accounts."""
import base64
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'models/lipnet-reference/unseen-weights178.h5'
VIDEO = ROOT / 'data/public_regression/sbia1a.mpg'


@unittest.skipUnless(WEIGHTS.exists() and VIDEO.exists(), 'Install public reference model and samples to run integration checks')
class PublicPipelineTests(unittest.TestCase):
    def test_upload_and_camera_frames_keep_history_and_model_provenance(self):
        import cv2
        from service import create_app
        from recognition import weight_fingerprint
        before = weight_fingerprint(WEIGHTS)
        with tempfile.TemporaryDirectory() as directory:
            app = create_app({'TESTING': True, 'ENVIRONMENT': 'development', 'API_TOKEN': '',
                              'UPLOAD_ROOT': directory, 'REFERENCE_WEIGHTS_PATH': str(WEIGHTS)})
            self.addCleanup(app.extensions['store'].engine.dispose)
            client = app.test_client()
            csrf = client.get('/api/auth/session').json['csrf']
            login = client.post('/api/auth/register', json={'username':'reference_test','password':'test-password-123'},
                                headers={'X-CSRF-Token':csrf})
            headers={'X-CSRF-Token':login.json['csrf']}
            upload=client.post('/api/v1/recognize',headers=headers,data={
                'video':(io.BytesIO(VIDEO.read_bytes()),'clip.mpg'),'engine':'lipnet-reference','mode':'mediapipe'})
            self.assertEqual(upload.status_code,200,upload.json)
            self.assertEqual(upload.json['engine'],'lipnet-reference')
            self.assertEqual(upload.json['domain'],'GRID six-word commands')
            self.assertTrue(upload.json['review_required'])
            self.assertNotIn('weights',upload.json)
            self.assertNotIn('source',upload.json)
            self.assertIn('input_quality',upload.json)
            self.assertEqual(list(Path(directory).iterdir()),[])
            capture=cv2.VideoCapture(str(VIDEO)); frames=[]
            try:
                while True:
                    ok,frame=capture.read()
                    if not ok: break
                    frames.append(base64.b64encode(cv2.imencode('.jpg',frame)[1]).decode())
            finally:
                capture.release()
            response=client.post('/api/v1/recognize-frames',headers=headers,json={
                'engine':'lipnet-reference','frames':frames,'timestamps_ms':list(range(0,3000,40))})
            self.assertEqual(response.status_code,200,response.json)
            self.assertEqual(response.json['engine'],'lipnet-reference')
            self.assertIn('capture_timing',response.json)
            self.assertNotIn('weights',response.json)
            self.assertEqual(len(client.get('/api/v1/history').json['items']),2)
            self.assertEqual(weight_fingerprint(WEIGHTS),before)
            invalid=client.post('/api/v1/recognize-frames',headers=headers,json={'engine':[]})
            self.assertEqual(invalid.status_code,400)


if __name__ == '__main__':
    unittest.main()
