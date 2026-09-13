import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from reference_lipnet import normalize_reference, predict_reference, legacy_hard_sigmoid


class ReferenceTests(unittest.TestCase):
    def test_rgb_axes_and_scale(self):
        frames = np.zeros((75, 50, 100, 3), np.uint8)
        frames[:, 10, 20, 0] = 255
        normalized = normalize_reference(frames)
        self.assertEqual(normalized.shape, (75, 100, 50, 3))
        self.assertEqual(normalized[0, 20, 10, 0], 1)
        self.assertEqual(normalized[0, 20, 10, 2], 0)
        with self.assertRaises(ValueError):
            normalize_reference(np.zeros((75,46,140),np.uint8))

    def test_original_recurrent_activation(self):
        import tensorflow as tf
        np.testing.assert_allclose(legacy_hard_sigmoid(tf.constant([-3.,-2.,0.,2.,3.])).numpy(),
                                   [0,.1,.5,.9,1], atol=1e-7)

    def test_reference_alphabet_mapping(self):
        import tensorflow as tf
        probs = np.zeros((1,75,28),np.float32)
        probs[:,:,27] = 1
        for index, label in [(0,0),(2,26),(4,25)]:
            probs[0,index,:] = 0
            probs[0,index,label] = 1
        with patch('reference_lipnet.get_reference_model') as get:
            get.return_value.return_value = tf.constant(probs)
            result = predict_reference(np.zeros((75,100,50,3),np.float32))
        self.assertEqual(result['text'],'a z')
        self.assertEqual(result['status'],'decoded')

    def test_engine_rejects_incompatible_prepared_input(self):
        from recognition import recognize_file
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sequence.npz'; path.touch()
            with self.assertRaisesRegex(ValueError,'video'):
                recognize_file(path,engine='lipnet-reference')


if __name__ == '__main__':
    unittest.main()
