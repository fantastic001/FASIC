
import unittest 

from FASIC.face_detection import HAARFaceDetectionMethod

import numpy as np 

class TestHAARFaceDetectionMethod(unittest.TestCase):
    
    def test_on_black_image(self):
        img = np.zeros([100, 100, 3])
        method = HAARFaceDetectionMethod()
        res = method.detect(img)
        self.assertEqual(res, [])
