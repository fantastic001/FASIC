
import unittest 

from FASIC.face_recognition_methods import PCAFaceRecognitionMethod

import numpy as np 

class TestPCAFaceRecognitionMethod(unittest.TestCase):
    
    def setUp(self):
        self.training = [
            ("a", np.array([[0,0], [0, 0]])),
            ("b", np.array([[1,1], [1, 1]]))
        ]

    def test_recognition_from_training(self):
        method = PCAFaceRecognitionMethod(None, self.training)
        method.train()
        res = method.classify(self.training[0][1])
        self.assertEqual(res, self.training[0][0])
