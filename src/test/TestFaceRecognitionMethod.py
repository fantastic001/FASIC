
from FASIC import FaceRecognitionMethod

import unittest

class DummyFaceRecognitionMethod(FaceRecognitionMethod):
    def train(self):
        pass

    def classify(self, data):
        return self.getTraining()[0][0]

class TestFaceRecognitionMethod(unittest.TestCase):

    def setUp(self):
        self.recognizer = DummyFaceRecognitionMethod(None, [("a", [1,2,3])])

    def test_classify(self):
        self.assertEqual(self.recognizer.classify([1,2,3]), "a")
