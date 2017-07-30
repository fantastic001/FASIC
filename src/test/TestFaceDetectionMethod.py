
import unittest

from FASIC import FaceDetectionMethod

class DummyFaceDetectionMethod(FaceDetectionMethod):
    def detect(self):
        return []

class TestFaceDetectionMethod(unittest.TestCase):

    def test_detect(self):
        detector = DummyFaceDetectionMethod()
        self.assertEqual(detector.detect(), [])
