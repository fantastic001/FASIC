
import unittest 

from FASIC.speaker_recognition_methods import NeuralNetworkSpeakerRecognitionMethod
from FASIC import SpeakerFeatureExtractor

class DummyExtractor(SpeakerFeatureExtractor):
    
    def generateFeatures(self, data):
        if data[0] == 0:
            return [0, 0]
        else:
            return [1, 1]

class TestNeuralNetworkSpeakerRecognitionMethod(unittest.TestCase):
    def setUp(self):
        self.extractor = DummyExtractor()
        self.method = NeuralNetworkSpeakerRecognitionMethod()

    def test_basic(self):
        data = [
            (0, [0, 0, 0]),
            (1, [1, 5, 5]),
            (0, [0, 1, 2])
        ]
        self.method.train(data[:2], [self.extractor])
        self.assertEqual(self.method.classify(data[2][1]), data[2][0])

    def test_with_string_labels(self):
        data = [
            ("a", [0, 0, 0]),
            ("b", [1, 5, 5]),
            ("a", [0, 1, 2])
        ]
        self.method.train(data[:2], [self.extractor])
        self.assertEqual(self.method.classify(data[2][1]), data[2][0])
