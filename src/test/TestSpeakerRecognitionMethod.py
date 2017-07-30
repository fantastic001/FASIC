
from FASIC import SpeakerRecognitionMethod, SpeakerFeatureExtractor

import unittest

class DummySpeakerRecognitionMethod(SpeakerRecognitionMethod):
    
    def onTrain(self, data):
        self.label = data[0][0]

    def onClassify(self, features):
        return self.label 

class DummyExtractor(SpeakerFeatureExtractor):
    def generateFeatures(self, data):
        return [1,2,3]

class TestSpeakerRecognitionMethod(unittest.TestCase):
    
    def test_dummy(self):   
        dummy = DummySpeakerRecognitionMethod()
        dummy.train([("a", [1,1,1])], [DummyExtractor()])
        self.assertEqual(dummy.classify([1,2,3]), "a")
