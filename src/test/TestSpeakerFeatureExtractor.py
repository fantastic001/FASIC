
import unittest 

from FASIC import SpeakerFeatureExtractor 

class DummySpeakerFeatureExtractor(SpeakerFeatureExtractor):
    
    def generateFeatures(self, data):
        return [1,2,3,]

class TestSpeakerFeatureExtractor(unittest.TestCase):
    
    def test_dummy(self):
        dummy = DummySpeakerFeatureExtractor()
        self.assertEqual(dummy.generateFeatures([1,1,1]), [1,2,3])
