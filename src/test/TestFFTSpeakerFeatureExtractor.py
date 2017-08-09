
import unittest 

from FASIC.speaker_feature_extractors import FFTSpeakerFeatureExtractor 

import numpy as np 
import numpy.testing as nptest

class TestFFTSpeakerFeatureExtractor(unittest.TestCase):
    
    def test_zeros(self):
        data = np.zeros(200)
        extractor = FFTSpeakerFeatureExtractor()
        f = extractor.generateFeatures(data)
        nptest.assert_allclose(f, np.zeros(256))
