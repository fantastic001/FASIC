
from ..SpeakerFeatureExtractor import * 

import scipy.fftpack 
import numpy as np 

class FFTSpeakerFeatureExtractor(SpeakerFeatureExtractor):
    
    def generateFeatures(self, data):
        res = scipy.fftpack.fft(data, n=256)
        return np.abs(res)
