
from ArgumentStack import * 
import sys 
import scipy.io.wavfile 

import matplotlib.pyplot as plt 

from FASIC.speaker_feature_extractors import * 

stack = ArgumentStack()

def show_speaker_features(fun):
    def inner(path, **kw):
        rate, data = scipy.io.wavfile.read(path)
        extractor = fun()
        res = extractor.generateFeatures(data)
        plt.plot(range(len(res)), res)
        plt.show()
    stack.pushCommand(fun.__name__.lower())
    stack.pushVariable("path")
    stack.assignAction(inner, "Show speaker features using %s" % fun.__name__.lower())
    stack.pop()
    stack.pop()
    return inner

stack.pushCommand("show")
stack.pushCommand("speaker")
stack.pushCommand("features")

@show_speaker_features
def fft():
    return FFTSpeakerFeatureExtractor()

stack.popAll()
stack.pushCommand("help")
stack.assignAction(lambda: print(stack.getHelp()), "Get help")

stack.execute(sys.argv)
