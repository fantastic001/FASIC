
from ArgumentStack import * 
import sys 
import scipy.io.wavfile 

import matplotlib.pyplot as plt 

from FASIC.speaker_feature_extractors import * 
from FASIC.speaker_recognition_methods import * 

import os

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


def sr_train_nn(path, sample_path, **kw):
    model = NeuralNetworkSpeakerRecognitionMethod()
    extractor = FFTSpeakerFeatureExtractor()
    samples = []
    for label in os.listdir(path):
        for filename in os.listdir("%s/%s" % (path, label)):
            fpath = "%s/%s/%s" % (path, label, filename)
            rate, data = scipy.io.wavfile.read(fpath)
            samples.append((label, data))
    model.train(samples, [extractor])
    rate, data = scipy.io.wavfile.read(sample_path)
    res = model.classify(data)
    print("Result: %s" % res)

stack.pushCommand("sr")
stack.pushCommand("check")
stack.pushVariable("path")
stack.pushVariable("sample_path")
stack.assignAction(sr_train_nn, "Train and classify using FFT and NN. PATH is path to training dir and sample_path is uknown.")


stack.popAll()

stack.pushCommand("help")
stack.assignAction(lambda: print(stack.getHelp()), "Get help")

stack.execute(sys.argv)
