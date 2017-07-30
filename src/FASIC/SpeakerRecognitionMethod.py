
class SpeakerRecognitionMethod(object):
    
    def getExtractors(self):
        """
        Returns list of extractors obtained during training 
        """
        return self.extractors

    def extractFeatures(self, data):
        """
        Extracts features from extractors provided to train
        """
        res = []
        for extractor in self.extractors:
            ff = extractor.generateFeatures(data)
            for f in ff:
                res.append(f)
        return res
    
    def train(self, samples, extractors):
        """
        samples: list of tuples (label, data)
        extractors: list of SpeakerFeatureExtractor objects which will serve to extract features from data 
        """
        self.extractors = extractors
        data = [] 
        for sample in samples:
            l,d = sample
            data.append((l, self.extractFeatures(d)))
        self.onTrain(data)

    def classify(self, data):
        return self.onClassify(self.extractFeatures(data))

    def onTrain(self, data):
        """
        data: list of tuples (label, features)
        """
        pass
    
    def onClassify(self, features):
        """
        Returns label 
        """
        pass
