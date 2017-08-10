
from ..SpeakerRecognitionMethod import * 

from sklearn.neural_network import MLPClassifier

class NeuralNetworkSpeakerRecognitionMethod(SpeakerRecognitionMethod):
    def onTrain(self, data):
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10), max_iter=1000)
        x = [] 
        y = [] 
        self.labels = []
        for d in data:
            label, sample = d
            if label in self.labels:
                label = self.labels.index(label)
            else:
                self.labels.append(label)
                label = self.labels.index(label)
            y.append(label)
            x.append(sample)
        self.clf.fit(x,y)
    
    def onClassify(self, features):
        res = self.clf.predict([features])[0]
        return self.labels[res]
