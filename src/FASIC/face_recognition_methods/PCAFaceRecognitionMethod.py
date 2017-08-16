
from ..FaceRecognitionMethod import * 

from sklearn.decomposition import PCA

import numpy as np 

class PCAFaceRecognitionMethod(FaceRecognitionMethod):

    def train(self):
        T = self.getTraining()
        X = [] 
        for label, x in T:
            X.append(x.reshape(x.shape[0] * x.shape[1]))
        self.model = PCA()
        self.model.fit(X)


    def classify(self, data):
        data = data.reshape(data.shape[0] * data.shape[1])
        res = self.model.transform([data])[0]
        T = self.getTraining()
        labels = [] 
        X = [] 
        for label, x in T:
            X.append(x.reshape(x.shape[0] * x.shape[1]))
            labels.append(label)
        # now project to eigenspace
        Y = [] 
        for x in X:
            Y.append(self.model.transform([x])[0])
        dmin = 10**9
        labelmin = labels[0]
        for i in range(len(Y)):
            x = Y[i]
            d = np.sqrt((x - res).dot(x-res))
            if d < dmin:
                dmin = d
                labelmin = labels[i]
        return labelmin
