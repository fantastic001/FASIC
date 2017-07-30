
class FaceRecognitionMethod:

    def __init__(self, detector, training):
        """
        detector: FaceDetectionMethod object - used to detect faces in training images
        training: list of tuples (label, image) where image is of shape [rows, columns, 3] 
        """
        self.detector = detector
        self.training = training

    def getTraining(self):
        return self.training

    def train(self):
        pass

    def classify(self):
        pass
