
class FaceDetectionMethod:
    """
    This class is aimed to be subclassed and to implement individual face detection methods. 
    """

    def detect(self, image):
        """
        image: array of shape [height, width, 

        Returns: list of tuples where each tuple is (i, j, h, w) where i is row, j is column of starting point and h,w are height
        and width of rectangle respective;y. 
        """
        pass
