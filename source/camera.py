import cv2


class Camera():
    def __init__(self, width, height, index=0):
        """Init of the camera

        Args:
            width (int): width of the image to be captured
            height (_type_): height of the image to be captured
            index (int, optional): index of the camera to open. Defaults to 0.
        """
        self.cap = cv2.VideoCapture(0)
        self.width = width
        self.height = height

    def read(self):
        ret, img = self.cap.read()

        if ret:
            img = cv2.resize(img, (self.width, self.height))
            return img
        else:
            raise ValueError("Could not grab image")
