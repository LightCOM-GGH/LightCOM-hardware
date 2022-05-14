import cv2


def show(windowname, img, waitKey=1):
    cv2.imshow(windowname, img)
    cv2.waitKey(waitKey)


def load_image(path):
    return cv2.imread(path)


class Stats():
    def __init__(self, time_window=120):
        """Init of the stat class.

        Args:
            time_window (int, optional): how big the sliding window should be big. Defaults to 120.
        """
        self.time_window = time_window

        self.data = []
        self.mean = 0

    def add(self, timestamp, count):
        """Add a point to the stat and process the mean.

        Args:
            timestamp (_type_): _description_
            count (_type_): _description_
        Returns:
            float: mean number of cars during a given period of time.
        """
        self.data.append((timestamp, count))

        sum_count = 0
        to_remove = 0
        points = 0
        for (tmp, c) in self.data:
            if tmp + self.time_window >= timestamp:
                sum_count += c
                points += 1
            else:
                to_remove += 1

        for _ in range(to_remove):
            self.data.pop(0)

        self.mean = sum_count / points
        return self.mean
