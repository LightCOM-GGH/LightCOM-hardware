import cv2


def show(windowname, img, waitKey=1):
    cv2.imshow(windowname, img)
    cv2.waitKey(waitKey)
