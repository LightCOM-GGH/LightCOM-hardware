import glob
import time

import inference
import api
import utils
import lighthandler
import cv2


def main():
    # model = inference.TFLite(416, 416)
    model = inference.MobileNetV2(416, 416)
    cars_stats = utils.Stats()
    pedestrian_stats = utils.Stats()
    trafficLight = lighthandler.TrafficLight(10)

    imgpaths = glob.glob("../data/*.jpg")
    for impath in imgpaths:
        img = utils.load_image(impath)
        h, w, _ = img.shape
        # resize with respect to aspect ratio
        img = cv2.resize(img, (832, int(832 * h / w)))
        timestamp = time.time()

        cars_count, pedestrian_count, vis_image = model.predict(img)

        cars_mean = cars_stats.add(timestamp, cars_count)
        pedestrian_mean = pedestrian_stats.add(timestamp, pedestrian_count)

        # update traffic light data
        trafficLight()

        # api.send_info(cars_count, pedestrian_count,
        #               cars_mean, pedestrian_mean, trafficLight.state)

        print(cars_count, pedestrian_count, trafficLight.state)
        # print(len(bbox), mean)
        utils.show("image", img, 0)


if __name__ == '__main__':
    main()
