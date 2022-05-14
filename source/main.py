import time

import camera
import inference
import utils
import api
import lighthandler


def main():
    # model = inference.TFLite(416, 416)
    model = inference.MobileNetV2(640, 480)
    cam = camera.Camera(640, 480)
    cars_stats = utils.Stats()
    pedestrian_stats = utils.Stats()
    trafficLight = lighthandler.TrafficLight(30)

    while True:
        img = cam.read()
        timestamp = time.time()

        cars_count, pedestrian_count, vis_image = model.predict(img)

        cars_mean = cars_stats.add(timestamp, cars_count)
        pedestrian_mean = pedestrian_stats.add(timestamp, pedestrian_count)

        # update traffic light data
        trafficLight()

        api.send_info(cars_count, pedestrian_count,
                      cars_mean, pedestrian_mean, trafficLight.state)

        print(cars_count, pedestrian_count, trafficLight.state)
        # print(len(bbox), mean)
        utils.show("image", vis_image)


if __name__ == '__main__':
    main()
