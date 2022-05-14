import time

import camera
import inference
import utils
import api


def main():
    # model = inference.TFLite(416, 416)
    model = inference.MobileNetV2(640, 480)
    cam = camera.Camera(640, 480)
    cars_stats = utils.Stats()
    pedestrian_stats = utils.Stats()

    while True:
        img = cam.read()
        timestamp = time.time()

        cars_count, pedestrian_count, vis_image = model.predict(img)

        cars_mean = cars_stats.add(timestamp, cars_count)
        pedestrian_mean = pedestrian_stats.add(timestamp, pedestrian_count)

        api.send_info(cars_count, pedestrian_count,
                      cars_mean, pedestrian_mean, "red")

        print(cars_count, pedestrian_count)
        # print(len(bbox), mean)
        utils.show("image", vis_image)


if __name__ == '__main__':
    main()
