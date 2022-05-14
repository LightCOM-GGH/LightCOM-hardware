import glob
import time

import inference
import api
import utils


def main():
    # model = inference.TFLite(416, 416)
    model = inference.MobileNetV2(416, 416)
    stats = utils.Stats()
    cars_stats = utils.Stats()
    pedestrian_stats = utils.Stats()

    for impath in glob.glob("../data/*.jpg"):
        img = utils.load_image(impath)
        timestamp = time.time()

        cars_count, pedestrian_count, vis_image = model.predict(img)

        cars_mean = cars_stats.add(timestamp, cars_count)
        pedestrian_mean = pedestrian_stats.add(timestamp, pedestrian_count)
        
        api.send_info(cars_count, pedestrian_count, cars_mean, pedestrian_mean, "red")

        print(cars_count, pedestrian_count)
        # print(len(bbox), mean)
        utils.show("image", img, 0)


if __name__ == '__main__':
    main()
