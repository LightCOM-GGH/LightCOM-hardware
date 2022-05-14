import sys

import camera
import inference
import utils
import glob


def main(demo=False):
    model = inference.TFLite(416, 416)
    if demo:
        for impath in glob.glob("../data/*.jpg"):
            img = utils.load_image(impath)
            boxes, scores, classes, detections, vis_image = model.predict(
                img, vis=True)
            utils.show("vis", vis_image, 0)

    else:
        cam = camera.Camera(224, 224)
        while True:
            img = cam.read()
            boxes, scores, classes, detections, vis_image = model.predict(img)
            utils.show("image", img)


if __name__ == '__main__':
    main(len(sys.argv) == 2 and sys.argv[1] == "demo")
