import camera
import utils
import inference


def main():
    cam = camera.Camera(416, 416)
    
    while True:
        img = cam.read()
        utils.show("image", img)

    return


if __name__ == '__main__':
    main()
