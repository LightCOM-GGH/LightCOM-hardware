import requests
import uuid
import time


URL = "http://192.168.117.218:3000"
UUID = "2f00494c0483cfd78b4c3c0c7a086f048eb8c918"  # uuid.uuid4()


def send_info(cars_count, pedestrian_count, cars_mean, pedestrian_mean, status):
    x = requests.post(
        f"{URL}/update",
        data={
            "uuid": UUID,
            "time": time.time(),
            "stats": {
                "cars_count": cars_count,
                "cars_mean": cars_mean,
                "pedestrian_count": pedestrian_count,
                "pedestrian_mean": pedestrian_mean
            },
            "status": status,
        },
    )
    print(x.text)
