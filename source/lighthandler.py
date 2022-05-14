import time


class TrafficLight():
    """Dummy class to simulate a change in the traffic light."""

    def __init__(self, timing):
        self.timing = timing
        self.last_changed = time.time()
        self.state = "red"

    def __call__(self):
        now = time.time()
        if self.last_changed + self.timing < now:
            if self.state == "green":
                self.state = "red"
            else:
                self.state = "green"
            self.last_changed = now
