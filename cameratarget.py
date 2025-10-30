from datetime import timedelta
import humanize

def ms_str(ms):
    return humanize.precisedelta(timedelta(milliseconds=ms))

class CameraTarget:
    def __init__(self, time, x):
        self.time = time
        self.x = x

    def __str__(self):
        return f"{self.x} @ {ms_str(self.time)}"

    def __repr__(self):
        return self.__str__()

