from datetime import timedelta
import humanize

def ms_str(ms):
    return humanize.precisedelta(timedelta(milliseconds=ms))

class CameraTarget:
    def __init__(self, time=0, x=0, y=0, cut_to=False, zoom=1.0):
        self.time = time
        self.x = x
        self.y = y
        self.cut_to = cut_to
        self.zoom = zoom

    def __str__(self):
        return f"{self.x} @ {ms_str(self.time)}"

    def __repr__(self):
        return self.__str__()

