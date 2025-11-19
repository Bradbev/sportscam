from datetime import timedelta

def ms_str(ms):
    return str(timedelta(milliseconds=ms)).split(".")[0]

class CameraTarget:
    def __init__(self, time=0, x=0, y=0, cut_to=False, zoom=1.0, score=(-1,-1), period=1):
        self.time = time
        self.x = x
        self.y = y
        self.cut_to = cut_to
        self.zoom = zoom
        self.score = score
        self.period = period

    def __str__(self):
        return f"({self.x}, {self.y}, {self.zoom}) @ {ms_str(self.time)}, [{self.score}, {self.period}], {self.cut_to}"

    def __repr__(self):
        return self.__str__()

