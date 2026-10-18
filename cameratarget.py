from datetime import timedelta

import gamestate

def ms_str(ms):
    return str(timedelta(milliseconds=ms)).split(".")[0]

class CameraTarget:
    # source: "human" (recorded in the app) or "auto" (autotrack.py).  Targets
    # pickled before this field existed have no attribute; treat them as human.
    def __init__(self, time=0, x=0, y=0, cut_to=False, zoom=1.0, game_state=None, source="human"):
        self.time = time
        self.source = source
        self.x = x
        self.y = y
        self.cut_to = cut_to
        self.zoom = zoom
        self.game_state = game_state.copy() if game_state is not None else gamestate.GameState()

    def __str__(self):
        return f"({self.x}, {self.y}, {self.zoom}) @ {ms_str(self.time)}, [{self.game_state}], {self.cut_to}"

    def __repr__(self):
        return self.__str__()

