class GameState:
    def __init__(self):
        self.period = 1
        self.score = (0,0)
        self.shots = (0,0)

    def copy(self):
        new = GameState()
        new.score = self.score
        new.period = self.period
        new.shots = self.shots
        return new

    def __str__(self):
        return f"P{self.period} {self.score}({self.shots})"

    def __repr__(self):
        return self.__str__()

    def _get_lr(self, delta):
        l,r = (delta[1], delta[0]) if self.period == 2 else delta
        return l,r

    def adjust_score(self, delta):
        l,r = self._get_lr(delta)
        self.score = (self.score[0]+l, self.score[1]+r)
        return

    def adjust_shots(self, delta):
        l,r = self._get_lr(delta)
        self.shots = (self.shots[0]+l, self.shots[1]+r)
        return