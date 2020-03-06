class Epsilon:
    def __init__(self, EPSILON_MAX, EPSILON_MIN, DECAY):
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON_MIN = EPSILON_MIN
        self.DECAY = DECAY
        self.EPSILON = EPSILON_MAX

    def __call__(self):
        ''' GET EPSILON '''
        self.EPSILON = self.EPSILON * self.DECAY

        if self.EPSILON <= self.EPSILON_MIN:
            self.EPSILON = self.EPSILON_MIN
        return self.EPSILON

