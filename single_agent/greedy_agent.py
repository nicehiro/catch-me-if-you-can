import math
import random

import numpy as np


class GreedyAgent():

    def __init__(self,
                 actions_n,
                 gamma=0.9,
                 need_reload=False,
                 reload_path=None):
        self.actions_n = actions_n
        self.save_path = 'chaser_single.npy'
        self.Q = self.restore()

    def act(self, state):
        sample = random.random()
        if sample > 0.1:
            observations = self.Q[state]
            max_Q = np.max(observations)
            actions = []
            for i in range(self.actions_n):
                if max_Q == observations[i]:
                    actions.append(i)
            if len(actions) == 0:
                return np.argmax(observations)
            return np.random.choice(actions)
        else:
            return random.randrange(self.actions_n)

    def restore(self):
        return np.load(self.save_path)