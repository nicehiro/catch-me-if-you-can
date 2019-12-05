import math
import random

import numpy as np


class EGreedyAgent():

    def __init__(self,
                 features_n,
                 actions_n,
                 n_width,
                 n_height,
                 gamma=0.9,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200,
                 learning_rate=0.01,
                 need_reload=False,
                 reload_path=None):
        self.actions_n = actions_n
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.features_n = features_n
        self.steps_count = 0
        self.lr = learning_rate
        self.save_path = 'chaser_single.npy'
        self.Q = np.zeros((n_width * n_height, self.actions_n), dtype=np.float)

    def act(self, state):
        eps_threhold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_count / self.eps_decay)
        # eps_threhold = 0.1
        self.steps_count += 1
        sample = random.random()
        if sample > eps_threhold:
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

    def update(self, state, action, reward, state_):
        self.Q[np.isnan(self.Q)] = 0.0
        self.Q[state][action] += self.lr * \
            (reward + self.gamma * np.max(self.Q[state_]) - self.Q[state][action])
        
    def save(self):
        np.save(self.save_path, self.Q)

    def restore(self):
        self.Q = np.load(self.save_path)
