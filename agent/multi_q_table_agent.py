import math
import random

import numpy as np

from grid_world.envs import MAEAgent


class QTEGreedyAgent(MAEAgent):

    def __init__(
        self,
        default_reward,
        name,
        color,
        env,
        agent_type,
        n_width,
        n_height,
        gamma=0.9,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=500,
        learning_rate=0.01,
        need_reload=False,
        reload_path=None
    ):
        super(QTEGreedyAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            name=name,
            color=color,
            env=env,
            default_type=agent_type,
            default_value=0.0
        )
        self.actions_n = self.action_space.n
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_count = 0
        self.lr = learning_rate
        self.save_path = './model/chaser_single.npy'
        self.n_width = n_width
        self.n_height = n_height
        self.Q = np.zeros(
            (n_width * n_height, n_width * n_height, self.actions_n),
            dtype=np.float)

    def act(self, state):
        eps_threhold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_count / self.eps_decay)
        # eps_threhold = 0.1
        self.steps_count += 1
        sample = random.random()
        chaser_state, runner_state = self._trans(state)
        if sample > eps_threhold:
            observations = self.Q[chaser_state][runner_state]
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

    def _xy_to_state(self, x, y):
        return x + self.n_width * y

    def _trans(self, state):
        chaser_x, chaser_y, runner_x, runner_y = state
        chaser_state = self._xy_to_state(chaser_x, chaser_y)
        runner_state = self._xy_to_state(runner_x, runner_y)
        return chaser_state, runner_state

    def update(self, state, action, reward, state_):
        self.Q[np.isnan(self.Q)] = 0.0
        chaser_state, runner_state = self._trans(state)
        chaser_state_, runner_state_ = self._trans(state_)
        self.Q[chaser_state][runner_state][action] += self.lr * \
            (reward + self.gamma * np.max(self.Q[chaser_state_][runner_state_]) -\
                 self.Q[chaser_state][runner_state][action])

    def save(self):
        np.save(self.save_path, self.Q)
    def restore(self):
        self.Q= np.load(self.save_path)
