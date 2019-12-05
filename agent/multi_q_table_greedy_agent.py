import random
import numpy as np


from grid_world.envs import MAEAgent


class QTGreedyAgent(MAEAgent):
    """Agent use greedy policy to chose action.
    """

    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type,
                 restore_path,
                 n_width,
                 n_height):
        super(QTGreedyAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            name=name,
            color=color,
            env=env,
            default_type=agent_type,
            default_value=0.0
        )
        self.n_width = n_width
        self.n_height = n_height
        self.load_path = restore_path
        self.actions_n = env.action_space.n
        self.Q = self.restore()

    def act(self, state):
        visits = self.get_visits(int(state[0]), int(state[1]))
        if visits >= 5:
            return random.randrange(self.actions_n)
        chaser_state, runner_state = self._trans(state)
        observations = self.Q[chaser_state][runner_state]
        max_Q = np.max(observations)
        actions = []
        for i in range(self.actions_n):
            if max_Q == observations[i]:
                actions.append(i)
        if len(actions) == 0:
            return np.argmax(observations)
        return np.random.choice(actions)

    def _xy_to_state(self, x, y):
        return x + self.n_width * y

    def _trans(self, state):
        chaser_x, chaser_y, runner_x, runner_y = state
        chaser_state = self._xy_to_state(chaser_x, chaser_y)
        runner_state = self._xy_to_state(runner_x, runner_y)
        return chaser_state, runner_state

    def restore(self):
        return np.load(self.load_path)
