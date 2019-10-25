import random

from grid_world.envs import MAEAgent


class RandomAgent(MAEAgent):
    """Chose action randomly.
    """

    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type):
        super(RandomAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            name=name,
            color=color,
            env=env,
            default_type=agent_type,
            default_value=0.0
        )
        self.actions_n = self.action_space.n

    def act(self, state):
        return random.randrange(self.actions_n)
