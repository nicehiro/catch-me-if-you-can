import random

from puck_world.envs import AgentWithWheel


class RandomAgent(AgentWithWheel):
    """Chose action randomly.
    """

    def __init__(self,
                 x,
                 y,
                 r,
                 color,
                 agent_type,
                 env):
        super(RandomAgent, self).__init__(x, y, r, color, agent_type)
        self.env = env

    def act(self, state=None):
        return self.env.action_space.sample()
