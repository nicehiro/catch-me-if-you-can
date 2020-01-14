import torch
from puck_world_no_wheel.envs import Agent, AgentType
import random
import math
import numpy


class RandomAgent(Agent):
    def __init__(self, x, y, r, color, agent_type, env):
        super(RandomAgent, self).__init__(x, y, r, color, agent_type)
        self.env = env


    def act(self, epi):
        sample = random.random()
        if sample > 0.5:
            return self.env.action_space.sample()
        if sample > 0.25:
            return random.random() * 180
        else:
            return random.random() * -180
