import random

import torch

from puck_world.envs import AgentWithWheel
from torch import distributions

from puck_world_wheel.dqn_agent import DQNet


class GreedyAgent(AgentWithWheel):
    """Agent use greedy policy to chose action.
    """

    def __init__(self,
                 x,
                 y,
                 r,
                 color,
                 agent_type,
                 features_n,
                 actions_n,
                 net,
                 load_path):
        super(GreedyAgent, self).__init__(
            x,
            y,
            r,
            color=color,
            agent_type=agent_type
        )
        self.load_path = load_path
        self.features_n = features_n
        self.policy_net = net
        self.policy_net.load_state_dict(torch.load(self.load_path))

    def act(self, state):
        state = torch.tensor([state], dtype=torch.float)
        left_actions_prob, right_actions_prob = self.policy_net(state)
        m1 = distributions.Categorical(left_actions_prob)
        m2 = distributions.Categorical(right_actions_prob)
        # print(state)
        # print(actions_prob)
        left_action = m1.sample()
        right_action = m2.sample()
        return left_action.squeeze(0).item(), right_action.squeeze(0).item()
