import random

import torch

from net.dqn import DQN
from grid_world.envs import MAEAgent


class GreedyAgent(MAEAgent):
    """Agent use greedy policy to chose action.
    """

    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type,
                 load_path,
                 features_n):
        super(GreedyAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            name=name,
            color=color,
            env=env,
            default_type=agent_type,
            default_value=0.0
        )
        self.load_path = load_path
        self.features_n = features_n
        self.policy_net = DQN(self.features_n, env.action_space.n, 50, 50, 50)
        self.policy_net.load_state_dict(torch.load(self.load_path))
        self.actions_n = env.action_space.n

    def act(self, state):
        visits = self.get_visits(int(state[0]), int(state[1]))
        if visits >= 5:
            return random.randrange(self.actions_n)
        with torch.no_grad():
            state = torch.FloatTensor([state])
            action = self.policy_net(state).max(1)[1].view(1, 1)
            return action.item()
