import torch
import torch.distributions as distributions

from grid_world.envs import MAEAgent
from policy_net import PolicyNet


class GreedyProbsAgent(MAEAgent):

    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type,
                 restore_path,
                 features_n):
        super(GreedyProbsAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            name=name,
            color=color,
            env=env,
            default_type=agent_type,
            default_value=0.0
        )
        self.restore_path = restore_path
        self.features_n = features_n
        self.actions_n = env.action_space.n
        self.policy_net = PolicyNet(self.features_n, self.actions_n, 50, 50, 50)
        self.policy_net.load_state_dict(torch.load(self.restore_path))

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float)
            actions_probs = self.policy_net(state)
            m = distributions.Categorical(actions_probs)
            return m.sample().item()
