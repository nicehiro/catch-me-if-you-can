import torch
import torch.distributions as distributions

from grid_world.envs import MAEAgent
from net.policy_net import PolicyNet


class GreedyProbsAgent(MAEAgent):
    """Chose action use probs.
    """

    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type,
                 features_n,
                 policy_net):
        super(GreedyProbsAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            name=name,
            color=color,
            env=env,
            default_type=agent_type,
            default_value=0.0
        )
        self.features_n = features_n
        self.actions_n = env.action_space.n
        self.policy_net = policy_net

    def act(self, state):
        """
        Chose action use probs .
        """
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float)
            actions_probs = self.policy_net(state)
            m = distributions.Categorical(actions_probs)
            return m.sample().item()
