import math

import torch
import torch.distributions as distributions
import torch.nn.functional as F
import torch.optim as optim

from grid_world.envs import MAEAgent
from net.dqn import DQN
from net.policy_net import PolicyNet


class ActorCriticAgent(MAEAgent):
    """
    This agent has actor net and critic net.
    """

    def __init__(self,
                 default_reward,
                 color,
                 name,
                 env,
                 agent_type,
                 features_n,
                 discounted_value,
                 init_value=0.0,
                 actor_learning_rate=0.01,
                 critic_learning_rate=0.01,
                 need_restore=False):
        super().__init__(
            (0, 0),
            default_reward=default_reward,
            color=color,
            env=env,
            name=name,
            default_type=agent_type,
            default_value=init_value
        )
        self.epi = 0
        self.gamma = discounted_value
        self.actions_n = env.action_space.n
        self.features_n = features_n
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.actor_net = PolicyNet(self.features_n, self.actions_n, 50, 50, 50)
        self.critic_net = DQN(self.features_n, self.actions_n, 50, 50, 50,)
        self.actor_optim = optim.Adam(
            self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.RMSprop(
            self.critic_net.parameters(), lr=self.critic_lr)
        self.save_file_path = './model/'
        if need_restore:
            self.restore()

    def act(self, state):
        """
        Chose action with probability.
        """
        state = torch.tensor([state], dtype=torch.float)
        actions_prob = self.actor_net(state)
        m = distributions.Categorical(actions_prob)
        action = m.sample()
        return action.item()

    def optimize_model(self, state, action, reward, state_, action_):
        """
        Use Actor-Critic TD(0) to train net.
        """
        self.epi += 1
        state = torch.tensor([state], dtype=torch.float)
        state_ = torch.tensor([state_], dtype=torch.float)
        td_error = reward + self.gamma * self.critic_net(state_).squeeze()[action_] \
            - self.critic_net(state).squeeze()[action]

        # optim actor net
        self.actor_optim.zero_grad()
        q_s_a = self.critic_net(state).squeeze()[action]
        actor_loss = - q_s_a * \
            torch.log(self.actor_net(state).squeeze()[action])
        actor_loss.backward()
        self.actor_optim.step()

        # optim critic net
        if self.epi % 10 == 0:
            self.critic_optim.zero_grad()
            critic_loss = F.mse_loss(self.critic_net(state).squeeze()[action],
                                     reward + self.gamma * self.critic_net(state_).squeeze()[action_])
            critic_loss.backward()
            self.critic_optim.step()

    def save(self):
        """
        Save trained model.
        """
        torch.save(self.actor_net.state_dict(),
                   self.save_file_path + 'actor.pkl')
        torch.save(self.critic_net.state_dict(),
                   self.save_file_path + 'critic.pkl')

    def restore(self):
        self.actor_net.load_state_dict(
            torch.load(self.save_file_path + 'actor.pkl'))
        self.critic_net.load_state_dict(
            torch.load(self.save_file_path + 'critic.pkl'))
