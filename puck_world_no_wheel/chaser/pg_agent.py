import torch
import random
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

from puck_world_no_wheel.envs import PuckWorld, Agent
from torch import optim, distributions


class ActorCriticNet(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(ActorCriticNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        # self.set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 180 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.001)
            nn.init.constant_(layer.bias, 0.)


class ActorCriticAgent(Agent):
    """
    This agent use actor critic algorithm optimize model.
    """

    def __init__(self,
                 x,
                 y,
                 r,
                 color,
                 agent_type,
                 features_n,
                 discounted_value,
                 learning_rate=0.001,
                 need_restore=False):
        super(ActorCriticAgent, self).__init__(x, y, r, color, agent_type)
        self.gamma = discounted_value
        self.features_n = features_n
        self.lr = learning_rate
        self.save_file_path = 'model/ac.pkl'
        self.net = ActorCriticNet(self.features_n, 1)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.memory = []
        if need_restore:
            self.restore()

    def act(self, state):
        """
        Chose action with probability.
        """
        self.net.training = False
        if random.random() > 0.9:
            return random.random() * 360 - 180
        state = torch.tensor([state], dtype=torch.float)
        mu, sigma, _ = self.net(state)
        m = self.net.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        action = m.sample()
        action = action.numpy().clip(-180, 180)
        action = action[0]
        if action == np.nan:
            action = np.random.randint(-180, 180)
        return action

    def optimize_model(self):
        """
        Use Actor-Critic TD(0) to train net.
        """
        state, action, state_, reward, done = [], [], [], [], []
        for trans in self.memory:
            state.append(trans[0])
            action.append(trans[1])
            state_.append(trans[2])
            reward.append(trans[3])
            done.append(0.0 if trans[4] else 1.0)
        state = torch.tensor(state, dtype=torch.float)
        state_ = torch.tensor(state_, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1)
        action = torch.tensor(action, dtype=torch.float).unsqueeze(1)

        mu, sigma, value = self.net(state)
        _, _, value_ = self.net(state_)
        # Use TD(0)
        # TD-target = reward + gamma * Q(state_)
        td_target = reward + self.gamma * value_ * done
        # TD-error = TD-target - Q(state)
        td_error = td_target - value
        # action probs
        m = self.net.distribution(mu, sigma)
        log_prob = m.log_prob(action)
        # entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        # use detach() get this tensor a copy
        loss = -(log_prob * td_error.detach()) + \
               F.smooth_l1_loss(value, td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.memory.clear()
        return loss.mean().item()

    def save(self):
        """
        Save trained model.
        """
        torch.save(self.net.state_dict(), self.save_file_path)

    def restore(self):
        """
        Restore model from saved file.
        """
        self.net.load_state_dict(torch.load(self.save_file_path))
