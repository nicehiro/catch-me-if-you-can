from torch import nn
import torch
from grid_world.envs import MAEAgent
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F


class A2CNet(nn.Module):

    def __init__(self, features_n, actions_n):
        super().__init__()
        self.fc1 = nn.Linear(features_n, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc_probs = nn.Linear(50, actions_n)
        self.fc_value = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def probs(self, x):
        x = self.forward(x)
        x = self.fc_probs(x)
        return torch.softmax(x, dim=-1)
    
    def value(self, x):
        x = self.forward(x)
        return self.fc_value(x)

class A2CAgent(MAEAgent):
    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type,
                 features_n,
                 discounted_value,
                 init_value=0.0,
                 learning_rate=0.01,
                 need_restore=False):
        super(A2CAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            color=color,
            env=env,
            name=name,
            default_type=agent_type,
            default_value=init_value
        )
        self.rewards = []
        self.action_probs = []
        self.states_ = []
        self.gamma = discounted_value
        self.actions_n = env.action_space.n
        self.features_n = features_n
        self.lr = learning_rate
        self.policy_net = A2CNet(self.features_n, self.actions_n)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.save_file_path = './model/a2c_runner.pkl'
        if need_restore:
            self.restore()

    def act(self, state):
        """Chose action with probability.
        """
        state = torch.tensor([state], dtype=torch.float)
        actions_prob = self.policy_net.probs(state)
        m = distributions.Categorical(actions_prob)
        # print(state)
        # print(actions_prob)
        action = m.sample()
        self.action_probs.append(m.log_prob(action))
        return action.unsqueeze(0)

    def _discount_and_norm_rewards(self):
        """Calc every state's return when an episode finished.
        """
        G = []
        temp = 0
        for r in self.rewards[::-1]:
            temp = self.gamma * temp + r
            G.insert(0, temp)
        G = torch.tensor(G, dtype=torch.float)
        # G -= G.mean()
        # G /= (G.std() + self.eps)
        return G

    def optimize_model(self):
        if len(self.rewards) == 0:
            return
        G = self._discount_and_norm_rewards()
        states_ = torch.tensor(self.states_, dtype=torch.float) # [n, ]
        rewards = torch.tensor(self.rewards, dtype=torch.float) # [n, ]
        loss = []
        for prob, r, state_, R in zip(self.action_probs, rewards, states_, G):
            v_ = self.policy_net.value(state_).squeeze()
            td_error = r + self.gamma * v_ - R
            policy_loss = - prob * td_error
            value_loss = F.smooth_l1_loss(r + self.gamma * v_, R)
            loss.append(policy_loss.squeeze() + value_loss)

        self.optimizer.zero_grad()
        loss = torch.mean(torch.stack(loss))
        loss.backward()
        self.optimizer.step()

        self.action_probs.clear()
        self.rewards.clear()
        self.states_.clear()
        
        return loss.item()

    def save(self):
        torch.save(self.policy_net.state_dict(), \
            self.save_file_path)

    def restore(self):
        params = torch.load(self.save_file_path)
        self.policy_net.load_state_dict(params)