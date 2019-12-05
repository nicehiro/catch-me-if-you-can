import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim
from grid_world.envs import MAEAgent


class PolicyNet(nn.Module):
    """Policy Net.
    Three fully connected hiden layers with softmax output.
    """

    def __init__(self, features_n, outputs_n, layer1_n, layer2_n, layer3_n):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(features_n, layer1_n)
        self.layer2 = nn.Linear(layer1_n, layer2_n)
        self.layer3 = nn.Linear(layer2_n, layer3_n)
        self.layer4 = nn.Linear(layer3_n, outputs_n)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = self.layer4(x)
        x = torch.softmax(x, dim=-1)
        return x


class PolicyGradAgent(MAEAgent):
    """The agent model that can use REINFORCE algorithm to train.
    """

    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type,
                 features_n,
                 discounted_value,
                 restore_path,
                 init_value=0.0,
                 learning_rate=0.01,
                 need_restore=False):
        super(PolicyGradAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            color=color,
            env=env,
            name=name,
            default_type=agent_type,
            default_value=init_value
        )
        self.gamma = discounted_value
        self.actions_n = env.action_space.n
        self.features_n = features_n
        self.lr = learning_rate
        self.policy_net = PolicyNet(self.features_n, self.actions_n, 50, 50, 50)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.memory = TransitionList()
        self.rewards = []
        self.actions_prob = []
        self.save_file_path = './model/'
        self.eps = np.finfo(np.float).eps.item()
        if need_restore:
            self.restore(restore_path)

    def act(self, state):
        """Chose action with probability.
        """
        state = torch.tensor([state], dtype=torch.float)
        actions_prob = self.policy_net(state)
        m = distributions.Categorical(actions_prob)
        # print(state)
        # print(actions_prob)
        action = m.sample()
        self.actions_prob.append(m.log_prob(action))
        return action.unsqueeze(0)

    def _discount_and_norm_rewards(self):
        """Calc every state's return when an episode finished.
        """
        G = []
        temp = 0
        for r in self.rewards[::-1]:
            temp = self.gamma * temp + r
            G.insert(0, temp)
        G = torch.tensor(G)
        # G -= G.mean()
        # G /= (G.std() + self.eps)
        return G

    def optimize_model(self):
        G = self._discount_and_norm_rewards()
        loss = []
        for prob, vt in zip(self.actions_prob, G):
            loss.append(- prob * vt)

        # for l in loss:
        #     self.optimizer.zero_grad()
        #     l.backward(retain_graph=True)
        #     self.optimizer.step()

        self.optimizer.zero_grad()
        if len(loss) == 0:
            return
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()

        self.actions_prob.clear()
        self.rewards.clear()
        return loss.item()

    def save(self, name):
        torch.save(self.policy_net.state_dict(), self.save_file_path + name)

    def restore(self, path):
        params = torch.load(path)
        self.policy_net.load_state_dict(params)
