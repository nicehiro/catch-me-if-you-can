import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim
from puck_world.envs import AgentWithWheel


class PolicyNet(nn.Module):
    """Policy Net.
    Three fully connected hiden layers with softmax output.
    """

    def __init__(self, features_n, outputs_n):
        super(PolicyNet, self).__init__()
        self.l1 = nn.Linear(features_n, 200)
        self.l2 = nn.Linear(200, 200)
        self.l3 = nn.Linear(200, 200)
        self.l4 = nn.Linear(200, outputs_n)

        self.l4 = nn.Linear(features_n, 200)
        self.l5 = nn.Linear(200, 200)
        self.l6 = nn.Linear(200, 200)
        self.l7 = nn.Linear(200, outputs_n)

    def forward(self, x):
        left = torch.tanh(self.layer1(x))
        left = torch.tanh(self.layer2(left))
        left = torch.tanh(self.layer3(left))
        left = self.layer4(left)
        left = torch.softmax(left, dim=-1)

        right = torch.tanh(self.layer1(x))
        right = torch.tanh(self.layer2(right))
        right = torch.tanh(self.layer3(right))
        right = self.layer4(right)
        right = torch.softmax(right, dim=-1)
        return left, right


class PolicyGradAgent(AgentWithWheel):
    """The agent model that can use REINFORCE algorithm to train.
    """

    def __init__(self,
                 x,
                 y,
                 r,
                 color,
                 agent_type,
                 features_n,
                 actions_n,
                 discounted_value,
                 restore_path,
                 learning_rate=0.01,
                 need_restore=False):
        super().__init__(x, y, r, color, agent_type)
        self.gamma = discounted_value
        self.features_n = features_n
        self.actions_n = actions_n
        self.lr = learning_rate
        self.policy_net = PolicyNet(self.features_n, self.actions_n)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.memory = TransitionList()
        self.rewards = []
        self.left_actions_prob = []
        self.right_actions_prob = []
        self.save_file_path = 'model/pg.pkl'
        self.eps = np.finfo(np.float).eps.item()
        if need_restore:
            self.restore(restore_path)

    def act(self, state):
        """Chose action with probability.
        """
        state = torch.tensor([state], dtype=torch.float)
        left_actions_prob, right_actions_prob = self.policy_net(state)
        m1 = distributions.Categorical(left_actions_prob)
        m2 = distributions.Categorical(right_actions_prob)
        # print(state)
        # print(actions_prob)
        left_action = m1.sample()
        right_action = m2.sample()
        self.left_actions_prob.append(m1.log_prob(left_action))
        self.right_actions_prob.append(m2.log_prob(right_action))
        return left_action.unsqueeze(0), right_action.unsqueeze(0)

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
        left_loss = []
        for prob, vt in zip(self.left_actions_prob, G):
            left_loss.append(- prob * vt)
        right_loss = []
        for prob, vt in zip(self.right_actions_prob, G):
            right_loss.append(- prob * vt)
        loss = left_loss + right_loss
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

        self.left_actions_prob.clear()
        self.right_actions_prob.clear()
        self.rewards.clear()
        return loss.item()

    def save(self):
        torch.save(self.policy_net.state_dict(), self.save_file_path)

    def restore(self):
        params = torch.load(self.save_file_path)
        self.policy_net.load_state_dict(params)
