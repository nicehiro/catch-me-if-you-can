import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F

from puck_world.envs import PuckWorldWheel, AgentWithWheel
from torch import optim, distributions
from utils import Transition, ReplayMemory


class DQNet(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(DQNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.l1 = nn.Linear(s_dim, 200)
        self.l2 = nn.Linear(s_dim, 200)
        self.l3 = nn.Linear(200, 200)
        self.l4 = nn.Linear(200, 200)
        self.l5 = nn.Linear(200, 200)
        self.l6 = nn.Linear(200, 200)
        self.left_v = nn.Linear(200, a_dim)
        self.right_v = nn.Linear(200, a_dim)

    def forward(self, x):
        a1 = F.relu(self.l1(x))
        a1 = F.relu(self.l3(a1))
        a1 = F.relu(self.l4(a1))
        a2 = F.relu(self.l2(x))
        a2 = F.relu(self.l5(a2))
        a2 = F.relu(self.l6(a2))
        return self.left_v(a1), self.right_v(a2)


class EGreedyAgent(AgentWithWheel):
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
                 actions_n,
                 discounted_value,
                 memory_capacity=4096,
                 batch_size=512,
                 learning_rate=0.0001,
                 need_restore=False):
        super(EGreedyAgent, self).__init__(x, y, r, color, agent_type)
        self.gamma = discounted_value
        self.features_n = features_n
        self.actions_n = actions_n
        self.lr = learning_rate
        self.save_file_path = 'model/dqn.pkl'
        self.device = 'cpu'
        self.policy_net = DQNet(self.features_n, self.actions_n)
        self.target_net = DQNet(self.features_n, self.actions_n)
        # let target net has the same params as policy net
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        self.memory = []
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 5000
        self.steps_count = 0
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        self.need_exploit = True
        if need_restore:
            self.restore()

    def act(self, state):
        """
        Chose action with probability.
        """
        state = torch.FloatTensor([state])
        sample = random.random()
        # chose action randomly at the beginning, then slowly chose max Q_value
        eps_threhold = self.eps_end + (self.eps_start - self.eps_end) * \
                       math.exp(-1. * self.steps_count / self.eps_decay) \
            if self.need_exploit else 0.01
        self.steps_count += 1
        if sample > eps_threhold:
            with torch.no_grad():
                left_v, right_v = self.policy_net(state)
                l, r = left_v.max(1)[1].view(1, 1).item(), right_v.max(1)[1].view(1, 1).item()
                # print('left: %d\tright: %d' % (l, r))
                return l, r
        else:
            l, r = random.randrange(self.actions_n), random.randrange(self.actions_n)
            return l, r

    def optimize_model(self):
        """
        Train model.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        transitions = self.memory.sample(self.batch_size)
        # batch is ([state], [left_v, right_v], [next_state], [reward])
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device)
        non_final_next_states = torch.cat([torch.tensor([s], dtype=torch.float)
                                           for s in batch.next_state if s is not None])
        state_batch = torch.cat(
            [torch.tensor([s], dtype=torch.float) for s in batch.state])
        left_batch = torch.cat(
            [torch.tensor([[s[0]]], dtype=torch.long) for s in batch.action])
        right_batch = torch.cat(
            [torch.tensor([[s[1]]], dtype=torch.long) for s in batch.action])
        reward_batch = torch.cat(
            [torch.tensor([[s]], dtype=torch.float) for s in batch.reward])
        left_eval, right_eval = self.policy_net(state_batch)
        left_q_eval = left_eval.gather(1, left_batch)
        right_q_eval = right_eval.gather(1, right_batch)
        left_q_next, right_q_next = self.target_net(non_final_next_states)
        left_q_next = left_q_next.max(1)[0].detach()
        right_q_next = right_q_next.max(1)[0].detach()
        left_q_target = (left_q_next * self.gamma) + reward_batch.squeeze()
        right_q_target = (right_q_next * self.gamma) + reward_batch.squeeze()

        loss = F.mse_loss(left_q_eval, left_q_target.unsqueeze(1)) + F.mse_loss(right_q_eval, right_q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        """
        Save trained model.
        """
        torch.save(self.policy_net.state_dict(), self.save_file_path)
        print('Model saved succeed!')

    def restore(self):
        """
        Restore model from saved file.
        """
        self.policy_net.load_state_dict(torch.load(self.save_file_path))
