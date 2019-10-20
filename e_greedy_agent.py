import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn import DQN
from grid_world.envs import MAEAgent
from utils import ReplayMemory, Transition


class EGreedyAgent(MAEAgent):

    def __init__(self,
                 default_reward,
                 name,
                 color,
                 env,
                 agent_type,
                 features_n,
                 memory_capacity,
                 init_value=0.0,
                 batch_size=128,
                 gamma=0.99,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200):
        super(EGreedyAgent, self).__init__(
            (0, 0),
            default_reward=default_reward,
            color=color,
            env=env,
            name=name,
            default_type=agent_type,
            default_value=init_value
        )
        self.actions_n = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.features_n = features_n
        self.memory_capacity = memory_capacity
        self.memory = ReplayMemory(self.memory_capacity)
        self.steps_count = 0
        self.device = 'cpu'
        # for evaluate Q_value
        self.policy_net = DQN(self.features_n, self.actions_n, 20, 20)
        # evaluate Q_target
        self.target_net = DQN(self.features_n, self.actions_n, 20, 20)
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.save_file_path = './model/'

    def act(self, state):
        """Chose action greedily.
        """
        sample = random.random()
        # chose action randomly at the beginning, then slowly chose max Q_value
        eps_threhold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_count / self.eps_decay)
        self.steps_count += 1
        if sample > eps_threhold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.actions_n)]])

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # batch is ([state], [action], [next_state], [reward])
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_eval = self.policy_net(state_batch).gather(1, action_batch)
        q_next = torch.zeros(self.batch_size, device=self.device)
        q_next[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        q_target = (q_next * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(q_eval, q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, name):
        torch.save(self.target_net.state_dict(), self.save_file_path + name)

    def restore(self, path):
        params = torch.load(path)
        self.target_net.load_state_dict(params)
        self.policy_net.load_state_dict(params)
