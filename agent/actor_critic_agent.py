import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from grid_world.envs import MAEAgent
from utils import ReplayMemory


class ActorCriticNet(nn.Module):
    """
    Actor Critic Net.

    Contains two outputs, one is policy probs, another is value output.
    """

    def __init__(self, features_n, outputs_n):
        super(ActorCriticNet, self).__init__()
        self.layer1 = nn.Linear(features_n, 256)
        # torch.nn.init.normal_(self.layer1.weight)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        # policy net
        self.layer_pi = nn.Linear(256, outputs_n)
        # value net
        self.layer_v = nn.Linear(256, 1)

    def pi(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer_pi(x)
        return F.softmax(x, dim=1)

    def v(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer_v(x)


class ActorCriticAgent(MAEAgent):
    """
    This agent use actor critic algorithm optimize model.
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
                 learning_rate=0.001,
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
        self.gamma = discounted_value
        self.actions_n = env.action_space.n
        self.features_n = features_n
        self.lr = learning_rate
        self.save_file_path = './model/'
        self.net = ActorCriticNet(self.features_n, self.actions_n)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.memory = []
        if need_restore:
            self.restore()

    def act(self, state):
        """
        Chose action with probability.
        """
        state = torch.tensor([state], dtype=torch.float)
        actions_prob = self.net.pi(state)
        # print(state)
        print(actions_prob)
        m = distributions.Categorical(actions_prob)
        action = m.sample()
        # print(action.item())
        return action.item()

    def optimize_model(self):
        """
        Use Actor-Critic TD(0) to train net.
        """
        state, action, state_, reward, done = [], [], [], [], []
        for trans in self.memory:
            state.append(trans[0])
            action.append(trans[1])
            state_.append(trans[2])
            reward.append(trans[3]/100)
            done.append(0.0 if trans[4] else 1.0)
        state = torch.tensor(state, dtype=torch.float)
        state_ = torch.tensor(state_, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1)

        td_target = reward + self.gamma * self.net.v(state_) * done
        td_error = td_target - self.net.v(state)
        pi = self.net.pi(state)
        action = torch.tensor(action).unsqueeze(1)
        pi_a = pi.gather(1, action)
        # use detach() make this tensor a copy
        loss = -torch.log(pi_a) * td_error.detach() + \
            F.smooth_l1_loss(self.net.v(state), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.memory.clear()
        return loss.mean().item()

    def save(self):
        """
        Save trained model.
        """
        torch.save(self.net.state_dict(),
                   self.save_file_path + 'ac.pkl')

    def restore(self):
        """
        Restore model from saved file.
        """
        self.net.load_state_dict(
            torch.load(self.save_file_path + 'ac.pkl'))
