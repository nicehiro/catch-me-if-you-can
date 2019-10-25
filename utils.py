import random
from collections import namedtuple

import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TransitionList(object):

    def __init__(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position += 1

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
        self.position = 0


def get_reverse_action(action):
    if action == 0:
        action = 1
    if action == 1:
        action = 0
    if action == 2:
        action = 3
    if action == 3:
        action = 2
    return action
