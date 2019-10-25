import torch.nn as nn
import torch


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
