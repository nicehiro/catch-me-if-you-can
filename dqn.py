import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self, features_n, outputs_n, layer1_kernels_n, layer2_kernels_n, layer3_kernels_n):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(features_n, layer1_kernels_n)
        self.layer2 = nn.Linear(layer1_kernels_n, layer2_kernels_n)
        self.layer3 = nn.Linear(layer2_kernels_n, layer3_kernels_n)
        self.layer4 = nn.Linear(layer3_kernels_n, outputs_n)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
