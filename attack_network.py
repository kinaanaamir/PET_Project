import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackNetwork(nn.Module):
    def __init__(self):
        super(AttackNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x))


if __name__ == "__main__":
    model = AttackNetwork()

    inp = torch.randn(64, 3)
    x = model.forward(inp)
    _ = 0
