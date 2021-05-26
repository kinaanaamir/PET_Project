import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackNetwork(nn.Module):
    def __init__(self):
        super(AttackNetwork, self).__init__()
        self.stacked_model = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 2))

    def forward(self, x):
        return F.log_softmax(self.stacked_model(x), 1)


if __name__ == "__main__":
    model = AttackNetwork()

    inp = torch.randn(64, 3)
    x = model.forward(inp)
    _ = 0
