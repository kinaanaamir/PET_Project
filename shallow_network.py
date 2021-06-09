import torch
import torch.nn as nn


class ShallowNetwork(nn.Module):
    def __init__(self):
        super(ShallowNetwork, self).__init__()
        self.stacked_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10))

    def forward(self, x):
        return self.stacked_model(x.float())


if __name__ == "__main__":
    model = ShallowNetwork()

    inp = torch.randn(64, 1, 28, 28)
    x = model.forward(inp)
    _ = 0
