import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stacked_model = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 2))

    def forward(self, x):
        return self.stacked_model(x)


if __name__ == "__main__":
    model = NeuralNetwork()

    inp = torch.randn(64, 3)
    x = model.forward(inp)
    _ = 0