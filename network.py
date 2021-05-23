import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stacked_model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50176, 128),
            nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, x):
        return F.log_softmax(self.stacked_model(x), 1)


if __name__ == "__main__":
    model = NeuralNetwork()

    inp = torch.randn(64, 1, 28, 28)
    x = model.forward(inp)
    _ = 0
