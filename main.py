import torch
import torchvision
import os
from torchvision import transforms
import torch.nn as nn
from network import NeuralNetwork
import torch.nn.functional as F
from shallow_network import ShallowNetwork

device = "cuda"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])


def download_datasets(batch_size_train, batch_size_test):
    train_load = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=transform),
        batch_size=batch_size_train, shuffle=True)

    test_load = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=transform),
        batch_size=batch_size_test, shuffle=True)
    return train_load, test_load


def train(dataloader, model, loss_fn, optimizer, epochs, path="./model_weights/mnist_net.pth", train_shallow=False):
    size = len(dataloader.dataset)
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if train_shallow:
                pred = F.log_softmax(pred, 1)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("training finished")

    torch.save(model.state_dict(), path)


def test(dataloader, model, path="./model_weights/mnist_net.pth", shallow=False):
    model.load_state_dict(torch.load(path))
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if shallow:
                outputs = F.log_softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy : ", 100 * correct / total)


if __name__ == "__main__":
    if not os.path.exists("./model_weights"):
        os.mkdir("./model_weights")
    train_loader, test_loader = download_datasets(128, 128)
    loss_fn = nn.CrossEntropyLoss()

    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(train_loader, model, loss_fn, optimizer, 50)
    test(test_loader, model)

    shallow_model = ShallowNetwork().to(device)
    optimizer = torch.optim.Adam(shallow_model.parameters(), lr=1e-3)
    train(train_loader, shallow_model, loss_fn, optimizer, 50, path="./model_weights/mnist_net_shallow_model.pth",
          train_shallow=True)
    test(test_loader, shallow_model, path="./model_weights/mnist_net_shallow_model.pth",
         shallow=True)
    # train(train_loader)
    # a = torch.empty(60000, 3, dtype=torch.float)
    # for batch, (X, y) in enumerate(train_loader):
    #     # Compute prediction and loss
    #     X, y = X.to(device), y.to(device)
    #     pred = model(X)
    #     # a[batch*128:pred] =
