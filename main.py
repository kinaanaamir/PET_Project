import torch
import torchvision
import os
import torchvision.datasets as datasets
from torchvision import transforms
import torch.nn as nn
from network import NeuralNetwork

device = "cuda"
tranform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])


def download_datasets(batch_size_train, batch_size_test):
    train_load = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=tranform),
        batch_size=batch_size_train, shuffle=True)

    test_load = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=tranform),
        batch_size=batch_size_test, shuffle=True)
    return train_load, test_load


def train(dataloader, model, loss_fn, optimizer, epochs):
    size = len(dataloader.dataset)
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("training finished")
    if not os.path.exists("./model_weights"):
        os.mkdir("./model_weights")

    path = "./model_weights/mnist_net.pth"
    torch.save(model.state_dict(), path)


def test(dataloader, model):
    model.load_state_dict(torch.load("./model_weights/mnist_net.pth"))
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy : ", 100 * correct / total)


if __name__ == "__main__":
    if not os.path.exists("./model_weights"):
        os.mkdir("./model_weights")
    train_loader, test_loader = download_datasets(128, 128)
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(train_loader, model, loss_fn, optimizer, 100)
    test(test_loader, model)