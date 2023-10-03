import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LeNet5v1(nn.Module):
    def __init__(self):
        super(LeNet5v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.dropout(self.mp1(out))
        out = self.relu2(self.conv2(out))
        out = self.dropout(self.mp1(out))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(self.fc3(out), dim=1)
        return out


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    stop = False
    patience = 2  # early stopping patience; how long to wait after last time validation loss improved.
    best_val_loss = None
    epochs_no_improve = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # negative log likelihood loss
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        # early stopping check
        if epoch > 1:
            if best_val_loss is None or best_val_loss > loss.item():
                best_val_loss = loss.item()
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_model.pt")
            elif best_val_loss < loss.item():
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print("Early stopping!")
                    stop = True
                    break
    return stop


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Setting some hyperparameters
    train_batch_size = 64
    test_batch_size = 1000
    lr = 0.01
    momentum = 0.5
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    train_dataset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    test_dataset = datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = LeNet5v1().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, num_epochs + 1):
        stop = train(model, device, train_loader, optimizer, epoch)
        if stop:
            break

    model.load_state_dict(torch.load("best_model.pt"))
    test(model, device, test_loader)


if __name__ == "__main__":
    main()
