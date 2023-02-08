from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


def load_mnist(
    batch_size: int = 128, shuffle: bool = True, root: str = "../Code/data"
) -> tuple[MNIST, MNIST]:
    """Load MNIST Dataset from memory or download it if it is not found

    Args:
        batch_size (int): Batch Size for DataLoader
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        root (str, optional): Path to store the data. Defaults to "data".

    Returns:
        tuple[MNIST, MNIST]: (train_data, test_data)
    """

    try:
        train = MNIST(
            root=root,
            train=True,
            download=False,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        )

        test = MNIST(
            root=root,
            train=False,
            download=False,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        )
    except RuntimeError:
        train = MNIST(
            root=root,
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        )

        test = MNIST(
            root=root,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        )

    train_data = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    return train_data, test_data
