import torch

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.classification.accuracy import MulticlassAccuracy

from tqdm import tqdm

from data_and_model import LeNet5, load_mnist

if __name__ == "__main__":
    device = "mps"
    model = LeNet5().to(device)
    train_data, test_data = load_mnist()

    loss_func = CrossEntropyLoss()

    optim = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = ReduceLROnPlateau(optim, factor=0.3)

    accuracy = MulticlassAccuracy(task="multiclass", num_classes=10, top_k=1)

    for e in tqdm(range(30)):
        train_loss = 0
        test_loss = 0

        y_train = []
        y_hat_train = []
        y_test = []
        y_hat_test = []

        for X, y in train_data:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            y_hat = model(X)
            loss = loss_func(y_hat, y)
            train_loss += loss
            loss.backward()
            optim.step()

        with torch.no_grad():
            preds = []
            targets = []
            for X, y in test_data:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = loss_func(y_hat, y)
                test_loss += loss
                preds.append(y_hat)
                targets.append(y)

        scheduler.step(test_loss)

        train_loss /= len(train_data)
        test_loss /= len(test_data)
        preds = torch.cat(preds).to("cpu")
        targets = torch.cat(targets).to("cpu")
        test_acc = accuracy(preds, targets)

        tqdm.write(
            f"- Epoch {e}: Train loss: {train_loss:>.4f}\t Test loss: {test_loss:>.4f}\t Test Accuracy: {test_acc:>.4f}"
        )

    # Epoch 29: Train loss: 0.0002   Test loss: 0.0454       Test Accuracy: 0.9909
    torch.save(model.state_dict(), "lenet5_trained.pth")
