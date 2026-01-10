import torch

from mnist_cnn.data import corrupt_mnist
from mnist_cnn.model import MyAwesomeModel   # <-- replace with your real model class


def test_training_step():
    model = MyAwesomeModel()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train, _ = corrupt_mnist()

    # single batch
    x, y = train[0]
    x = x.unsqueeze(0)          # (1, 1, 28, 28)
    y = torch.tensor([y])       # (1,)

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
