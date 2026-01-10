import os
import pytest
import torch

from tests import _PATH_DATA
from mnist_cnn.data import corrupt_mnist


@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA),
    reason="Data files not found"
)
def test_data():
    train, test = corrupt_mnist()

    # dataset sizes
    assert len(train) in (30000, 50000), (
        f"Training dataset has wrong size: {len(train)}"
    )
    assert len(test) == 5000, (
        f"Test dataset has wrong size: {len(test)}"
    )

    # shape and label checks (small subset for speed)
    for dataset in (train, test):
        for i in range(min(256, len(dataset))):
            x, y = dataset[i]
            assert x.shape == (1, 28, 28), (
                f"Expected shape (1, 28, 28), got {x.shape}"
            )
            assert int(y) in range(10), (
                f"Label {y} not in range [0, 9]"
            )

    # ensure all labels exist
    train_targets = torch.unique(train.tensors[1])
    test_targets = torch.unique(test.tensors[1])

    assert (train_targets == torch.arange(0, 10)).all(), (
        "Not all labels 0–9 are present in training data"
    )
    assert (test_targets == torch.arange(0, 10)).all(), (
        "Not all labels 0–9 are present in test data"
    )
