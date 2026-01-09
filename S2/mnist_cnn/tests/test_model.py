import torch
import pytest

from mnist_cnn.model import MyAwesomeModel  # <-- replace with your real class


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_model_output_shape_parametrized(batch_size):
    model = MyAwesomeModel()

    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)

    assert y.shape == (batch_size, 10), (
        f"Expected output shape ({batch_size}, 10), got {y.shape}"
    )
