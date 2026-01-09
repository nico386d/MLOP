import torch
import typer
from pathlib import Path


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory.

    Tip: When running this from anywhere, pass absolute paths or rely on the defaults
    below by running from the package root.
    """
    raw_dir_p = Path(raw_dir)
    processed_dir_p = Path(processed_dir)
    processed_dir_p.mkdir(parents=True, exist_ok=True)

    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(raw_dir_p / f"train_images_{i}.pt"))
        train_target.append(torch.load(raw_dir_p / f"train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(raw_dir_p / "test_images.pt")
    test_target: torch.Tensor = torch.load(raw_dir_p / "test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, processed_dir_p / "train_images.pt")
    torch.save(train_target, processed_dir_p / "train_target.pt")
    torch.save(test_images, processed_dir_p / "test_images.pt")
    torch.save(test_target, processed_dir_p / "test_target.pt")


def corrupt_mnist(
    processed_dir: str | None = None,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST.

    This version is robust to the current working directory by defaulting to
    the repository/package root relative to this file.

    If you want to override, pass processed_dir explicitly (absolute or relative).
    """
    if processed_dir is None:
        # Assumes this file lives in: <repo>/src/<package>/data.py
        # -> parents[2] = <repo>
        repo_root = Path(__file__).resolve().parents[2]
        processed_path = repo_root / "data" / "processed"
    else:
        processed_path = Path(processed_dir).expanduser().resolve()

    train_images = torch.load(processed_path / "train_images.pt")
    train_target = torch.load(processed_path / "train_target.pt")
    test_images = torch.load(processed_path / "test_images.pt")
    test_target = torch.load(processed_path / "test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


def main(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> None:
    """CLI entrypoint to preprocess corrupt MNIST."""
    # Make defaults robust regardless of where you run from:
    repo_root = Path(__file__).resolve().parents[2]
    raw_path = (repo_root / raw_dir).resolve() if not Path(raw_dir).is_absolute() else Path(raw_dir)
    processed_path = (
        (repo_root / processed_dir).resolve() if not Path(processed_dir).is_absolute() else Path(processed_dir)
    )

    preprocess_data(str(raw_path), str(processed_path))


if __name__ == "__main__":
    typer.run(main)
