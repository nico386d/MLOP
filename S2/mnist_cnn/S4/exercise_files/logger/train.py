import matplotlib.pyplot as plt
import torch
import typer
import wandb
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.utils import make_grid

from mnist_cnn.data import corrupt_mnist
from mnist_cnn.model import MyAwesomeModel

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on corrupt MNIST and log metrics + images + ROC + model artifact to W&B."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    last_epoch_targets = None
    last_epoch_preds = None

    for epoch in range(epochs):
        model.train()

        preds_batches, targets_batches = [], []
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        num_batches = 0

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            acc = (y_pred.argmax(dim=1) == target).float().mean().item()

            wandb.log(
                {"train_loss": loss.item(), "train_accuracy": acc, "epoch": epoch},
                step=global_step,
            )
            global_step += 1

            preds_batches.append(y_pred.detach().cpu())
            targets_batches.append(target.detach().cpu())

            epoch_loss_sum += loss.item()
            epoch_acc_sum += acc
            num_batches += 1

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # Log input images as a single grid (avoids shape error [N,C,H,W])
                grid = make_grid(img[:5].detach().cpu(), nrow=5, normalize=True)
                wandb.log({"images": wandb.Image(grid, caption="Input images (grid)")}, step=global_step)

                # Log histogram of gradients
                grad_tensors = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
                if len(grad_tensors) > 0:
                    grads = torch.cat(grad_tensors, dim=0).cpu().numpy()
                    wandb.log({"gradients": wandb.Histogram(grads)}, step=global_step)

        # Epoch-level averages
        wandb.log(
            {
                "epoch_train_loss": epoch_loss_sum / max(num_batches, 1),
                "epoch_train_accuracy": epoch_acc_sum / max(num_batches, 1),
                "epoch": epoch,
            },
            step=global_step,
        )

        # Build ROC plot from the epoch predictions
        preds = torch.cat(preds_batches, dim=0)        # logits [N,10]
        targets = torch.cat(targets_batches, dim=0)    # labels [N]
        probs = torch.softmax(preds, dim=1)            # probabilities [N,10]

        fig, ax = plt.subplots(figsize=(8, 6))
        for class_id in range(10):
            y_true = (targets == class_id).int().numpy()
            y_score = probs[:, class_id].numpy()
            RocCurveDisplay.from_predictions(y_true, y_score, name=f"class {class_id}", ax=ax)
        ax.set_title(f"One-vs-rest ROC curves (epoch {epoch})")

        wandb.log({"roc": wandb.Image(fig)}, step=global_step)
        plt.close(fig)

        # Keep last epoch tensors for final metrics + artifact metadata
        last_epoch_targets = targets
        last_epoch_preds = preds

    # Final metrics (computed on last epoch training data)
    assert last_epoch_targets is not None and last_epoch_preds is not None
    y_true = last_epoch_targets.numpy()
    y_hat = last_epoch_preds.argmax(dim=1).numpy()

    final_accuracy = accuracy_score(y_true, y_hat)
    final_precision = precision_score(y_true, y_hat, average="weighted", zero_division=0)
    final_recall = recall_score(y_true, y_hat, average="weighted", zero_division=0)
    final_f1 = f1_score(y_true, y_hat, average="weighted", zero_division=0)

    wandb.log(
        {
            "final_accuracy": final_accuracy,
            "final_precision": final_precision,
            "final_recall": final_recall,
            "final_f1": final_f1,
        },
        step=global_step,
    )

    # ---- Log the model as an artifact ----
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={
            "accuracy": final_accuracy,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(DEVICE),
        },
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    # Optional: mark artifact as the run output (nice in UI)
    # run.link_artifact(artifact, "model")

    wandb.finish()


if __name__ == "__main__":
    typer.run(train)
