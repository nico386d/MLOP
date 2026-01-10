import wandb

api = wandb.Api()

artifact_path = (
    "nico386d-danmarks-tekniske-universitet-dtu/"
    "corrupt_mnist/"
    "corrupt_mnist_model:v0"
)

artifact = api.artifact(artifact_path)

artifact.link(
    target_path=(
        "nico386d-danmarks-tekniske-universitet-dtu/"
        "model-registry/"
        "corrupt-mnist-model"
    )
)

artifact.save()
