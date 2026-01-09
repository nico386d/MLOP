import sys
import torch

if __name__ == "__main__":
    exp1 = sys.argv[1]
    exp2 = sys.argv[2]

    print(f"Comparing run {exp1} to {exp2}")

    sd1 = torch.load(f"{exp1}/trained_model.pt", map_location="cpu")
    sd2 = torch.load(f"{exp2}/trained_model.pt", map_location="cpu")

    # Check keys match
    if sd1.keys() != sd2.keys():
        raise RuntimeError("State dict keys differ, runs are not reproducible")

    # Check tensors match
    for k in sd1.keys():
        if not torch.allclose(sd1[k], sd2[k]):
            raise RuntimeError(f"Difference found in parameter '{k}', runs are not fully reproducible")

    print("✅ Reproducibility test passed: weights are identical")
