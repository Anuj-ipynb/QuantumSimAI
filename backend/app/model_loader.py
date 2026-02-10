import torch
import joblib
from torch import nn
from pathlib import Path


# -------------------------
# Neural Network loader
# -------------------------
def load_nn(path: str = "models/surrogate.pt"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NN model not found at {path}")

    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# -------------------------
# Random Forest loader
# -------------------------
def load_rf(path: str = "models/rf_surrogate.joblib"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RF model not found at {path}")

    return joblib.load(path)
