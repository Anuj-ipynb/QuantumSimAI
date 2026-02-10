

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



SEED = 42
torch.manual_seed(SEED)

DATA_PATH = Path("data/raw/noise_dataset.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "surrogate.pt"


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Run `python generate_data.py` first."
    )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

required_columns = {
    "n_qubits",
    "depth",
    "p1",
    "p2",
    "distribution_fidelity",
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset schema mismatch. Expected columns: {required_columns}"
    )

X = torch.tensor(
    df[["n_qubits", "depth", "p1", "p2"]].values,
    dtype=torch.float32
)

y = torch.tensor(
    df["distribution_fidelity"].values,
    dtype=torch.float32
).unsqueeze(1)


dataset = TensorDataset(X, y)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False
)


model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()  # Fidelity ∈ [0, 1]
)


# -----------------------------
# Training setup
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCHS = 50


model.train()

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0

    for xb, yb in loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(loader)

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"[Epoch {epoch:03d}/{EPOCHS}] "
            f"MSE Loss: {epoch_loss:.6f}"
        )

torch.save(model.state_dict(), MODEL_PATH)

print("\n✅ Training complete")
print(f" Model saved to: {MODEL_PATH.resolve()}")
