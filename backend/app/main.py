from fastapi import FastAPI
import torch
import numpy as np
from scipy.stats import entropy

from .schemas import (
    NoiseRequest,
    NoiseResponse,
    EvaluationResponse,
    ModelResult
)
from .model_loader import load_nn, load_rf

from quantum.simulator import (
    build_random_circuit,
    simulate_distribution
)
from qiskit_aer.noise import NoiseModel, depolarizing_error

# -------------------------
# App
# -------------------------
app = FastAPI(title="Quantum Noise Surrogate")

# -------------------------
# Load surrogate models
# -------------------------
surrogate_models = {
    "nn_surrogate": load_nn(),
    "rf_surrogate": load_rf(),
}

# -------------------------
# Utility: KL divergence
# -------------------------
def kl_divergence(p: dict, q: dict, shots: int) -> float:
    keys = set(p) | set(q)
    p_vec = np.array([p.get(k, 0) / shots for k in keys]) + 1e-12
    q_vec = np.array([q.get(k, 0) / shots for k in keys]) + 1e-12
    return float(entropy(p_vec, q_vec))


# -------------------------
# Fast surrogate endpoint
# -------------------------
@app.post("/predict", response_model=NoiseResponse)
def predict(req: NoiseRequest):
    model = surrogate_models["nn_surrogate"]

    x = torch.tensor(
        [[req.n_qubits, req.depth, req.p1, req.p2]],
        dtype=torch.float32
    )

    with torch.no_grad():
        y = model(x).item()

    return NoiseResponse(predicted_fidelity=y)


# -------------------------
# Full evaluation endpoint
# -------------------------
@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(req: NoiseRequest):

    # ---- build circuit
    qc = build_random_circuit(
        n_qubits=req.n_qubits,
        depth=req.depth,
        seed=123
    )

    # ---- ideal simulation
    ideal_counts = simulate_distribution(
        qc, noise_model=None, seed=123
    )

    # ---- noisy simulation
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(req.p1, 1), ["h"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(req.p2, 2), ["cx"]
    )

    noisy_counts = simulate_distribution(
        qc, noise_model=noise_model, seed=123
    )

    # ---- true KL divergence
    true_kl = kl_divergence(
        ideal_counts, noisy_counts, req.shots
    )

    results = []

    for name, model in surrogate_models.items():

        if name == "nn_surrogate":
            x = torch.tensor(
                [[req.n_qubits, req.depth, req.p1, req.p2]],
                dtype=torch.float32
            )
            with torch.no_grad():
                pred_fidelity = model(x).item()

        elif name == "rf_surrogate":
            pred_fidelity = model.predict(
                [[req.n_qubits, req.depth, req.p1, req.p2]]
            )[0]

        rmse = abs(pred_fidelity - (1.0 - true_kl))

        results.append(
            ModelResult(
                model=name,
                kl_divergence=true_kl,
                rmse=rmse
            )
        )

    # sort best → worst
    results.sort(key=lambda r: r.kl_divergence)

    return EvaluationResponse(
        results=results,
        recommendation=results[0].model
    )
