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

    # --------------------------------------------
    # 1. Build deterministic circuit
    # --------------------------------------------
    qc = build_random_circuit(
        n_qubits=req.n_qubits,
        depth=req.depth,
        seed=123
    )

    # --------------------------------------------
    # 2. Ideal simulation
    # --------------------------------------------
    ideal_counts = simulate_distribution(
        qc,
        noise_model=None,
        seed=123
    )

    # --------------------------------------------
    # 3. Noisy simulation
    # --------------------------------------------
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(req.p1, 1), ["h"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(req.p2, 2), ["cx"]
    )

    noisy_counts = simulate_distribution(
        qc,
        noise_model=noise_model,
        seed=123
    )

    # --------------------------------------------
    # 4. True KL divergence
    # --------------------------------------------
    true_kl = kl_divergence(
        ideal_counts,
        noisy_counts,
        req.shots
    )

    true_fidelity = 1.0 - true_kl

    # --------------------------------------------
    # 5. Evaluate Surrogates
    # --------------------------------------------
    results = []

    for name, model in surrogate_models.items():

        # ----- NN -----
        if name == "nn_surrogate":
            x = torch.tensor(
                [[req.n_qubits, req.depth, req.p1, req.p2]],
                dtype=torch.float32
            )
            with torch.no_grad():
                pred_fidelity = model(x).item()

        # ----- RF -----
        elif name == "rf_surrogate":
            pred_fidelity = float(
                model.predict(
                    [[req.n_qubits, req.depth, req.p1, req.p2]]
                )[0]
            )

        # --------------------------------------------
        # Compute RMSE properly
        # --------------------------------------------
        error = pred_fidelity - true_fidelity
        rmse = float(np.sqrt(error ** 2))

        results.append(
            ModelResult(
                model=name,
                kl_divergence=true_kl,
                prediction=pred_fidelity,
                rmse=rmse
            )
        )

    # --------------------------------------------
    # 6. Sort by RMSE (correct ranking)
    # --------------------------------------------
    results.sort(key=lambda r: r.rmse)

    return EvaluationResponse(
        results=results,
        true_fidelity=true_fidelity,
        recommendation=results[0].model
    )
