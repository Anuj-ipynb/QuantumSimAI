import numpy as np
import pandas as pd
from qiskit_aer.noise import NoiseModel, depolarizing_error

from quantum.simulator import (
    build_random_circuit,
    simulate_distribution,
    classical_fidelity
)

SEED = 42
NUM_SAMPLES = 1000

rng = np.random.default_rng(SEED)
records = []

for i in range(NUM_SAMPLES):
    n_qubits = rng.integers(2, 6)
    depth = rng.integers(5, 25)

    p1 = rng.uniform(0.01, 0.05)
    p2 = rng.uniform(0.05, 0.15)

    qc = build_random_circuit(n_qubits, depth, seed=i)

    ideal_counts = simulate_distribution(
        qc, noise_model=None, seed=i
    )

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p1, 1), ['h']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p2, 2), ['cx']
    )

    noisy_counts = simulate_distribution(
        qc, noise_model=noise_model, seed=i
    )

    fidelity = classical_fidelity(ideal_counts, noisy_counts)

    records.append({
        "n_qubits": n_qubits,
        "depth": depth,
        "p1": p1,
        "p2": p2,
        "distribution_fidelity": fidelity
    })

df = pd.DataFrame(records)
df.to_csv("data/raw/noise_dataset.csv", index=False)

print("Dataset generated: data/raw/noise_dataset.csv")
