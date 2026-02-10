import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

SHOTS = 1024


def build_random_circuit(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    for _ in range(depth):
        q = rng.integers(0, n_qubits)
        qc.h(q)
        if n_qubits > 1:
            qc.cx(q, (q + 1) % n_qubits)

    qc.measure_all()
    return qc


def simulate_distribution(
    qc: QuantumCircuit,
    noise_model: NoiseModel | None,
    seed: int
) -> dict:
    backend = AerSimulator(
        noise_model=noise_model,
        seed_simulator=seed
    )
    tqc = transpile(qc, backend, optimization_level=0, seed_transpiler=seed)
    result = backend.run(tqc, shots=SHOTS).result()
    return result.get_counts()


def classical_fidelity(p: dict, q: dict) -> float:
    all_keys = set(p) | set(q)
    p_norm = {k: p.get(k, 0) / SHOTS for k in all_keys}
    q_norm = {k: q.get(k, 0) / SHOTS for k in all_keys}
    return sum(np.sqrt(p_norm[k] * q_norm[k]) for k in all_keys)
