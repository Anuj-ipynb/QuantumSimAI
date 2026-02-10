
# Quantum AI: Sim-to-Real Noise Surrogate System

This repository contains an end-to-end Quantum AI system for analyzing noise-induced degradation in quantum circuits and learning machine-learning surrogate models to approximate sim-to-real gaps efficiently.

The project targets near-term (NISQ) quantum systems, where noise dominates circuit behavior and repeated noisy simulation is computationally expensive.

---

## Motivation

Near-term quantum computers suffer from noise arising from imperfect gate operations, decoherence, and hardware-level imperfections. While ideal (noiseless) quantum simulations are easy to compute, noisy simulations are significantly more expensive and scale poorly with circuit size.

This project explores the use of classical machine-learning surrogate models to learn the mapping:

```

(circuit size, circuit depth, noise parameters)
→ noise-induced degradation

```

Once trained, these surrogates provide fast approximations of sim-to-real behavior without requiring full noisy quantum simulation for every configuration.

---

## Quantum Setup

### Circuits
- Randomly generated quantum circuits
- Number of qubits: 2–6
- Circuit depth: 5–30
- Gates used:
  - Hadamard (H)
  - CNOT (CX)
- Measurement in the computational basis

### Noise Model
- NISQ-style depolarizing noise
- Simulated using Qiskit Aer
- Noise parameters:
  - 1-qubit gate error: p₁ ∈ [0.01, 0.05]
  - 2-qubit gate error: p₂ ∈ [0.05, 0.20]

Each circuit is simulated twice:
1. Ideal (noiseless) execution
2. Noisy execution with depolarizing noise

---

## Sim-to-Real Metric

The sim-to-real gap is quantified using Kullback–Leibler (KL) divergence between the ideal and noisy output probability distributions:

```

KL(P_ideal || P_noisy)

```

KL divergence measures how much noise alters the observable behavior of a quantum circuit and is treated as a physics-level property of the system, independent of any surrogate model.

For learning stability, surrogate models are trained to predict:

```

distribution_fidelity = 1 − KL(P_ideal || P_noisy)

```

---

## Dataset

Each data point consists of:

### Inputs
- n_qubits: number of qubits
- depth: circuit depth
- p1: 1-qubit depolarizing error
- p2: 2-qubit depolarizing error

### Target
- distribution_fidelity ∈ [0, 1]

The dataset is:
- synthetically generated
- deterministic (fixed random seeds)
- stored as CSV
- not committed to the repository (fully reproducible from code)

---

## Surrogate Models

Two surrogate models are trained and compared.

### Neural Network (MLP)
- Input dimension: 4
- Architecture: two hidden layers (32 → 32)
- Activation: ReLU
- Output: sigmoid-scaled fidelity
- Loss function: Mean Squared Error (MSE)

### Random Forest Regressor
- Classical baseline model
- Trained on the same dataset
- Used for comparison against the neural surrogate

---

## Evaluation

Two types of metrics are reported.

### Physics-Level Metric
- KL divergence
- Measures noise severity of the quantum system
- Identical for all surrogate models under the same configuration

### Model-Level Metric
- RMSE (Root Mean Squared Error)
- Measures surrogate prediction accuracy
- Used to compare different surrogate models

The surrogate with the lowest RMSE is considered the best-performing model.

---

## System Architecture

```

quantum/     → quantum circuit simulation and noise modeling
training/    → dataset generation, training, evaluation
backend/     → FastAPI service for inference and evaluation
frontend/    → Streamlit user interface
data/        → generated datasets (ignored in git)
models/      → trained models (ignored in git)

````

### Backend
- Implemented with FastAPI
- Endpoints:
  - /predict: fast surrogate inference
  - /evaluate: full sim-to-real evaluation

### Frontend
- Implemented with Streamlit
- Interactive control of circuit and noise parameters
- Visualizes:
  - KL divergence (system-level metric)
  - RMSE comparison across surrogate models

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
````

### Generate dataset

```bash
python generate_data.py
```

### Train surrogate models

```bash
python training/train_surrogate.py
python training/train_rf.py
```

### Start backend

```bash
uvicorn backend.app.main:app --reload
```

### Start frontend

```bash
streamlit run frontend/app.py
```

---

## Reproducibility

* Fixed random seeds
* Deterministic dataset generation
* Explicit noise models
* No reliance on real quantum hardware

All results can be reproduced end-to-end from source.

---

## Limitations

* Uses simulated noise models only
* Restricted circuit family (H and CX gates)
* Small qubit counts (NISQ regime)
* Surrogate models approximate, not replace, full simulation

---

## Future Work

* Additional noise channels (readout, amplitude damping)
* Larger circuit families
* Hardware validation
* Uncertainty estimation
* More expressive surrogate architectures

---

## Disclaimer

This project is intended for research and educational purposes. It does not claim hardware-level accuracy and does not replace full quantum simulation or experimental benchmarking.


