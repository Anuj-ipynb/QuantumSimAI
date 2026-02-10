# Quantum AI: Sim-to-Real Noise Surrogate System

This repository contains an end-to-end **Quantum AI system** for studying
noise-induced degradation in quantum circuits and learning fast
machine-learning surrogates for sim-to-real analysis.

## Overview
- Quantum circuit simulation with NISQ-style noise (Qiskit Aer)
- Dataset generation from ideal vs noisy executions
- ML surrogate models (Neural Network, Random Forest)
- Evaluation using KL divergence and RMSE
- FastAPI backend for model serving
- Streamlit frontend for interactive analysis

## Project Structure
