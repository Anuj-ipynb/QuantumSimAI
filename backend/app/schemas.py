from pydantic import BaseModel, Field
from typing import List


class NoiseRequest(BaseModel):
    n_qubits: int = Field(..., ge=2)
    depth: int = Field(..., ge=1)
    p1: float = Field(..., ge=0, le=1)
    p2: float = Field(..., ge=0, le=1)
    shots: int = Field(..., ge=1)


class ModelResult(BaseModel):
    model: str
    kl_divergence: float
    prediction: float
    rmse: float


class EvaluationResponse(BaseModel):
    results: list[ModelResult]
    true_fidelity: float
    recommendation: str



class NoiseResponse(BaseModel):
    predicted_fidelity: float
