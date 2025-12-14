from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    product: str
    quantity: str
    confidence: float | None = None


class PrediksiResponse(BaseModel):
    """Alias Bahasa Indonesia untuk respons prediksi."""

    product: str
    quantity: str
    confidence: float | None = None
