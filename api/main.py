"""FastAPI service that serves penguin species predictions."""
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.model import load_model

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts, release it on shutdown."""
    ml_models["penguins"] = load_model()
    yield
    ml_models.clear()


app = FastAPI(
    title="Penguin Species Classifier",
    description="Predicts penguin species from body measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


class PenguinFeatures(BaseModel):
    """Input schema: raw measurements, exactly as collected in the field."""
    bill_length_mm: float = Field(gt=0, examples=[39.1])
    bill_depth_mm: float = Field(gt=0, examples=[18.7])
    flipper_length_mm: float = Field(gt=0, examples=[181])
    body_mass_g: float = Field(gt=0, examples=[3750])
    island: Literal["Biscoe", "Dream", "Torgersen"]
    sex: Literal["male", "female"] | None = None


class Prediction(BaseModel):
    """Output schema: predicted species and the probability of each class."""
    species: str
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "penguins" in ml_models}


@app.post("/predict", response_model=Prediction)
def predict(features: PenguinFeatures) -> Prediction:
    model = ml_models["penguins"]

    # None -> NaN so the pipeline's imputer recognises it as a missing value.
    row = {k: (np.nan if v is None else v) for k, v in features.model_dump().items()}
    X = pd.DataFrame([row])

    species = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return Prediction(
        species=str(species),
        probabilities={c: round(float(p), 4) for c, p in zip(model.classes_, probabilities)},
    )