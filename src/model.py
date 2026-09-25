"""Model definition and persistence: how the pipeline is built, saved and loaded."""
import shutil
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "penguins_model"

# Types inside our own sklearn pipeline that skops does not trust by default.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def build_model(classifier: ClassifierMixin) -> Pipeline:
    """Full pipeline: preprocessing + classifier, trained and saved as one object."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", classifier),
    ])


def export_model(model: Pipeline, path: Path = MODEL_DIR, input_example=None) -> None:
    """Save the production model to a fixed local folder, replacing any previous one."""
    if path.exists():
        shutil.rmtree(path)
    mlflow.sklearn.save_model(
        model,
        path=str(path),
        input_example=input_example,
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )


def load_model(path: Path = MODEL_DIR) -> Pipeline:
    """Load the production model. Fails with a clear message if it has not been trained."""
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {path}. Run `python -m src.train` first."
        )
    return mlflow.sklearn.load_model(str(path))


if __name__ == "__main__":
    model = load_model()
    sample = pd.DataFrame([{
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750,
        "island": "Torgersen",
        "sex": "male",
    }])
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    print("Prediction:", prediction)
    print("Probabilities:", {c: round(float(p), 3) for c, p in zip(model.classes_, probabilities)})
