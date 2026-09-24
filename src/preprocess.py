"""Data loading and preprocessing for the Palmer Penguins dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "penguins.csv"

TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and keep only the columns we use."""
    df = pd.read_csv(path)
    df = df[FEATURES + [TARGET]]
    # Drop rows where every measurement is missing: nothing useful to learn from them.
    df = df.dropna(subset=NUMERIC_FEATURES, how="all")
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split so every species appears in both sets."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numeric columns, imputation + one-hot for categorical."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nClass counts:\n", df[TARGET].value_counts())