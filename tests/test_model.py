"""Tests for data loading and the trained model."""
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.model import load_model
from src.preprocess import FEATURES, TARGET, load_data, split_data

MIN_TEST_ACCURACY = 0.95


def test_load_data_has_expected_columns_and_no_missing_target():
    df = load_data()
    assert list(df.columns) == FEATURES + [TARGET]
    assert df[TARGET].notna().all()
    assert set(df[TARGET]) == {"Adelie", "Chinstrap", "Gentoo"}


@pytest.mark.parametrize(
    "features, expected",
    [
        ({"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181,
          "body_mass_g": 3750, "island": "Torgersen", "sex": "male"}, "Adelie"),
        ({"bill_length_mm": 46.1, "bill_depth_mm": 13.2, "flipper_length_mm": 211,
          "body_mass_g": 4500, "island": "Biscoe", "sex": "female"}, "Gentoo"),
        ({"bill_length_mm": 46.5, "bill_depth_mm": 17.9, "flipper_length_mm": 192,
          "body_mass_g": 3500, "island": "Dream", "sex": "female"}, "Chinstrap"),
    ],
)
def test_model_predicts_known_species(features, expected):
    model = load_model()
    assert model.predict(pd.DataFrame([features]))[0] == expected


def test_model_meets_accuracy_threshold():
    model = load_model()
    _, X_test, _, y_test = split_data(load_data())
    accuracy = accuracy_score(y_test, model.predict(X_test))
    assert accuracy >= MIN_TEST_ACCURACY, f"Accuracy {accuracy:.3f} below {MIN_TEST_ACCURACY}"


def test_load_model_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        load_model(tmp_path / "does_not_exist")
    