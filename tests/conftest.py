"""Shared pytest fixtures."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    """A test client with the app fully started, so the model is loaded once."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    """A known Adelie penguin (first row of the dataset)."""
    return {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750,
        "island": "Torgersen",
        "sex": "male",
    }