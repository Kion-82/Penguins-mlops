"""Tests for the FastAPI service."""
import pytest

SPECIES = {"Adelie", "Chinstrap", "Gentoo"}


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_predict_returns_species_and_probabilities(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

    body = response.json()
    assert body["species"] == "Adelie"
    assert set(body["probabilities"]) == SPECIES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_accepts_missing_sex(client, valid_payload):
    del valid_payload["sex"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json()["species"] in SPECIES


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("island", "Mars"),
        ("sex", "unknown"),
        ("bill_length_mm", -5),
        ("body_mass_g", "heavy"),
    ],
)
def test_predict_rejects_invalid_values(client, valid_payload, field, bad_value):
    valid_payload[field] = bad_value
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client, valid_payload):
    del valid_payload["island"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422