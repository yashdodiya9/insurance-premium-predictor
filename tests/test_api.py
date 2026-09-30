import pytest
from fastapi.testclient import TestClient
import app.main as main
from app.schema.prediction_response import PredictionResponse

client = TestClient(main.app)


def payload(**overrides):
    data = {
        "age": 30,
        "weight": 65,
        "height": 1.7,
        "income_lpa": 10,
        "smoker": False,
        "city": "Mumbai",
        "occupation": "private_job",
    }
    data.update(overrides)
    return data


def test_home():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "OK"
    assert body["model_loaded"] is True


def test_predict_returns_documented_response_shape():
    """Guards against the response model and the real output drifting apart."""
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predicted_category", "confidence", "class_probabilities"}
    PredictionResponse.model_validate(body)
    assert sum(body["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("city", ["mumbai", "  SURAT  ", "Pithampur"])
def test_predict_handles_city_case_whitespace_and_unknown_cities(city):
    assert client.post("/predict", json=payload(city=city)).status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [
        ("age", 0),
        ("age", 120),
        ("weight", -5),
        ("height", 0),
        ("height", 3),
        ("income_lpa", 0),
        ("occupation", "astronaut"),
        ("smoker", "maybe"),
    ],
)
def test_predict_rejects_invalid_values(field, value):
    r = client.post("/predict", json=payload(**{field: value}))
    assert r.status_code == 422


def test_predict_rejects_missing_field():
    body = payload()
    del body["income_lpa"]
    assert client.post("/predict", json=body).status_code == 422


def test_unexpected_failure_returns_generic_500_without_leaking_details(monkeypatch):
    def boom(_user_input):
        raise RuntimeError("secret internal details")

    monkeypatch.setattr(main, "predict_output", boom)
    r = client.post("/predict", json=payload())
    assert r.status_code == 500
    assert r.json() == {"detail": "Prediction failed. Please try again later."}
    assert "secret" not in r.text
