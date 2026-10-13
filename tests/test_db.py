import pytest
from fastapi.testclient import TestClient

import app.db as db
import app.main as main


@pytest.fixture
def client_with_db(tmp_path, monkeypatch):
    """Run the real API against a throwaway SQLite file (no PostgreSQL needed)."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "_engine", None)
    with TestClient(main.app) as client:  # `with` runs the lifespan -> init_db()
        yield client
    monkeypatch.setattr(db, "_engine", None)


PAYLOAD = {
    "age": 30, "weight": 65, "height": 1.7, "income_lpa": 10,
    "smoker": False, "city": "Mumbai", "occupation": "private_job",
}


def test_prediction_is_logged(client_with_db):
    r = client_with_db.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    stats = client_with_db.get("/stats").json()
    assert stats["total_predictions"] == 1
    assert r.json()["predicted_category"] in stats["by_category"]


def test_stats_counts_multiple_requests(client_with_db):
    for _ in range(3):
        client_with_db.post("/predict", json=PAYLOAD)
    assert client_with_db.get("/stats").json()["total_predictions"] == 3


def test_invalid_request_is_not_logged(client_with_db):
    assert client_with_db.post("/predict", json={**PAYLOAD, "age": 0}).status_code == 422
    assert client_with_db.get("/stats").json()["total_predictions"] == 0


def test_predict_still_works_when_database_is_down(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    monkeypatch.setattr(db, "_engine", None)
    client = TestClient(main.app)  # no lifespan: table creation is skipped
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    assert client.get("/stats").status_code == 503
    monkeypatch.setattr(db, "_engine", None)


def test_stats_returns_503_without_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_engine", None)
    assert TestClient(main.app).get("/stats").status_code == 503
