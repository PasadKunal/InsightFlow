"""End-to-end tests for the FastAPI service.

Each test runs against a fresh in-memory SQLite database, wired in by overriding the
``get_db`` dependency. That keeps the tests fast, isolated, and free of any external
database — while still exercising the *real* ASGI app, routes, ORM, and statistics.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from insightflow.api.database import Base, get_db
from insightflow.api.main import app


@pytest.fixture
def client():
    # A single shared in-memory database for the lifetime of one test.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_experiment(client, **overrides):
    body = {
        "name": "Checkout redesign",
        "metric_type": "proportion",
        "baseline_rate": 0.10,
        "minimum_detectable_effect": 0.10,
    }
    body.update(overrides)
    resp = client.post("/experiments", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── meta ─────────────────────────────────────────────────────────────────────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── experiment CRUD ──────────────────────────────────────────────────────────
def test_create_runs_power_analysis(client):
    exp = _make_experiment(client)
    # baseline 10%, 10% relative MDE, 80% power -> a known, positive sample size.
    assert exp["required_sample_size_per_arm"] is not None
    assert exp["required_sample_size_per_arm"] > 1000
    assert exp["status"] == "draft"


def test_list_and_get(client):
    exp = _make_experiment(client)
    assert any(e["id"] == exp["id"] for e in client.get("/experiments").json())
    assert client.get(f"/experiments/{exp['id']}").json()["id"] == exp["id"]


def test_get_missing_is_404(client):
    assert client.get("/experiments/does-not-exist").status_code == 404


def test_status_transition(client):
    exp = _make_experiment(client)
    resp = client.patch(f"/experiments/{exp['id']}/status", json={"status": "running"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_delete(client):
    exp = _make_experiment(client)
    assert client.delete(f"/experiments/{exp['id']}").status_code == 204
    assert client.get(f"/experiments/{exp['id']}").status_code == 404


# ── assignment ───────────────────────────────────────────────────────────────
def test_assignment_is_deterministic_and_idempotent(client):
    exp = _make_experiment(client)
    first = client.post(f"/experiments/{exp['id']}/assign", json={"user_id": "u-1"}).json()
    second = client.post(f"/experiments/{exp['id']}/assign", json={"user_id": "u-1"}).json()
    assert first["variant"] == second["variant"]
    assert first["variant"] in {"control", "treatment"}


# ── data ingestion + analysis ────────────────────────────────────────────────
def test_full_experiment_flow_detects_real_lift(client):
    exp = _make_experiment(client)
    client.patch(f"/experiments/{exp['id']}/status", json={"status": "running"})

    # Simulate a real +40% relative lift (10% -> 14%) on plenty of users.
    rng = np.random.default_rng(0)
    observations = []
    for i in range(8000):
        uid = f"user-{i}"
        # We don't know the arm yet; assign deterministically to pick the true rate.
        variant = client.post(
            f"/experiments/{exp['id']}/assign", json={"user_id": uid}
        ).json()["variant"]
        rate = 0.14 if variant == "treatment" else 0.10
        observations.append({"user_id": uid, "value": float(rng.random() < rate)})

    summary = client.post(
        f"/experiments/{exp['id']}/observe/bulk", json={"observations": observations}
    ).json()
    assert summary["ingested"] == 8000

    results = client.get(f"/experiments/{exp['id']}/results").json()
    assert results["n_control"] + results["n_treatment"] == 8000
    assert not results["srm"]["mismatch_detected"]          # clean split
    assert results["frequentist"][0]["significant"]          # z-test fires
    assert results["bayesian"]["prob_treatment_best"] > 0.95
    assert results["recommendation"].startswith("SHIP")


def test_continuous_metric_flow(client):
    exp = _make_experiment(client, name="Revenue test", metric_type="continuous",
                           baseline_rate=None, minimum_detectable_effect=None)
    rng = np.random.default_rng(1)
    observations = []
    for i in range(4000):
        uid = f"user-{i}"
        variant = client.post(
            f"/experiments/{exp['id']}/assign", json={"user_id": uid}
        ).json()["variant"]
        mean = 55.0 if variant == "treatment" else 50.0
        observations.append({"user_id": uid, "value": float(rng.normal(mean, 20))})
    client.post(f"/experiments/{exp['id']}/observe/bulk", json={"observations": observations})

    results = client.get(f"/experiments/{exp['id']}/results").json()
    assert results["metric_type"] == "continuous"
    assert results["bayesian"] is None                       # Bayesian is proportion-only
    assert results["frequentist"][0]["test_name"].endswith("t-test")
    assert results["recommendation"].startswith("SHIP")


def test_results_needs_data(client):
    exp = _make_experiment(client)
    resp = client.get(f"/experiments/{exp['id']}/results")
    assert resp.status_code == 400  # no observations yet


# ── reporting endpoints (Phase 5) ────────────────────────────────────────────
def _seed_winning_experiment(client):
    exp = _make_experiment(client)
    rng = np.random.default_rng(0)
    observations = []
    for i in range(6000):
        uid = f"user-{i}"
        variant = client.post(
            f"/experiments/{exp['id']}/assign", json={"user_id": uid}
        ).json()["variant"]
        rate = 0.14 if variant == "treatment" else 0.10
        observations.append({"user_id": uid, "value": float(rng.random() < rate)})
    client.post(f"/experiments/{exp['id']}/observe/bulk", json={"observations": observations})
    return exp


def test_report_endpoint_returns_narrative(client):
    exp = _seed_winning_experiment(client)
    report = client.get(f"/experiments/{exp['id']}/report").json()
    assert report["recommendation"] == "SHIP"
    assert isinstance(report["narrative"], str) and len(report["narrative"]) > 40
    assert report["bayesian"]["prob_treatment_best"] > 0.95


def test_report_pdf_downloads(client):
    exp = _seed_winning_experiment(client)
    resp = client.get(f"/experiments/{exp['id']}/report.pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:5] == b"%PDF-"


def test_charts_endpoint_returns_plotly_json(client):
    exp = _seed_winning_experiment(client)
    charts = client.get(f"/experiments/{exp['id']}/charts").json()
    assert "conversion_rate" in charts
    assert "posteriors" in charts
    # A Plotly figure serializes to an object with "data" and "layout".
    assert "data" in charts["conversion_rate"]
