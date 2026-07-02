"""The InsightFlow HTTP API.

A thin FastAPI layer over the service functions. The endpoints follow the natural
life of an experiment:

    create  ->  assign users  ->  record observations  ->  read results

Interactive docs are served at ``/docs`` (Swagger) and ``/redoc``. Run locally with:

    uvicorn insightflow.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from . import schemas, service
from .cache import cache
from .config import settings
from .database import get_db, init_db


def _cached(db: Session, experiment, kind: str, compute):
    """Return a cached JSON-able result for an experiment, or compute + store it.

    The key embeds the observation count, so any new data transparently invalidates
    the entry. Returns (payload, "HIT"|"MISS") so the route can set an X-Cache header.
    """
    key = f"{kind}:{experiment.id}:{service.observation_count(db, experiment.id)}"
    hit = cache.get(key)
    if hit is not None:
        return hit, "HIT"
    payload = compute()
    cache.set(key, payload)
    return payload, "MISS"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup. In production this would be an Alembic migration step.
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    summary="Production A/B testing & statistical experimentation platform",
    lifespan=lifespan,
)

# Allow the React dashboard (Phase 5) to call the API from the browser during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_experiment(experiment_id: str, db: Session):
    experiment = service.get_experiment(db, experiment_id)
    if experiment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Experiment {experiment_id!r} not found.")
    return experiment


# ── Meta ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health():
    # Report which optional backends are active, so a .env change is easy to verify.
    try:
        from insightflow.reporting import get_provider

        llm = get_provider().name
    except Exception:
        llm = "unavailable"
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "database": "postgres" if settings.database_url.startswith("postgre") else "sqlite",
        "cache": cache.backend,
        "llm": llm,
    }


# ── Experiments ──────────────────────────────────────────────────────────────
@app.post(
    "/experiments",
    response_model=schemas.ExperimentOut,
    status_code=status.HTTP_201_CREATED,
    tags=["experiments"],
)
def create_experiment(payload: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    """Create an experiment. If design parameters are supplied, a power analysis runs
    automatically and the required per-arm sample size is stored on the experiment."""
    return service.create_experiment(db, payload)


@app.get("/experiments", response_model=list[schemas.ExperimentOut], tags=["experiments"])
def list_experiments(db: Session = Depends(get_db)):
    return service.list_experiments(db)


@app.get("/experiments/{experiment_id}", response_model=schemas.ExperimentOut, tags=["experiments"])
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    return _require_experiment(experiment_id, db)


@app.patch(
    "/experiments/{experiment_id}/status",
    response_model=schemas.ExperimentOut,
    tags=["experiments"],
)
def update_status(experiment_id: str, payload: schemas.StatusUpdate, db: Session = Depends(get_db)):
    experiment = _require_experiment(experiment_id, db)
    return service.set_status(db, experiment, payload.status)


@app.delete(
    "/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["experiments"],
)
def delete_experiment(experiment_id: str, db: Session = Depends(get_db)):
    experiment = _require_experiment(experiment_id, db)
    service.delete_experiment(db, experiment)


# ── Assignment & data ingestion ──────────────────────────────────────────────
@app.post(
    "/experiments/{experiment_id}/assign",
    response_model=schemas.AssignmentOut,
    tags=["data"],
)
def assign(experiment_id: str, payload: schemas.AssignRequest, db: Session = Depends(get_db)):
    """Deterministically assign a user to an arm (idempotent)."""
    experiment = _require_experiment(experiment_id, db)
    assignment = service.assign_user(db, experiment, payload.user_id)
    return schemas.AssignmentOut(
        experiment_id=experiment.id, user_id=assignment.user_id, variant=assignment.variant
    )


@app.post("/experiments/{experiment_id}/observe", tags=["data"])
def observe(experiment_id: str, payload: schemas.ObserveRequest, db: Session = Depends(get_db)):
    """Record a single metric observation for a user (auto-assigns if needed)."""
    experiment = _require_experiment(experiment_id, db)
    obs = service.record_observation(db, experiment, payload.user_id, payload.value)
    return {"user_id": obs.user_id, "variant": obs.variant, "value": obs.value}


@app.post(
    "/experiments/{experiment_id}/observe/bulk",
    response_model=schemas.IngestSummary,
    tags=["data"],
)
def observe_bulk(
    experiment_id: str, payload: schemas.BulkObserveRequest, db: Session = Depends(get_db)
):
    """Ingest many observations at once - handy for seeding and demos."""
    experiment = _require_experiment(experiment_id, db)
    return service.bulk_record(db, experiment, payload.observations)


@app.post(
    "/experiments/{experiment_id}/simulate",
    response_model=schemas.IngestSummary,
    tags=["data"],
)
def simulate(
    experiment_id: str, payload: schemas.SimulateRequest, db: Session = Depends(get_db)
):
    """Generate synthetic traffic with a realistic treatment effect (demo convenience)."""
    experiment = _require_experiment(experiment_id, db)
    return service.simulate(db, experiment, payload)


# ── Results ──────────────────────────────────────────────────────────────────
@app.get(
    "/experiments/{experiment_id}/results",
    response_model=schemas.ResultsOut,
    tags=["results"],
)
def results(experiment_id: str, response: Response, db: Session = Depends(get_db)):
    """Run the full analysis: SRM guardrail, frequentist test(s), Bayesian view, and a
    single ship / hold recommendation. Cached per data version for fast repeat loads."""
    experiment = _require_experiment(experiment_id, db)
    try:
        payload, state = _cached(
            db, experiment, "results",
            lambda: service.analyze(db, experiment).model_dump(mode="json"),
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    response.headers["X-Cache"] = state
    return payload


# ── Reporting ────────────────────────────────────────────────────────────────
@app.get("/experiments/{experiment_id}/report", tags=["reporting"])
def report(experiment_id: str, response: Response, db: Session = Depends(get_db)):
    """Full experiment report: recommendation, headline, key stats, and a plain-English
    narrative summary (generated by the free/pluggable LLM layer)."""
    experiment = _require_experiment(experiment_id, db)
    try:
        payload, state = _cached(
            db, experiment, "report", lambda: service.build_report(db, experiment).to_dict()
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    response.headers["X-Cache"] = state
    return payload


@app.get("/experiments/{experiment_id}/report.pdf", tags=["reporting"])
def report_pdf(experiment_id: str, db: Session = Depends(get_db)):
    """Download the experiment report as a one-page PDF for stakeholder distribution."""
    from insightflow.reporting import report_to_pdf_bytes

    experiment = _require_experiment(experiment_id, db)
    try:
        pdf = report_to_pdf_bytes(service.build_report(db, experiment))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    filename = f"insightflow-{experiment.name.replace(' ', '_')}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/experiments/{experiment_id}/charts", tags=["reporting"])
def charts(experiment_id: str, response: Response, db: Session = Depends(get_db)):
    """Plotly figures for the experiment (JSON), ready for the dashboard to render."""
    experiment = _require_experiment(experiment_id, db)
    try:
        payload, state = _cached(
            db, experiment, "charts", lambda: service.build_charts(db, experiment)
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    response.headers["X-Cache"] = state
    return payload
