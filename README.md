<h1 align="center">InsightFlow</h1>
<p align="center">
  <strong>Production-grade A/B testing &amp; statistical experimentation platform.</strong><br>
  Design experiments, randomize users, run rigorous statistics, and ship decisions, not just p-values.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="CI" src="https://github.com/PasadKunal/InsightFlow/actions/workflows/ci.yml/badge.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-107%20passing-2ea44f">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-in%20development-orange">
</p>

---

## Why this exists

Every product team at Google, Meta, Airbnb, and Stripe runs experiments daily, yet
most teams still reach for a lone `t.test()` in a notebook with no infrastructure
around it. That approach silently ignores the things that actually decide whether an
experiment can be trusted: **was it powered? was the split clean? did we peek? did we
test twenty metrics and celebrate the one that hit `p < 0.05`?**

InsightFlow is the infrastructure that answers all of that. It handles the full
experiment lifecycle (**design → randomization → analysis → sequential stopping →
multiple-testing correction → automated reporting**) as deployable software, not a
one-off script.

> This is not a statistics notebook. It's an experimentation platform.

---

## Architecture

```
insightflow/
├── core/          Statistical engine (framework-independent, 100% unit-tested)
│   ├── frequentist.py       t-test · z-test · chi-squared · Mann-Whitney (+ effect sizes)
│   ├── power_analysis.py    sample-size & power calculators
│   ├── randomization.py     deterministic + stratified assignment
│   ├── srm_detector.py      Sample Ratio Mismatch guard rail
│   ├── bayesian.py          Beta-Binomial posteriors, P(best), expected loss
│   ├── sequential.py        SPRT early stopping
│   └── multiple_testing.py  Bonferroni · Benjamini-Hochberg FDR
├── uplift/        Heterogeneous treatment effects (CATE)
│   ├── x_learner.py         X-Learner meta-learner on gradient-boosted trees
│   ├── shap_analysis.py     SHAP attributions: which features drive the effect
│   ├── segment_analyzer.py  responder quantiles + segment ranking
│   └── synthetic_validator.py  PEHE / correlation vs a known ground-truth effect
├── reporting/     Decision-ready reports
│   ├── report_generator.py  ship/no-ship/extend report from core results
│   ├── insight_generator.py free pluggable LLM summaries (Groq · Ollama · template)
│   ├── visualizations.py    Plotly charts (CI · posteriors · SPRT · uplift)
│   ├── pdf_exporter.py      one-page stakeholder PDF (ReportLab)
│   └── scheduler.py         APScheduler weekly digests
├── api/           FastAPI service: experiment CRUD, ingest, results, reports
│   ├── models.py            SQLAlchemy ORM: Experiment · Assignment · Observation
│   ├── service.py           bridges stored rows -> the core engine
│   └── main.py              FastAPI routes (Swagger docs at /docs)
├── frontend/      React + Vite + Tailwind dashboard (list · results · charts · PDF)
├── validation/    500-experiment simulation harness (empirical Type I error + power)
├── infra/         Dockerfiles · compose (api · web · postgres · redis) · nginx
└── tests/         Unit + property tests for every stat function
```

The **`core/` engine is deliberately isolated** from the web and database layers:
numbers go in, trustworthy answers come out. That separation is what lets us validate
it against SciPy and against Monte-Carlo simulation, independent of any server.

---

## Quickstart

```bash
git clone https://github.com/PasadKunal/InsightFlow.git
cd InsightFlow

python3 -m venv ifvenv
source ifvenv/bin/activate          # Windows: ifvenv\Scripts\activate
pip install -e ".[dev]"

python examples/quickstart.py        # runs a full experiment end-to-end
pytest                               # 107 tests, incl. empirical power validation
```

Want the ML layer too? `pip install -e ".[uplift]" && python examples/uplift_demo.py`

### Running the API

```bash
pip install -e ".[api]"
uvicorn insightflow.api.main:app --reload      # -> http://127.0.0.1:8000/docs
```

The service defaults to **SQLite** (zero setup). To develop against **PostgreSQL**:

```bash
docker compose -f infra/docker-compose.yml up -d
export DATABASE_URL="postgresql+psycopg://insightflow:insightflow@localhost:5432/insightflow"
uvicorn insightflow.api.main:app --reload
```

**The experiment lifecycle over HTTP.** Create → assign → observe → read results:

| Method & path | Purpose |
|---|---|
| `POST /experiments` | Create an experiment (auto power analysis) |
| `GET /experiments` · `GET /experiments/{id}` | List / fetch |
| `PATCH /experiments/{id}/status` | Start, stop, or complete |
| `POST /experiments/{id}/assign` | Deterministically assign a user to an arm |
| `POST /experiments/{id}/observe` · `.../observe/bulk` | Ingest metric values |
| `GET /experiments/{id}/results` | Full analysis: **SRM + frequentist + Bayesian + a ship/hold call** |
| `GET /experiments/{id}/report` | Decision report + plain-English narrative summary |
| `GET /experiments/{id}/report.pdf` | One-page stakeholder PDF |
| `GET /experiments/{id}/charts` | Plotly figures (JSON) for the dashboard |

**Free LLM summaries.** The narrative uses a pluggable backend, no paid API required:
`template` (the default: deterministic, no key, always works), `groq` (free Llama 3.3 70B;
set `INSIGHTFLOW_LLM=groq` + `GROQ_API_KEY`), or `ollama` (100% local). Any failure
falls back to the template, so a report always gets a summary.

### Running the dashboard

A refined React + Vite + Tailwind UI: experiment list, a design-an-experiment modal,
a one-click **⚡ Simulate data** button, and a results view with the recommendation
banner, stat cards, Plotly charts, a narrative summary, and a PDF download.

```bash
uvicorn insightflow.api.main:app --reload    # backend on :8000

cd frontend
npm install
npm run dev                                   # dashboard on http://localhost:5173
```

Create an experiment → click **Simulate data** → watch the full analysis appear.

### Is the engine actually correct? (validation)

Rather than trust the math, InsightFlow *measures* it. The harness runs hundreds of
simulated experiments and checks that the empirical error rates match the promises:

```bash
python validation/simulation_harness.py
```

```
  [PASS] Two-proportion z-test    Type I error   target=0.05  empirical=0.046   (n=500)
  [PASS] Two-proportion z-test    Power          target=0.80  empirical=0.808   (n=500)
  [PASS] Welch's t-test           Type I error   target=0.05  empirical=0.060   (n=500)
  [PASS] Welch's t-test           Power          target=0.80  empirical=0.832   (n=500)
  ALL CHECKS PASSED ✓
```

Type I error stays at or below α and power lands on target. Empirical proof, not just theory.

### The whole stack in Docker

```bash
docker compose -f infra/docker-compose.yml up --build
# dashboard → http://localhost:8080   ·   API docs → http://localhost:8000/docs
```

Brings up the dashboard (nginx), the API (uvicorn), **PostgreSQL**, and **Redis**:
the production-shaped setup. With Redis present, repeat result/report/chart loads are
served from cache (watch the `X-Cache: HIT` response header) for sub-2s dashboard loads.

### A 20-second taste

```python
from insightflow.core import sample_size_for_proportion, proportion_ztest, detect_srm

# 1. Design: how many users to detect a 10% relative lift on a 10% baseline?
plan = sample_size_for_proportion(baseline_rate=0.10, minimum_detectable_effect=0.10)
print(plan)          # Need 14,751 per arm (29,502 total) ...

# 2. Guard: is the observed split healthy, or did the pipeline break?
print(detect_srm({"control": 14_752, "treatment": 14_750}))   # No SRM ✔

# 3. Analyze: did treatment actually move the rate?
result = proportion_ztest(1450, 14_752, 1680, 14_750)
print(result.summary())      # SIGNIFICANT (p=...) | relative lift=... | 95% CI [...]
print(result.significant)    # True
```

---

## What's implemented

| Capability | Module | Notes |
|---|---|---|
| Two-sample **t-test** (Welch + Student) | `core/frequentist.py` | Cohen's d + CI on the mean difference |
| **Two-proportion z-test** | `core/frequentist.py` | relative lift + CI on rate difference |
| **Chi-squared** independence test | `core/frequentist.py` | Cramér's V; agrees with the z-test on 2×2 |
| **Mann-Whitney U** | `core/frequentist.py` | rank-biserial effect size for skewed metrics |
| **Sample-size & power** calculators | `core/power_analysis.py` | inverses of each other; empirically validated |
| **Deterministic + stratified** randomization | `core/randomization.py` | stable SHA-256 hashing, per-stratum balance |
| **SRM detection** | `core/srm_detector.py` | strict `p < 0.001` chi-squared guard rail |
| **Bayesian Beta-Binomial** | `core/bayesian.py` | P(treatment best), expected uplift, expected loss, credible intervals |
| **SPRT sequential testing** | `core/sequential.py` | valid early stopping; streaming & batch APIs |
| **Multiple-testing correction** | `core/multiple_testing.py` | Bonferroni (FWER) + Benjamini-Hochberg (FDR) |
| **X-Learner CATE** | `uplift/x_learner.py` | per-user treatment effect via gradient-boosted meta-learner |
| **SHAP effect attribution** | `uplift/shap_analysis.py` | which features drive the treatment response |
| **Segment ranking** | `uplift/segment_analyzer.py` | responder quintiles + known-segment ranking for targeting |
| **Synthetic validation** | `uplift/synthetic_validator.py` | PEHE + correlation vs. known ground-truth effects |
| **500-experiment validation** | `validation/simulation_harness.py` | empirical Type I error + power across the frequentist tests |
| **Redis caching** | `api/cache.py` | data-versioned cache for results/report/charts; in-memory fallback |

Every frequentist test returns a self-describing `TestResult` carrying the estimate,
effect size, confidence interval, and a plain-English verdict, because a p-value with
no context is how experiments get misread. The Bayesian and sequential tests return
their own rich result objects with built-in ship / keep-running recommendations.

---

## Roadmap

- [x] **Phase 1, Core stats:** frequentist tests, power analysis, randomization, SRM, tests
- [x] **Phase 2, Bayesian + SPRT:** Beta-Binomial posteriors, sequential testing, multiple-testing correction
- [x] **Phase 3, Experiment service:** PostgreSQL/SQLite + FastAPI CRUD, ingestion & results API
- [x] **Phase 4, Uplift modeling:** X-Learner + SHAP CATE, segment ranking, synthetic validation
- [x] **Phase 5a, Reporting engine:** ship/hold reports, free LLM summaries, Plotly charts, PDF export, scheduler + report/PDF/chart API endpoints
- [x] **Phase 5b, Dashboard:** refined React + Vite + Tailwind UI (list, create, simulate, results, charts, narrative, PDF)
- [x] **Phase 6, Validation + infra:** 500-experiment simulation harness, Redis caching, Docker (api · web · postgres · redis), GitHub Actions CI

---

## Tech stack

**Statistics** SciPy · Statsmodels · NumPy  ·  **ML** XGBoost · SHAP  ·  **Backend** FastAPI · PostgreSQL · Redis  ·  **Frontend** React · Vite · Tailwind  ·  **Infra** Docker · GitHub Actions

---

<p align="center"><sub>Built by Kunal Pasad · MIT licensed</sub></p>
