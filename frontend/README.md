# InsightFlow Dashboard

The web UI for InsightFlow — a refined, light-themed React dashboard for designing
experiments, simulating traffic, and reading out results with charts, a
ship/hold recommendation, and a plain-English summary.

**Stack:** React 19 · TypeScript · Vite 7 · Tailwind CSS 4 · Plotly.

## Develop

The dashboard talks to the FastAPI backend, so start that first:

```bash
# from the repo root
uvicorn insightflow.api.main:app --reload        # backend on :8000
```

Then, in this folder:

```bash
npm install
npm run dev            # dashboard on http://localhost:5173
```

Vite proxies `/api/*` to the backend on :8000 (see `vite.config.ts`), so the browser
makes clean same-origin requests. Open http://localhost:5173, create an experiment,
and click **⚡ Simulate data** to see the full analysis instantly.

## Build

```bash
npm run build          # type-checks and bundles to dist/
npm run preview        # serve the production build on http://localhost:4173
```

## Structure

```
src/
├── api/client.ts        typed API client (mirrors the FastAPI schemas)
├── hooks.ts             useAsync — tiny data-fetching hook with reload
├── components/
│   ├── Layout.tsx       app shell (header + backdrop)
│   ├── ui.tsx           Card, StatusBadge, StatCard, Button, formatters
│   ├── Plot.tsx         Plotly wrapper (renders backend figure JSON)
│   └── CreateExperiment.tsx   design-an-experiment modal
└── pages/
    ├── DashboardPage.tsx      experiment list + create
    └── ExperimentPage.tsx     results, charts, recommendation, narrative, PDF
```
