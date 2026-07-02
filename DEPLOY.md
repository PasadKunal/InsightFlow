# Deploying InsightFlow (free tier)

This guide hosts the platform on **Vercel** (dashboard) + **Render** (API) + a free
**Postgres** database. Everything here is free. Budget about 20 minutes for the first
deploy; there's nothing to pay for and no credit card required for the free tiers.

The pieces:

```
  Browser ──> Vercel (React dashboard) ──> Render (FastAPI API) ──> Neon (Postgres)
```

---

## Step 1 — Create a free Postgres database (Neon)

1. Go to https://neon.tech and sign up (free).
2. Create a project. Neon shows a **connection string** like:
   ```
   postgresql://user:password@ep-xxxx.aws.neon.tech/neondb?sslmode=require
   ```
3. **Important:** change the scheme from `postgresql://` to `postgresql+psycopg://`
   so SQLAlchemy uses the driver this project ships with. Keep `?sslmode=require`:
   ```
   postgresql+psycopg://user:password@ep-xxxx.aws.neon.tech/neondb?sslmode=require
   ```
   Save this string, you'll paste it into Render next.

(Supabase works too, the same scheme change applies.)

---

## Step 2 — Deploy the API on Render

1. Go to https://render.com and sign up, then authorize access to your GitHub.
2. Click **New > Blueprint**, pick the `InsightFlow` repo. Render reads `render.yaml`
   and proposes an `insightflow-api` web service.
3. When prompted for environment variables, set **`DATABASE_URL`** to the Neon string
   from Step 1. (Leave `REDIS_URL` blank, the app falls back to an in-memory cache.)
4. Click **Apply / Deploy**. First build takes a few minutes (it builds the Docker
   image). Tables are created automatically on first startup.
5. When it's live, note the URL, e.g. `https://insightflow-api.onrender.com`. Verify:
   ```
   https://insightflow-api.onrender.com/health   -> {"status":"ok", "database":"postgres", ...}
   https://insightflow-api.onrender.com/docs      -> interactive API docs
   ```

> Note: Render's free service sleeps after ~15 min idle, so the first request after a
> nap takes ~30-50s to wake. Normal for free tier.

---

## Step 3 — Deploy the dashboard on Vercel

1. Go to https://vercel.com and sign up, then import the `InsightFlow` repo.
2. In the import screen, set:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite (auto-detected)
3. Add one **Environment Variable**:
   - `VITE_API_URL` = your Render URL from Step 2 (no trailing slash), e.g.
     `https://insightflow-api.onrender.com`
4. Click **Deploy**. Vercel builds and gives you a URL like
   `https://insight-flow.vercel.app`.

---

## Step 4 — Try it

Open your Vercel URL, create an experiment, and click **Simulate data**. You should
see the recommendation, charts, narrative, and PDF, all served by the Render API and
persisted in Neon.

That Vercel link is the one to put on your resume.

---

## Updating

Both platforms auto-deploy on every push to `main`. If you change `VITE_API_URL`,
trigger a fresh Vercel build (Vite inlines it at build time).

## Troubleshooting

- **Dashboard loads but calls fail** — check `VITE_API_URL` is set on Vercel and has
  no trailing slash; confirm the Render `/health` endpoint responds.
- **API won't start / DB errors** — confirm `DATABASE_URL` uses `postgresql+psycopg://`
  and keeps `?sslmode=require`.
- **First request is slow** — the free Render service was asleep; it wakes in under a
  minute.
