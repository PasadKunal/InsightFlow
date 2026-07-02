import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Status } from "../api/client";
import { useAsync } from "../hooks";
import { Plot } from "../components/Plot";
import {
  Button,
  Card,
  RECOMMENDATION_THEME,
  Spinner,
  StatCard,
  StatusBadge,
  fmtNum,
  fmtP,
  fmtPct,
  fmtSignedPct,
} from "../components/ui";

const CHART_TITLES: Record<string, string> = {
  conversion_rate: "Conversion rate by arm",
  posteriors: "Posterior distributions",
  effect_ci: "Treatment effect (95% CI)",
};

export default function ExperimentPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [version, setVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const refresh = () => setVersion((v) => v + 1);

  const experiment = useAsync(() => api.getExperiment(id), [id, version]);
  const results = useAsync(() => api.getResults(id), [id, version]);
  const report = useAsync(() => api.getReport(id), [id, version]);
  const charts = useAsync(() => api.getCharts(id), [id, version]);

  const exp = experiment.data;
  const hasData = !!results.data && !results.error;

  async function simulate() {
    setBusy(true);
    try {
      await api.simulate(id);
      if (exp?.status === "draft") await api.setStatus(id, "running");
      refresh();
    } finally {
      setBusy(false);
    }
  }

  async function changeStatus(status: Status) {
    setBusy(true);
    try {
      await api.setStatus(id, status);
      refresh();
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm("Delete this experiment and all its data?")) return;
    await api.deleteExperiment(id);
    navigate("/");
  }

  if (experiment.loading) return <Spinner label="Loading experiment…" />;
  if (experiment.error || !exp)
    return <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">{experiment.error}</div>;

  return (
    <div className="space-y-6">
      <Link to="/" className="text-sm text-slate-500 hover:text-[#0f3460]">
        ← All experiments
      </Link>

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold text-[#1a1a2e]">{exp.name}</h1>
            <StatusBadge status={exp.status} />
          </div>
          {exp.hypothesis && <p className="mt-1 max-w-2xl text-sm text-slate-500">{exp.hypothesis}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={exp.status}
            disabled={busy}
            onChange={(e) => changeStatus(e.target.value as Status)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm capitalize outline-none focus:border-[#0f3460]"
          >
            {["draft", "running", "stopped", "completed"].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <Button variant="secondary" onClick={simulate} disabled={busy}>
            {busy ? "Working…" : "⚡ Simulate data"}
          </Button>
          {hasData && (
            <a href={api.reportPdfUrl(id)} target="_blank" rel="noreferrer">
              <Button variant="secondary">↓ PDF</Button>
            </a>
          )}
          <Button variant="danger" onClick={remove}>
            Delete
          </Button>
        </div>
      </div>

      {/* Empty state */}
      {!hasData && (
        <Card className="flex flex-col items-center justify-center gap-3 py-16 text-center">
          <div className="text-4xl">📊</div>
          <div className="text-lg font-medium text-[#1a1a2e]">No results yet</div>
          <p className="max-w-md text-sm text-slate-500">
            This experiment hasn't collected any observations. Simulate a batch of realistic
            traffic to see the full analysis — recommendation, charts, and a narrative summary.
          </p>
          {exp.required_sample_size_per_arm && (
            <p className="text-xs text-slate-400">
              Power analysis suggests {fmtNum(exp.required_sample_size_per_arm)} users per arm.
            </p>
          )}
          <Button onClick={simulate} disabled={busy} className="mt-2">
            {busy ? "Simulating…" : "⚡ Simulate data"}
          </Button>
        </Card>
      )}

      {/* Results */}
      {hasData && results.data && report.data && (
        <>
          <RecommendationBanner
            recommendation={report.data.recommendation}
            headline={report.data.headline}
          />

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Sample size"
              value={fmtNum(results.data.n_control + results.data.n_treatment)}
              hint={`${fmtNum(results.data.n_control)} / ${fmtNum(results.data.n_treatment)} split`}
            />
            <StatCard
              label="p-value"
              value={fmtP(results.data.frequentist[0].p_value)}
              tone={results.data.frequentist[0].significant ? "good" : "default"}
              hint={results.data.frequentist[0].significant ? "significant" : "not significant"}
            />
            {results.data.frequentist[0].effect_size != null && (
              <StatCard
                label={results.data.frequentist[0].effect_size_name || "effect size"}
                value={results.data.frequentist[0].effect_size!.toFixed(3)}
              />
            )}
            {results.data.bayesian && (
              <StatCard
                label="P(treatment best)"
                value={fmtPct(results.data.bayesian.prob_treatment_best)}
                tone={results.data.bayesian.prob_treatment_best > 0.9 ? "good" : "default"}
                hint={`expected uplift ${fmtSignedPct(results.data.bayesian.expected_relative_uplift)}`}
              />
            )}
          </div>

          {/* Charts */}
          {charts.data && (
            <div className="grid gap-4 lg:grid-cols-2">
              {Object.entries(charts.data).map(([key, figure]) => (
                <Card key={key} className="p-4">
                  <div className="mb-1 px-1 text-sm font-medium text-slate-600">
                    {CHART_TITLES[key] ?? key}
                  </div>
                  <Plot figure={figure} />
                </Card>
              ))}
            </div>
          )}

          {/* Narrative + tests */}
          <div className="grid gap-4 lg:grid-cols-2">
            {report.data.narrative && (
              <Card className="p-5">
                <div className="text-sm font-semibold text-[#1a1a2e]">Summary</div>
                <p className="mt-2 text-sm leading-relaxed text-slate-600">{report.data.narrative}</p>
              </Card>
            )}
            <Card className="p-5">
              <div className="text-sm font-semibold text-[#1a1a2e]">Statistical tests</div>
              <table className="mt-3 w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-400">
                    <th className="pb-2 font-medium">Test</th>
                    <th className="pb-2 font-medium">p-value</th>
                    <th className="pb-2 font-medium">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {results.data.frequentist.map((t) => (
                    <tr key={t.test_name} className="border-t border-slate-100">
                      <td className="py-2 pr-2 text-slate-700">{t.test_name}</td>
                      <td className="py-2 tabular-nums text-slate-600">{fmtP(t.p_value)}</td>
                      <td className="py-2">
                        {t.significant ? (
                          <span className="text-emerald-600">significant</span>
                        ) : (
                          <span className="text-slate-400">n.s.</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-4 flex items-center gap-2 text-xs">
                <span
                  className={`rounded px-2 py-0.5 ${
                    results.data.srm.mismatch_detected
                      ? "bg-red-50 text-red-700"
                      : "bg-emerald-50 text-emerald-700"
                  }`}
                >
                  SRM: {results.data.srm.mismatch_detected ? "mismatch!" : "healthy"}
                </span>
                <span className="text-slate-400">p = {fmtP(results.data.srm.p_value)}</span>
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function RecommendationBanner({
  recommendation,
  headline,
}: {
  recommendation: keyof typeof RECOMMENDATION_THEME;
  headline: string;
}) {
  const theme = RECOMMENDATION_THEME[recommendation] ?? RECOMMENDATION_THEME.INVALID;
  return (
    <div className={`rounded-2xl ${theme.bg} px-6 py-5 ring-1 ${theme.ring}`}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className={`text-xs font-semibold uppercase tracking-widest ${theme.text} opacity-80`}>
          Recommendation
        </span>
        <span className={`text-xl font-bold ${theme.text}`}>{recommendation}</span>
      </div>
      <p className={`mt-1 text-sm ${theme.text} opacity-90`}>{headline}</p>
    </div>
  );
}
