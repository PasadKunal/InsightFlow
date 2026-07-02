import { useState } from "react";
import { api, type MetricType } from "../api/client";
import { Button } from "./ui";

// A focused modal for designing a new experiment. For proportion metrics the
// baseline + minimum-detectable-effect fields feed the backend's power analysis,
// which returns the required sample size — surfaced back on the experiment page.
export function CreateExperiment({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (id: string) => void;
}) {
  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [metricType, setMetricType] = useState<MetricType>("proportion");
  const [baseline, setBaseline] = useState("0.10");
  const [mde, setMde] = useState("0.10");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!name.trim()) {
      setError("Give the experiment a name.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const exp = await api.createExperiment({
        name: name.trim(),
        hypothesis: hypothesis.trim() || undefined,
        metric_type: metricType,
        baseline_rate: metricType === "proportion" ? Number(baseline) : undefined,
        minimum_detectable_effect: metricType === "proportion" ? Number(mde) : undefined,
      });
      onCreated(exp.id);
    } catch (e) {
      setError((e as Error).message);
      setSubmitting(false);
    }
  }

  const field = "mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-[#0f3460] focus:ring-2 focus:ring-[#0f3460]/10";
  const label = "text-sm font-medium text-slate-700";

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-slate-900/30 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-6 shadow-xl">
        <h2 className="text-lg font-semibold text-[#1a1a2e]">Design an experiment</h2>
        <p className="mt-1 text-sm text-slate-500">
          For conversion metrics, we'll compute the required sample size for you.
        </p>

        <div className="mt-5 space-y-4">
          <div>
            <label className={label}>Name</label>
            <input
              className={field}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Checkout redesign"
              autoFocus
            />
          </div>
          <div>
            <label className={label}>Hypothesis (optional)</label>
            <input
              className={field}
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
              placeholder="The new checkout increases purchase conversion."
            />
          </div>
          <div>
            <label className={label}>Metric type</label>
            <select
              className={field}
              value={metricType}
              onChange={(e) => setMetricType(e.target.value as MetricType)}
            >
              <option value="proportion">Proportion (conversion rate)</option>
              <option value="continuous">Continuous (revenue, time, …)</option>
            </select>
          </div>

          {metricType === "proportion" && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={label}>Baseline rate</label>
                <input
                  className={field}
                  type="number"
                  step="0.01"
                  value={baseline}
                  onChange={(e) => setBaseline(e.target.value)}
                />
              </div>
              <div>
                <label className={label}>Min. detectable lift</label>
                <input
                  className={field}
                  type="number"
                  step="0.01"
                  value={mde}
                  onChange={(e) => setMde(e.target.value)}
                />
              </div>
            </div>
          )}

          {error && <div className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={submitting}>
            {submitting ? "Creating…" : "Create experiment"}
          </Button>
        </div>
      </div>
    </div>
  );
}
