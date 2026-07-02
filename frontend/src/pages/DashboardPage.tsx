import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type Experiment } from "../api/client";
import { useAsync } from "../hooks";
import { CreateExperiment } from "../components/CreateExperiment";
import { Button, Card, Spinner, StatusBadge } from "../components/ui";

export default function DashboardPage() {
  const { data: experiments, error, loading, reload } = useAsync(() => api.listExperiments());
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();

  return (
    <div>
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-[#1a1a2e]">Experiments</h1>
          <p className="mt-1 text-sm text-slate-500">
            Design, run, and read out A/B tests with statistical rigour.
          </p>
        </div>
        <Button onClick={() => setCreating(true)}>+ New experiment</Button>
      </div>

      <div className="mt-8">
        {loading && <Spinner label="Loading experiments…" />}
        {error && <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}

        {experiments && experiments.length === 0 && (
          <Card className="flex flex-col items-center justify-center gap-3 py-16 text-center">
            <div className="text-4xl">🧪</div>
            <div className="text-lg font-medium text-[#1a1a2e]">No experiments yet</div>
            <p className="max-w-sm text-sm text-slate-500">
              Create your first experiment - you'll get an automatic power analysis and a
              one-click way to simulate results.
            </p>
            <Button onClick={() => setCreating(true)} className="mt-2">
              + New experiment
            </Button>
          </Card>
        )}

        {experiments && experiments.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {experiments.map((exp) => (
              <ExperimentCard key={exp.id} exp={exp} />
            ))}
          </div>
        )}
      </div>

      {creating && (
        <CreateExperiment
          onClose={() => setCreating(false)}
          onCreated={(id) => {
            setCreating(false);
            reload();
            navigate(`/experiments/${id}`);
          }}
        />
      )}
    </div>
  );
}

function ExperimentCard({ exp }: { exp: Experiment }) {
  return (
    <Link to={`/experiments/${exp.id}`}>
      <Card className="h-full p-5 transition-shadow hover:shadow-md">
        <div className="flex items-start justify-between gap-3">
          <h3 className="font-semibold text-[#1a1a2e]">{exp.name}</h3>
          <StatusBadge status={exp.status} />
        </div>
        <p className="mt-2 line-clamp-2 min-h-[2.5rem] text-sm text-slate-500">
          {exp.hypothesis || "No hypothesis recorded."}
        </p>
        <div className="mt-4 flex items-center gap-2 text-xs text-slate-400">
          <span className="rounded bg-slate-100 px-2 py-0.5 capitalize text-slate-600">
            {exp.metric_type}
          </span>
          {exp.required_sample_size_per_arm && (
            <span>{exp.required_sample_size_per_arm.toLocaleString()} / arm target</span>
          )}
        </div>
      </Card>
    </Link>
  );
}
