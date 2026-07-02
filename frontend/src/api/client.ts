// Typed client for the InsightFlow API.
//
// In development, requests go to `/api/...` and Vite proxies them to the FastAPI
// backend on :8000 (see vite.config.ts). One place owns the fetch plumbing so the
// components stay declarative.

const BASE = "/api";

export type MetricType = "proportion" | "continuous";
export type Status = "draft" | "running" | "stopped" | "completed";
export type Recommendation = "SHIP" | "DO NOT SHIP" | "EXTEND" | "INVALID";

export interface Experiment {
  id: string;
  name: string;
  hypothesis: string | null;
  metric_type: MetricType;
  status: Status;
  treatment_fraction: number;
  baseline_rate: number | null;
  minimum_detectable_effect: number | null;
  alpha: number;
  power: number;
  required_sample_size_per_arm: number | null;
  created_at: string;
  updated_at: string;
}

export interface FrequentistTest {
  test_name: string;
  statistic: number;
  p_value: number;
  significant: boolean;
  effect_size: number | null;
  effect_size_name: string | null;
  ci_lower: number | null;
  ci_upper: number | null;
  extra: Record<string, number>;
}

export interface Bayesian {
  prob_treatment_best: number;
  expected_relative_uplift: number;
  expected_loss: number;
  recommendation: string;
}

export interface Srm {
  mismatch_detected: boolean;
  p_value: number;
  observed: Record<string, number>;
  expected: Record<string, number>;
  message: string;
}

export interface Results {
  experiment_id: string;
  metric_type: MetricType;
  status: Status;
  n_control: number;
  n_treatment: number;
  srm: Srm;
  frequentist: FrequentistTest[];
  bayesian: Bayesian | null;
  recommendation: string;
}

export interface Report {
  name: string;
  metric_type: MetricType;
  status: Status;
  n_control: number;
  n_treatment: number;
  recommendation: Recommendation;
  headline: string;
  primary_test_name: string;
  p_value: number;
  significant: boolean;
  effect_size: number | null;
  effect_size_name: string | null;
  ci_lower: number | null;
  ci_upper: number | null;
  bayesian: Bayesian | null;
  srm: { mismatch_detected: boolean; p_value: number } | null;
  top_quintile_lift: number | null;
  required_sample_size_per_arm: number | null;
  narrative: string | null;
}

export interface CreateExperimentInput {
  name: string;
  hypothesis?: string;
  metric_type: MetricType;
  baseline_rate?: number;
  minimum_detectable_effect?: number;
  alpha?: number;
  power?: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* body wasn't JSON; keep the status text */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  listExperiments: () => request<Experiment[]>("/experiments"),
  getExperiment: (id: string) => request<Experiment>(`/experiments/${id}`),
  createExperiment: (body: CreateExperimentInput) =>
    request<Experiment>("/experiments", { method: "POST", body: JSON.stringify(body) }),
  setStatus: (id: string, status: Status) =>
    request<Experiment>(`/experiments/${id}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }),
  deleteExperiment: (id: string) =>
    request<void>(`/experiments/${id}`, { method: "DELETE" }),
  getResults: (id: string) => request<Results>(`/experiments/${id}/results`),
  getReport: (id: string) => request<Report>(`/experiments/${id}/report`),
  getCharts: (id: string) => request<Record<string, any>>(`/experiments/${id}/charts`),
  simulate: (id: string, n_users = 8000, control_rate = 0.1, treatment_rate = 0.13) =>
    request(`/experiments/${id}/simulate`, {
      method: "POST",
      body: JSON.stringify({ n_users, control_rate, treatment_rate }),
    }),
  reportPdfUrl: (id: string) => `${BASE}/experiments/${id}/report.pdf`,
};
