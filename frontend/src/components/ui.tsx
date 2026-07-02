// Small presentational building blocks shared across pages. Keeping them here keeps
// the pages focused on data flow rather than class-name soup.

import type { ReactNode } from "react";
import type { Recommendation, Status } from "../api/client";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-2xl border border-slate-200 bg-white shadow-sm shadow-slate-200/50 ${className}`}
    >
      {children}
    </div>
  );
}

const STATUS_STYLES: Record<Status, string> = {
  draft: "bg-slate-100 text-slate-600",
  running: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
  stopped: "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
  completed: "bg-blue-50 text-blue-700 ring-1 ring-blue-200",
};

export function StatusBadge({ status }: { status: Status }) {
  const dot: Record<Status, string> = {
    draft: "bg-slate-400",
    running: "bg-emerald-500",
    stopped: "bg-amber-500",
    completed: "bg-blue-500",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium capitalize ${STATUS_STYLES[status]}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot[status]}`} />
      {status}
    </span>
  );
}

export const RECOMMENDATION_THEME: Record<
  Recommendation,
  { bg: string; text: string; ring: string; label: string }
> = {
  SHIP: { bg: "bg-emerald-600", text: "text-white", ring: "ring-emerald-600/20", label: "Ship it" },
  "DO NOT SHIP": { bg: "bg-red-600", text: "text-white", ring: "ring-red-600/20", label: "Do not ship" },
  EXTEND: { bg: "bg-amber-500", text: "text-white", ring: "ring-amber-500/20", label: "Keep running" },
  INVALID: { bg: "bg-slate-500", text: "text-white", ring: "ring-slate-500/20", label: "Invalid" },
};

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "good" | "bad";
}) {
  const valueColor =
    tone === "good" ? "text-emerald-600" : tone === "bad" ? "text-red-600" : "text-[#1a1a2e]";
  return (
    <Card className="p-5">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${valueColor}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-400">{hint}</div>}
    </Card>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  type = "button",
  disabled = false,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  type?: "button" | "submit";
  disabled?: boolean;
  className?: string;
}) {
  const styles: Record<string, string> = {
    primary: "bg-[#0f3460] text-white hover:bg-[#0c2a4d] shadow-sm",
    secondary: "bg-white text-[#0f3460] ring-1 ring-slate-300 hover:bg-slate-50",
    ghost: "text-slate-600 hover:bg-slate-100",
    danger: "text-red-600 hover:bg-red-50 ring-1 ring-red-200",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${styles[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-slate-500">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-[#0f3460]" />
      {label && <span className="text-sm">{label}</span>}
    </div>
  );
}

// ── formatting helpers ───────────────────────────────────────────────────────
export const fmtP = (p: number) => (p < 0.001 ? p.toExponential(2) : p.toFixed(3));
export const fmtPct = (x: number) => `${(x * 100).toFixed(1)}%`;
export const fmtSignedPct = (x: number) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`;
export const fmtNum = (x: number) => x.toLocaleString();
