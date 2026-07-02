import type { ReactNode } from "react";
import { Link } from "react-router-dom";

// The app shell: a slim, sticky header with the wordmark, over a subtle grid
// backdrop. Everything else renders inside a centered, breathable container.
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app-backdrop min-h-screen">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#0f3460] text-sm font-bold text-white">
              IF
            </span>
            <div className="leading-tight">
              <div className="text-[15px] font-semibold text-[#1a1a2e]">InsightFlow</div>
              <div className="text-[11px] text-slate-500">Experimentation platform</div>
            </div>
          </Link>
          <a
            href="/api/docs"
            target="_blank"
            rel="noreferrer"
            className="text-sm font-medium text-slate-500 hover:text-[#0f3460]"
          >
            API docs ↗
          </a>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-10">{children}</main>
    </div>
  );
}
