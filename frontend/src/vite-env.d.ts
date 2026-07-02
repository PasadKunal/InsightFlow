/// <reference types="vite/client" />

// plotly.js-dist-min ships without types; we use it through a thin wrapper.
declare module "plotly.js-dist-min";

// Typed environment variables the dashboard reads at build time.
interface ImportMetaEnv {
  // Base URL of the deployed API (e.g. https://insightflow-api.onrender.com).
  // Unset in dev, where requests fall back to the "/api" Vite proxy.
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
