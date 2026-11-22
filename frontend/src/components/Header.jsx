import { useEffect, useState } from "react";
import { LogOut, Moon, Sun, UserRound } from "lucide-react";
import { cn } from "../lib/cn";
import { getHealth } from "../lib/api";
import { tabsFor } from "../lib/roles";

const ROLE_LABEL = { patient: "Patient", clinician: "Clinician", admin: "Administrator" };

function StatusPill({ label, ok, title }) {
  // ok: null = checking, true = ready, "partial" = degraded but working, false = down
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-lg border border-slate-border bg-surface px-3 py-1 text-[11px] font-semibold text-slate-600"
      title={title || (ok === null ? "Checking…" : ok === "partial" ? "Partly available" : ok ? "Ready" : "Unavailable")}
    >
      <span
        className={cn(
          "inline-block h-[7px] w-[7px] rounded-full",
          ok === null ? "bg-slate-300" : ok === "partial" ? "bg-amber-500" : ok ? "bg-green-600" : "bg-red-500"
        )}
      />
      {label}
    </span>
  );
}

export default function Header({ username, role, activeTab, onTabChange, onSignOut, theme, onToggleTheme }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    if (role === "patient") return undefined;
    let alive = true;
    const load = () =>
      getHealth()
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth({ failed: true }));
    load();
    const id = setInterval(load, 30000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [role]);

  const known = health && !health.failed;
  const pill = (v) => (health === null ? null : known && !!v);

  return (
    <header className="rounded-xl border border-slate-border bg-surface px-6 py-4 shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-[20px] font-bold tracking-tight text-clinical-ink">MedLM</h1>
          <p className="mt-0.5 text-sm text-clinical-muted">
            {role === "patient"
              ? "Your private health assistant. Answers are checked against medical guidelines."
              : "Model-agnostic adversarial defense for medical language models: every answer passes nine safety stages."}
          </p>
          {role !== "patient" && (
          <div className="mt-2 flex flex-wrap gap-2">
            <StatusPill label={known && health.model ? `Model · ${health.model.split("/").pop()}` : "Model"} ok={pill(health?.colab_medgemma)} />
            <StatusPill
              label={known && health.reranker && !health.chromadb ? "Evidence retrieval · keyword only" : "Evidence retrieval"}
              ok={health === null ? null : !known || !health.reranker ? false : health.chromadb ? true : "partial"}
              title={known && health.reranker && !health.chromadb ? "Dense vector index is empty; using BM25 keyword search + reranker" : undefined}
            />
            <StatusPill label={known && health.guard_model ? `Guard · ${health.guard_model}` : "Llama Guard"} ok={pill(health?.guard_model)} />
            <StatusPill label="Output guard" ok={pill(health?.output_guard)} />
          </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onToggleTheme}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-border bg-surface text-clinical-muted transition hover:text-clinical-ink"
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            title={theme === "dark" ? "Light theme" : "Dark theme"}
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <div className="flex items-center gap-2 rounded-lg border border-slate-border bg-surface px-3 py-2 text-sm font-semibold text-slate-600">
            <UserRound size={16} className="text-clinical-teal" />
            {username}
            <span className="rounded-md bg-clinical-teal-tint px-1.5 py-0.5 text-[11px] font-semibold text-clinical-teal-dark">
              {ROLE_LABEL[role] || "Patient"}
            </span>
          </div>
          <button
            type="button"
            onClick={onSignOut}
            className="flex items-center gap-1.5 rounded-lg border border-transparent bg-transparent px-3 py-2 text-sm font-semibold text-clinical-muted transition hover:bg-slate-bg hover:text-slate-700"
          >
            <LogOut size={15} />
            Sign Out
          </button>
        </div>
      </div>

      <nav className="mt-4 flex gap-1 overflow-x-auto border-t border-slate-border pt-3">
        {tabsFor(role).map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => onTabChange(tab.key)}
            className={cn(
              "whitespace-nowrap rounded-t-lg border-b-2 px-3 py-2 text-sm font-semibold transition",
              activeTab === tab.key
                ? "border-clinical-teal text-clinical-teal-dark"
                : "border-transparent text-clinical-muted hover:text-slate-700"
            )}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
