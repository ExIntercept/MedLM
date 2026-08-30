import { LogOut, UserRound } from "lucide-react";
import { cn } from "../lib/cn";

const TABS = [
  { key: "consultation", label: "Active Consultation" },
  { key: "records", label: "Medical Records & History" },
  { key: "prescriptions", label: "Prescriptions" },
  { key: "evidence", label: "Evidence Explorer" },
  { key: "evaluation", label: "Model Evaluation & Proof" },
];

function StatusPill({ label }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-border bg-white px-3 py-1 text-[11px] font-semibold text-slate-600">
      <span className="status-dot" />
      {label}
    </span>
  );
}

export default function Header({ username = "Clinician", activeTab, onTabChange, onSignOut }) {
  return (
    <header className="rounded-xl border border-slate-border bg-white px-6 py-4 shadow-card">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-[20px] font-bold tracking-tight text-clinical-ink">MedIntake AI</h1>
          <p className="mt-0.5 text-sm text-clinical-muted">
            Hybrid retrieval-augmented reasoning with evidence verification and safety guardrails.
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            <StatusPill label="System Online" />
            <StatusPill label="HIPAA Local Storage" />
            <StatusPill label="Evidence Grounded" />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 rounded-lg border border-slate-border bg-white px-3 py-2 text-sm font-semibold text-slate-600">
            <UserRound size={16} className="text-clinical-teal" />
            {username}
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

      <nav className="mt-4 flex gap-1 border-t border-slate-border pt-3">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => onTabChange(tab.key)}
            className={cn(
              "rounded-t-lg border-b-2 px-3 py-2 text-sm font-semibold transition",
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
