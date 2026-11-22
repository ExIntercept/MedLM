import { useState } from "react";
import { Activity, Cpu, LayoutDashboard, ScrollText, ShieldCheck, Users as UsersIcon } from "lucide-react";
import { cn } from "../../lib/cn";
import PipelineLog from "../Security/PipelineLog";
import SecurityConsole from "../Security/SecurityConsole";
import Overview from "./Overview";
import Users from "./Users";
import Models from "./Models";
import AccessLog from "./AccessLog";

const PAGES = [
  { key: "overview", label: "Overview", Icon: LayoutDashboard },
  { key: "pipeline", label: "Live pipeline", Icon: Activity },
  { key: "users", label: "Users", Icon: UsersIcon },
  { key: "model", label: "Model", Icon: Cpu },
  { key: "defense", label: "Gateway stages", Icon: ShieldCheck },
  { key: "audit", label: "Access log", Icon: ScrollText },
];

export function Card({ title, action, children, className }) {
  return (
    <section className={cn("rounded-xl border border-slate-border bg-surface p-4 shadow-card", className)}>
      {(title || action) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          {typeof title === "string" ? <h2 className="text-[15px] font-semibold text-clinical-ink">{title}</h2> : title}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export default function AdminConsole() {
  const [page, setPage] = useState("overview");
  return (
    <div className="flex flex-col gap-4">
      <nav className="flex gap-1 overflow-x-auto rounded-xl border border-slate-border bg-surface p-1 shadow-card">
        {PAGES.map(({ key, label, Icon }) => (
          <button
            key={key}
            type="button"
            onClick={() => setPage(key)}
            className={cn(
              "flex items-center gap-1.5 whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-semibold transition",
              page === key ? "bg-clinical-teal text-white" : "text-clinical-muted hover:bg-slate-bg hover:text-clinical-ink"
            )}
          >
            <Icon size={14} /> {label}
          </button>
        ))}
      </nav>
      {page === "overview" && <Overview onOpenPipeline={() => setPage("pipeline")} />}
      {page === "pipeline" && (
        <Card title="Live pipeline" action={<span className="text-xs text-clinical-muted">Every request, stage by stage, as it happens</span>}>
          <PipelineLog />
        </Card>
      )}
      {page === "users" && <Users />}
      {page === "model" && <Models />}
      {page === "defense" && <SecurityConsole hidePipeline />}
      {page === "audit" && <AccessLog />}
    </div>
  );
}
