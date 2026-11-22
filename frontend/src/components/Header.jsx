import { useEffect, useRef, useState } from "react";
import { ChevronDown, LogOut, Moon, ShieldPlus, Sun } from "lucide-react";
import { cn } from "../lib/cn";
import { getHealth } from "../lib/api";
import { tabsFor } from "../lib/roles";
import { shortModel } from "./Admin/FlowMap";

const ROLE_LABEL = { patient: "Patient", clinician: "Clinician", admin: "Administrator" };

/** Close a popover when clicking outside it. */
function useOutside(ref, onOutside) {
  useEffect(() => {
    const h = (e) => ref.current && !ref.current.contains(e.target) && onOutside();
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [ref, onOutside]);
}

function Dot({ state }) {
  return (
    <span
      className={cn(
        "inline-block h-2 w-2 rounded-full",
        state === null ? "bg-slate-300" : state === "partial" ? "bg-amber-500" : state ? "bg-emerald-500" : "bg-red-500"
      )}
    />
  );
}

/** One button summarising system health; details in a small popover. */
function SystemStatus() {
  const [health, setHealth] = useState(null);
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false));

  useEffect(() => {
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
  }, []);

  const known = health && !health.failed;
  const items = [
    { label: "Medical model", detail: known ? shortModel(health.model) : "", state: known ? !!health.colab_medgemma : false },
    {
      label: "Evidence search",
      detail: known && health.reranker && !health.chromadb ? "keyword search + reranker" : "",
      state: !known || !health.reranker ? false : health.chromadb ? true : "partial",
    },
    { label: "Llama Guard", detail: known ? health.guard_model || "not configured" : "", state: known && !!health.guard_model },
    { label: "Output guard", detail: "", state: known && !!health.output_guard },
  ];
  const overall = health === null ? null : items.some((i) => i.state === false) ? false : items.some((i) => i.state === "partial") ? "partial" : true;
  const label = health === null ? "Checking…" : overall === true ? "All systems ready" : overall === "partial" ? "Running, limited" : "Needs attention";

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 whitespace-nowrap rounded-full border border-slate-border px-3 py-1.5 text-xs font-medium text-clinical-muted transition hover:text-clinical-ink"
        aria-expanded={open}
      >
        <Dot state={overall} />
        <span className="hidden sm:inline">{label}</span>
      </button>
      {open && (
        <div className="absolute right-0 z-30 mt-2 w-72 rounded-xl border border-slate-border bg-surface p-3 shadow-elevated">
          <div className="mb-2 text-xs font-semibold text-clinical-ink">System status</div>
          <ul className="space-y-2">
            {items.map((i) => (
              <li key={i.label} className="flex items-start gap-2 text-xs">
                <span className="mt-1"><Dot state={health === null ? null : i.state} /></span>
                <span>
                  <span className="font-medium text-clinical-ink">{i.label}</span>
                  {i.detail && <span className="block text-clinical-muted">{i.detail}</span>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function UserMenu({ username, role, onSignOut }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false));
  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-full py-1 pl-1 pr-2 transition hover:bg-slate-bg"
        aria-expanded={open}
        aria-label="Account menu"
      >
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-clinical-teal text-sm font-semibold uppercase text-white">
          {(username || "?").slice(0, 1)}
        </span>
        <span className="hidden text-left leading-tight md:block">
          <span className="block max-w-[140px] truncate text-sm font-medium text-clinical-ink">{username}</span>
          <span className="block text-[11px] text-clinical-muted">{ROLE_LABEL[role] || "Patient"}</span>
        </span>
        <ChevronDown size={14} className="text-clinical-muted" />
      </button>
      {open && (
        <div className="absolute right-0 z-30 mt-2 w-52 rounded-xl border border-slate-border bg-surface p-1.5 shadow-elevated">
          <div className="px-2.5 py-2 md:hidden">
            <div className="truncate text-sm font-medium text-clinical-ink">{username}</div>
            <div className="text-[11px] text-clinical-muted">{ROLE_LABEL[role] || "Patient"}</div>
          </div>
          <button
            type="button"
            onClick={onSignOut}
            className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-clinical-ink hover:bg-slate-bg"
          >
            <LogOut size={15} className="text-clinical-muted" /> Sign out
          </button>
        </div>
      )}
    </div>
  );
}

function Tabs({ tabs, activeTab, onTabChange, className }) {
  return (
    <nav className={className} aria-label="Sections">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => onTabChange(tab.key)}
          aria-current={activeTab === tab.key ? "page" : undefined}
          className={cn(
            "whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-medium transition",
            activeTab === tab.key
              ? "bg-clinical-teal-tint text-clinical-teal-dark"
              : "text-clinical-muted hover:bg-slate-bg hover:text-clinical-ink"
          )}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}

export default function Header({ username, role, activeTab, onTabChange, onSignOut, theme, onToggleTheme }) {
  const tabs = tabsFor(role);
  return (
    <header className="sticky top-0 z-20 border-b border-slate-border bg-surface/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-2.5 md:px-6">
        <div className="flex shrink-0 items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-clinical-teal text-white">
            <ShieldPlus size={17} />
          </span>
          <span className="text-[17px] font-bold tracking-tight text-clinical-ink">MedLM</span>
        </div>

        <Tabs tabs={tabs} activeTab={activeTab} onTabChange={onTabChange} className="mx-2 hidden flex-1 items-center gap-1 lg:flex" />

        <div className="ml-auto flex items-center gap-1.5">
          {role !== "patient" && <SystemStatus />}
          <button
            type="button"
            onClick={onToggleTheme}
            className="flex h-9 w-9 items-center justify-center rounded-full text-clinical-muted transition hover:bg-slate-bg hover:text-clinical-ink"
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            title={theme === "dark" ? "Light theme" : "Dark theme"}
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <UserMenu username={username} role={role} onSignOut={onSignOut} />
        </div>
      </div>

      {/* Smaller screens: tabs on their own scrollable row. */}
      <Tabs
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={onTabChange}
        className="flex gap-1 overflow-x-auto border-t border-slate-border px-4 py-1.5 lg:hidden"
      />
    </header>
  );
}
