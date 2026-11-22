import { useCallback, useEffect, useState } from "react";
import { Activity, ChevronDown, ListChecks, Loader2, RefreshCw, ShieldCheck } from "lucide-react";
import { relativeTime } from "../Sidebar/ConsultationHistory";
import { cn } from "../../lib/cn";
import { getDefenseConfig, getDefenseLog, updateDefenseToggles } from "../../lib/api";
import { TOGGLES, reasonLabels } from "../../lib/defense";
import PipelineLog from "./PipelineLog";

const ACTION_DOT = { BLOCK: "bg-red-500", FLAG: "bg-amber-500", ALLOW: "bg-emerald-500" };
const ACTION_TEXT = { BLOCK: "text-red-700", FLAG: "text-amber-700", ALLOW: "text-emerald-700" };
const ACTION_LABEL = { BLOCK: "Blocked", FLAG: "Flagged", ALLOW: "Allowed" };

function titleCase(s) {
  return s ? s.charAt(0) + s.slice(1).toLowerCase().replace(/_/g, " ") : "";
}

/** A tiny labelled bar: one detector's score from 0 to 1. */
function ScoreBar({ label, value }) {
  if (value == null) return null;
  const v = Math.max(0, Math.min(1, Number(value)));
  const tone = v >= 0.8 ? "bg-red-500" : v >= 0.5 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="flex items-center gap-1.5" title={`${label} ${v.toFixed(2)}`}>
      <span className="w-2.5 text-[10px] font-semibold text-clinical-muted">{label}</span>
      <span className="h-1.5 w-12 overflow-hidden rounded-full bg-slate-bg">
        <span className={cn("block h-full rounded-full", tone)} style={{ width: `${v * 100}%` }} />
      </span>
      <span className="w-7 font-mono text-[10px] text-clinical-muted">{v.toFixed(2)}</span>
    </div>
  );
}

function DetectorRow({ r }) {
  const [open, setOpen] = useState(false);
  const why = reasonLabels(r.reasons);
  const when = r.timestamp ? new Date(r.timestamp) : null;
  return (
    <div className={cn("transition", open ? "bg-slate-bg/60" : "hover:bg-slate-bg/60")}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-3 py-2.5 text-left"
        aria-expanded={open}
      >
        <span className={cn("h-2 w-2 shrink-0 rounded-full", ACTION_DOT[r.action])} />
        <span className={cn("w-16 shrink-0 text-xs font-semibold", ACTION_TEXT[r.action])}>
          {ACTION_LABEL[r.action] || r.action}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm text-clinical-ink">{r.message_excerpt || "—"}</span>
          <span className="block truncate text-[11px] text-clinical-muted">
            {[titleCase(r.category), why[0], why.length > 1 ? `+${why.length - 1} more` : null]
              .filter(Boolean)
              .join(" · ") || "No detector fired"}
          </span>
        </span>
        <span className="hidden shrink-0 space-y-0.5 lg:block">
          <ScoreBar label="A" value={r.pg_score} />
          <ScoreBar label="B" value={r.tm_max_sim ?? 0} />
        </span>
        <span
          className="w-16 shrink-0 text-right text-[11px] text-clinical-muted"
          title={when ? when.toLocaleString() : ""}
        >
          {when ? relativeTime(r.timestamp) : "—"}
        </span>
        <ChevronDown size={14} className={cn("shrink-0 text-clinical-muted transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="grid gap-4 px-3 pb-3 pl-8 text-xs sm:grid-cols-2">
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-clinical-muted">Message</div>
            <p className="whitespace-pre-wrap break-words text-clinical-ink">{r.message_excerpt}</p>
            {r.sanitize_notes?.length > 0 && (
              <p className="text-amber-700">Cleaned before checking: {r.sanitize_notes.join(", ")}</p>
            )}
          </div>
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-clinical-muted">Detectors</div>
            <ScoreBar label="A" value={r.pg_score} />
            <ScoreBar label="B" value={r.tm_max_sim ?? 0} />
            <ScoreBar label="H" value={r.harm_score} />
            {r.top_templates?.length > 0 && (
              <p className="text-clinical-muted">
                Closest jailbreak templates: {[...new Set(r.top_templates.map((t) => t.family).filter(Boolean))].join(", ")}
              </p>
            )}
            {why.length > 0 && <p className="text-clinical-ink">Why: {why.join(", ")}</p>}
            <p className="text-clinical-muted">
              {when ? when.toLocaleString() : ""} · {Math.round(r.latency_ms?.total ?? 0)} ms
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function Switch({ checked, onChange, disabled }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-5 w-9 shrink-0 rounded-full transition disabled:opacity-50",
        checked ? "bg-clinical-teal" : "bg-slate-300"
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition",
          checked ? "left-[18px]" : "left-0.5"
        )}
      />
    </button>
  );
}

function Panel({ title, action, children }) {
  return (
    <section className="min-w-0 rounded-xl border border-slate-border bg-surface p-4 shadow-card">
      <div className="mb-3 flex items-center justify-between gap-2">
        {typeof title === "string" ? <h2 className="text-[15px] font-semibold text-clinical-ink">{title}</h2> : title}
        {action}
      </div>
      {children}
    </section>
  );
}

export default function SecurityConsole({ hidePipeline = false }) {
  const [config, setConfig] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(null);
  const [view, setView] = useState(hidePipeline ? "detector" : "pipeline");
  const [filter, setFilter] = useState("ALL");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [cfg, log] = await Promise.all([getDefenseConfig(), getDefenseLog(100)]);
      setConfig(cfg);
      setLogs(log.logs || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const toggle = async (key, value) => {
    setSaving(key);
    try {
      const res = await updateDefenseToggles({ [key]: value });
      setConfig((prev) => ({ ...prev, toggles: res.toggles }));
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(null);
    }
  };

  if (loading && !config) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center rounded-xl border border-slate-border bg-surface shadow-card">
        <Loader2 size={22} className="animate-spin text-clinical-teal" />
      </div>
    );
  }

  if (error && !config) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-800 shadow-card">
        <p className="font-semibold">Security console unavailable</p>
        <p className="mt-1">{error}</p>
      </div>
    );
  }

  const toggles = config?.toggles || {};
  const counts = logs.reduce((acc, r) => ({ ...acc, [r.action]: (acc[r.action] || 0) + 1 }), {});

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
      <Panel
        title="Gateway stages"
        action={<span className="text-xs text-clinical-muted">Changes apply to the next message</span>}
      >
        {error && <p className="mb-2 rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700">{error}</p>}
        <p className="mb-1 text-[11px] text-clinical-muted">
          Numbers are the stage order (1–9). Stages 5, 7 and 8 (retrieval, generation, grounding check) have no switch.
          The attack-detector master switch also turns the output guard off.
        </p>
        <ul className="divide-y divide-slate-border">
          {TOGGLES.map((t) => (
            <li key={t.key} className={cn("flex items-center justify-between gap-3 py-2.5", t.sub && "pl-7")}>
              <div className="flex min-w-0 gap-2">
                {!t.sub && (
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-clinical-teal text-[10px] font-bold text-white">
                    {t.stage}
                  </span>
                )}
                <div className="min-w-0">
                  <div className={cn("font-semibold text-clinical-ink", t.sub ? "text-[13px]" : "text-sm")}>{t.label}</div>
                  <div className="text-xs text-clinical-muted">{t.hint}</div>
                </div>
              </div>
              <Switch
                checked={toggles[t.key] !== false}
                disabled={saving === t.key}
                onChange={(v) => toggle(t.key, v)}
              />
            </li>
          ))}
        </ul>
        <p className="mt-3 text-xs text-clinical-muted">
          Injection classifier (Layer A): <span className="font-semibold">{config?.prompt_guard_status}</span>
        </p>
        {config?.thresholds && (
          <div className="mt-4 border-t border-slate-border pt-3">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-clinical-muted">Detection thresholds</h3>
            <p className="mt-0.5 text-[11px] text-clinical-muted">
              Calibrated values used for the reported results; read-only here so a demo cannot change them.
            </p>
            <dl className="mt-2 grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-xs">
              {Object.entries(config.thresholds).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="font-mono text-clinical-muted">{k}</dt>
                  <dd className="text-right tabular-nums text-clinical-ink">{typeof v === "number" ? v : String(v)}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </Panel>

      <Panel
        title={
          <div className="flex overflow-hidden rounded-lg border border-slate-border text-xs font-semibold">
            {[
              { key: "pipeline", label: "Live pipeline", Icon: Activity },
              { key: "detector", label: "Detector log", Icon: ListChecks },
            ].filter((t) => !(hidePipeline && t.key === "pipeline")).map(({ key, label, Icon }) => (
              <button
                key={key}
                type="button"
                onClick={() => setView(key)}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 transition",
                  view === key ? "bg-clinical-teal text-white" : "bg-surface text-clinical-muted hover:bg-slate-bg"
                )}
              >
                <Icon size={13} /> {label}
              </button>
            ))}
          </div>
        }
        action={
          view === "detector" && <button
            type="button"
            onClick={load}
            className="flex items-center gap-1.5 rounded-lg border border-slate-border px-2.5 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-bg"
          >
            <RefreshCw size={13} className={cn(loading && "animate-spin")} />
            Refresh
          </button>
        }
      >
        {view === "pipeline" ? (
          <PipelineLog />
        ) : (
        <>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="inline-flex rounded-lg bg-slate-bg p-0.5 text-xs">
            {["ALL", "BLOCK", "FLAG", "ALLOW"].map((a) => (
              <button
                key={a}
                type="button"
                onClick={() => setFilter(a)}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-2.5 py-1 font-medium transition",
                  filter === a ? "bg-surface text-clinical-ink shadow-card" : "text-clinical-muted hover:text-clinical-ink"
                )}
              >
                {a !== "ALL" && <span className={cn("h-1.5 w-1.5 rounded-full", ACTION_DOT[a])} />}
                {a === "ALL" ? "All" : ACTION_LABEL[a]}
                <span className="text-clinical-muted">{a === "ALL" ? logs.length : counts[a] || 0}</span>
              </button>
            ))}
          </div>
          <p className="text-[11px] text-clinical-muted">
            Scores: <b className="font-medium">A</b> injection · <b className="font-medium">B</b> jailbreak template ·{" "}
            <b className="font-medium">H</b> harmful intent
          </p>
        </div>
        {logs.length === 0 ? (
          <p className="flex items-center gap-2 text-sm text-clinical-muted">
            <ShieldCheck size={15} /> No decisions logged yet.
          </p>
        ) : (
          <div className="max-h-[65vh] divide-y divide-slate-border overflow-auto rounded-lg border border-slate-border">
            {logs
              .filter((r) => filter === "ALL" || r.action === filter)
              .map((r, i) => (
                <DetectorRow key={r.timestamp ? `${r.timestamp}-${i}` : i} r={r} />
              ))}
          </div>
        )}
        </>
        )}
      </Panel>
    </div>
  );
}
