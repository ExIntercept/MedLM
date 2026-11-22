import { useCallback, useEffect, useState } from "react";
import { Activity, ListChecks, Loader2, RefreshCw, ShieldCheck } from "lucide-react";
import { cn } from "../../lib/cn";
import { getDefenseConfig, getDefenseLog, updateDefenseToggles } from "../../lib/api";
import { TOGGLES, reasonLabels } from "../../lib/defense";
import PipelineLog from "./PipelineLog";

const ACTION_TONE = {
  BLOCK: "border-red-200 bg-red-50 text-red-700",
  FLAG: "border-amber-200 bg-amber-50 text-amber-700",
  ALLOW: "border-emerald-200 bg-emerald-50 text-emerald-700",
};

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
    <section className="rounded-xl border border-slate-border bg-surface p-4 shadow-card">
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
    <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
      <Panel
        title="Defense layers"
        action={<span className="text-xs text-clinical-muted">Changes apply to the next message</span>}
      >
        {error && <p className="mb-2 rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700">{error}</p>}
        <ul className="divide-y divide-slate-border">
          {TOGGLES.map((t) => (
            <li key={t.key} className="flex items-center justify-between gap-3 py-2.5">
              <div>
                <div className="text-sm font-semibold text-clinical-ink">{t.label}</div>
                <div className="text-xs text-clinical-muted">{t.hint}</div>
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
        <p className="mb-2 text-[11px] text-clinical-muted">
          One row per message that reached the attack detectors, with each detector&apos;s score.
        </p>
        <div className="mb-3 flex flex-wrap gap-2 text-xs">
          {["ALL", "BLOCK", "FLAG", "ALLOW"].map((a) => (
            <button
              key={a}
              type="button"
              onClick={() => setFilter(a)}
              className={cn(
                "rounded-full border px-2.5 py-1 font-semibold transition",
                a === "ALL" ? "border-slate-border text-clinical-muted" : ACTION_TONE[a],
                filter === a && "ring-2 ring-clinical-teal ring-offset-1 ring-offset-surface"
              )}
            >
              {a === "ALL" ? `All · ${logs.length}` : `${a} · ${counts[a] || 0}`}
            </button>
          ))}
        </div>
        {logs.length === 0 ? (
          <p className="flex items-center gap-2 text-sm text-clinical-muted">
            <ShieldCheck size={15} /> No decisions logged yet.
          </p>
        ) : (
          <div className="max-h-[60vh] overflow-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-surface text-clinical-muted">
                <tr>
                  <th className="py-1.5 pr-2 font-semibold">Time</th>
                  <th className="py-1.5 pr-2 font-semibold">Decision</th>
                  <th className="py-1.5 pr-2 font-semibold">Category</th>
                  <th className="py-1.5 pr-2 font-semibold">Scores</th>
                  <th className="py-1.5 pr-2 font-semibold">Why</th>
                  <th className="py-1.5 pr-2 font-semibold">Message</th>
                  <th className="py-1.5 font-semibold">ms</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-border">
                {logs.filter((r) => filter === "ALL" || r.action === filter).map((r, i) => (
                  <tr key={i} className="align-top">
                    <td className="whitespace-nowrap py-1.5 pr-2 text-clinical-muted">
                      {r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}
                    </td>
                    <td className="py-1.5 pr-2">
                      <span className={cn("rounded-full border px-2 py-0.5 font-semibold", ACTION_TONE[r.action])}>
                        {r.action}
                      </span>
                    </td>
                    <td className="py-1.5 pr-2 font-semibold text-clinical-muted">{r.category || "—"}</td>
                    <td className="whitespace-nowrap py-1.5 pr-2 font-mono text-[11px] text-clinical-ink">
                      {r.pg_score != null && <div>A {Number(r.pg_score).toFixed(2)}</div>}
                      <div title={(r.top_templates || []).map((t) => t.family).join(", ")}>
                        B {Number(r.tm_max_sim ?? 0).toFixed(2)}
                        {r.top_templates?.[0]?.family && <span className="text-clinical-muted"> {r.top_templates[0].family}</span>}
                      </div>
                      {r.harm_score != null && <div>H {Number(r.harm_score).toFixed(2)}</div>}
                    </td>
                    <td className="py-1.5 pr-2 text-clinical-ink">
                      {reasonLabels(r.reasons).join(", ") || "—"}
                      {r.sanitize_notes?.length > 0 && (
                        <div className="text-[11px] text-amber-700">Sanitized: {r.sanitize_notes.join(", ")}</div>
                      )}
                    </td>
                    <td className="max-w-[320px] py-1.5 pr-2 text-clinical-ink">
                      <span className="line-clamp-2">{r.message_excerpt}</span>
                    </td>
                    <td className="py-1.5 text-clinical-muted">{Math.round(r.latency_ms?.total ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        </>
        )}
      </Panel>
    </div>
  );
}
