import { useCallback, useEffect, useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { cn } from "../../lib/cn";
import { getAdminStats, getPipelineLog } from "../../lib/api";
import { STAGES, reasonLabels } from "../../lib/defense";
import FlowMap, { shortModel } from "./FlowMap";
import { Card } from "./AdminConsole";

const STAGE_LABEL = Object.fromEntries(STAGES.map((s) => [s.key, s.label]));

function Kpi({ label, value, sub, tone }) {
  return (
    <div className="rounded-lg border border-slate-border bg-slate-bg p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-clinical-muted">{label}</div>
      <div className={cn("mt-1 text-2xl font-bold tabular-nums text-clinical-ink", tone)}>{value}</div>
      {sub && <div className="text-[11px] text-clinical-muted">{sub}</div>}
    </div>
  );
}

function Bars({ rows }) {
  const max = Math.max(1, ...rows.map((r) => r.requests));
  return (
    <div className="flex h-36 items-end gap-1.5">
      {rows.map((r) => (
        <div key={r.day} className="flex flex-1 flex-col items-center gap-1" title={`${r.day}: ${r.answered} answered, ${r.stopped} stopped`}>
          <div className="flex h-28 w-full flex-col justify-end overflow-hidden rounded bg-slate-bg">
            {r.requests > 0 && <span className="mb-0.5 text-center text-[10px] tabular-nums text-clinical-muted">{r.requests}</span>}
            <div className="bg-red-400" style={{ height: `${(100 * r.stopped) / max}%` }} />
            <div className="bg-clinical-teal" style={{ height: `${(100 * r.answered) / max}%` }} />
          </div>
          <span className="text-[10px] tabular-nums text-clinical-muted">{r.day.slice(5)}</span>
        </div>
      ))}
    </div>
  );
}

function RankList({ items, label }) {
  const entries = Object.entries(items || {});
  const max = Math.max(1, ...entries.map(([, v]) => v));
  if (!entries.length) return <p className="text-xs text-clinical-muted">Nothing yet.</p>;
  return (
    <ul className="space-y-1.5">
      {entries.map(([k, v]) => (
        <li key={k} className="text-xs">
          <div className="flex justify-between gap-2">
            <span className="truncate text-clinical-ink">{label ? label(k) : k}</span>
            <span className="tabular-nums text-clinical-muted">{v}</span>
          </div>
          <div className="mt-0.5 h-1.5 rounded bg-slate-bg">
            <div className="h-1.5 rounded bg-clinical-teal" style={{ width: `${(100 * v) / max}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function Overview({ onOpenPipeline }) {
  const [days, setDays] = useState(7);
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, l] = await Promise.all([getAdminStats(days), getPipelineLog(200)]);
      setStats(s);
      setLogs(l.logs || []);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

  if (!stats) {
    return (
      <Card>
        {error ? <p className="text-sm text-red-600">{error}</p> : <Loader2 size={18} className="animate-spin text-clinical-teal" />}
      </Card>
    );
  }
  const pct = (n) => (stats.total ? `${Math.round((100 * n) / stats.total)}%` : "—");
  const ms = (v) => (v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Overview"
        action={
          <div className="flex items-center gap-2">
            {[1, 7, 30].map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDays(d)}
                className={cn(
                  "rounded-full border px-2.5 py-1 text-xs font-semibold",
                  days === d ? "border-clinical-teal bg-clinical-teal text-white" : "border-slate-border text-clinical-muted"
                )}
              >
                {d === 1 ? "24 h" : `${d} days`}
              </button>
            ))}
            <button type="button" onClick={load} className="rounded-lg border border-slate-border p-1.5 text-clinical-muted hover:bg-slate-bg" aria-label="Refresh">
              <RefreshCw size={13} className={cn(loading && "animate-spin")} />
            </button>
          </div>
        }
      >
        <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Kpi label="Requests" value={stats.total} sub={`${stats.active_users} active users`} />
          <Kpi label="Answered" value={stats.answered} sub={pct(stats.answered)} tone="text-clinical-teal-dark" />
          <Kpi label="Stopped" value={stats.stopped} sub={pct(stats.stopped)} tone="text-red-600" />
          <Kpi label="Flagged" value={stats.flagged} sub="answered with extra rules" tone="text-amber-600" />
          <Kpi label="Sanitized" value={stats.sanitized} sub="input cleaned" />
          <Kpi label="Response time" value={ms(stats.latency_ms.median)} sub={`p95 ${ms(stats.latency_ms.p95)}`} />
        </div>
      </Card>

      <Card
        title="Where requests went"
        action={
          <button type="button" onClick={onOpenPipeline} className="text-xs font-semibold text-clinical-teal-dark hover:underline">
            Open live pipeline →
          </button>
        }
      >
        <p className="mb-2 text-xs text-clinical-muted">
          Last {logs.length} requests. Line width shows traffic; a red stage stopped at least one request.
        </p>
        {logs.length ? <FlowMap logs={logs} /> : <p className="text-sm text-clinical-muted">No requests yet.</p>}
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Requests per day">
          <Bars rows={stats.per_day} />
          <div className="mt-2 flex gap-3 text-[11px] text-clinical-muted">
            <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-clinical-teal" /> answered</span>
            <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-red-400" /> stopped</span>
          </div>
        </Card>
        <Card title="Stopped at stage">
          <RankList items={stats.stopped_by_stage} label={(k) => STAGE_LABEL[k] || k} />
        </Card>
        <Card title="Why attacks were caught">
          <RankList items={stats.top_reasons} label={(k) => `${reasonLabels([k])[0]} · ${k.endsWith("_block") ? "block" : k.endsWith("_flag") ? "flag" : "note"}`} />
          {Object.keys(stats.models || {}).length > 0 && (
            <>
              <h3 className="mb-2 mt-4 text-xs font-semibold uppercase tracking-wide text-clinical-muted">Answers by model</h3>
              <RankList items={stats.models} label={shortModel} />
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
