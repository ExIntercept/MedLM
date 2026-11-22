import { useEffect, useState } from "react";
import { ChevronDown, Loader2, ShieldCheck } from "lucide-react";
import { cn } from "../lib/cn";
import { Markdown } from "../lib/markdown";
import { getCrossModelResults } from "../lib/api";
import { STOP_STAGES as STAGES } from "../lib/defense";

/** Says which bar is which, for every paired-bar column. */
function BarLegend({ judge }) {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg bg-slate-bg px-3 py-2 text-[11px] text-clinical-muted">
      <span className="font-semibold text-clinical-ink">Reading the bars:</span>
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-5 rounded bg-red-400" /> top bar = the model alone (<b className="font-semibold">undefended</b>)
      </span>
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-5 rounded bg-clinical-teal" /> bottom bar = the same model through the MedLM gateway (<b className="font-semibold">defended</b>)
      </span>
      <span>
        Attack success = share of attack prompts whose final answer the judge rated harmful; judge: {judge}.
        In MedQA accuracy the top bar is grey.
      </span>
    </div>
  );
}

function pct(r) {
  return r && r.n ? `${r.pct.toFixed(1)}%` : "n/a";
}

function ci(r) {
  return r && r.n ? `[${r.lo.toFixed(0)}–${r.hi.toFixed(0)}]` : "";
}

/** Two bars on one 0–100 scale: undefended above, defended below. */
function PairBars({ before, after, beforeTone = "bg-red-400", afterTone = "bg-clinical-teal" }) {
  const w = (r) => `${r && r.n ? Math.max(r.pct, 0.6) : 0}%`;
  return (
    <div className="w-40 space-y-1">
      {[[before, beforeTone], [after, afterTone]].map(([r, tone], i) => (
        <div key={i} className="flex items-center gap-2" title={`${i === 0 ? "Model alone (undefended)" : "Through the MedLM gateway (defended)"}: ${pct(r)}`}>
          <div className="h-2 flex-1 rounded bg-slate-100">
            <div className={cn("h-2 rounded", tone)} style={{ width: w(r) }} />
          </div>
          <span className="w-12 text-right tabular-nums">{pct(r)}</span>
        </div>
      ))}
    </div>
  );
}

function pValue(p) {
  if (p === undefined || p === null) return "—";
  return p < 0.001 ? `${p.toExponential(1)}` : p.toFixed(3);
}

export default function CrossModelResults() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [showReport, setShowReport] = useState(false);

  useEffect(() => {
    getCrossModelResults()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="rounded-xl border border-slate-border bg-surface p-4 text-sm text-clinical-muted shadow-card">
        <p className="font-semibold text-clinical-ink">Cross-model defense results</p>
        <p className="mt-1">{error}</p>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-slate-border bg-surface p-4 text-sm text-clinical-muted shadow-card">
        <Loader2 size={16} className="animate-spin text-clinical-teal" /> Loading cross-model results…
      </div>
    );
  }

  const main = data.results.main || {};
  const models = Object.keys(main);
  const stages = data.results.stages || {};
  const ablations = data.results.ablations || {};
  const avg = (cond, key) => {
    const rs = models.map((m) => main[m][cond][key]).filter((r) => r && r.n);
    return rs.length ? rs.reduce((a, r) => a + r.pct, 0) / rs.length : null;
  };
  const kpis = [
    { label: "Attack success · model alone", value: avg("undefended", "asr_j1"), tone: "text-red-600" },
    { label: "Attack success · through gateway", value: avg("defended", "asr_j1"), tone: "text-clinical-teal-dark" },
    { label: "Medical questions refused · through gateway", value: avg("defended", "med_refused"), tone: "text-clinical-ink" },
  ];

  return (
    <div className="rounded-xl border border-slate-border bg-surface p-4 shadow-card">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="flex items-center gap-2 text-[15px] font-semibold text-clinical-ink">
            <ShieldCheck size={16} className="text-clinical-teal" />
            Cross-model defense results
          </h2>
          <p className="mt-0.5 text-xs text-clinical-muted">
            The same gateway in front of {models.length} medical models · each model called directly (undefended) vs
            through the gateway (defended) · averages over models, judge Llama Guard 3 8B
          </p>
        </div>
        {data.markdown && (
          <button
            type="button"
            onClick={() => setShowReport((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg border border-slate-border px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-bg"
          >
            Full tables
            <ChevronDown size={13} className={cn("transition", showReport && "rotate-180")} />
          </button>
        )}
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        {kpis.map((k) => (
          <div key={k.label} className="rounded-lg border border-slate-border bg-slate-bg p-3">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-clinical-muted">{k.label}</div>
            <div className={cn("mt-1 text-2xl font-bold tabular-nums", k.tone)}>
              {k.value === null ? "n/a" : `${k.value.toFixed(1)}%`}
            </div>
          </div>
        ))}
      </div>

      <BarLegend judge="Llama Guard 3 8B (judge 1); the 2nd judge is Qwen3-8B, a different model family" />

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-clinical-muted">
            <tr className="border-b border-slate-border">
              <th className="py-2 pr-3 font-semibold">Model</th>
              <th className="py-2 pr-3 font-semibold">Attack success<br /><span className="font-normal">top: alone · bottom: gateway</span></th>
              <th className="py-2 pr-3 font-semibold">2nd judge<br /><span className="font-normal">alone / gateway</span></th>
              <th className="py-2 pr-3 font-semibold">McNemar p</th>
              <th className="py-2 pr-3 font-semibold">MedQA accuracy<br /><span className="font-normal">top: alone · bottom: gateway</span></th>
              <th className="py-2 pr-3 font-semibold">Medical refused<br /><span className="font-normal">defended</span></th>
              <th className="py-2 font-semibold">General refused<br /><span className="font-normal">defended</span></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-border">
            {models.map((m) => {
              const u = main[m].undefended;
              const d = main[m].defended;
              return (
                <tr key={m} className="align-middle">
                  <td className="py-2 pr-3 font-semibold text-clinical-ink">{m}</td>
                  <td className="py-2 pr-3"><PairBars before={u.asr_j1} after={d.asr_j1} /></td>
                  <td className="py-2 pr-3 tabular-nums">{pct(u.asr_j2)} / {pct(d.asr_j2)}</td>
                  <td className="py-2 pr-3 tabular-nums">{pValue(main[m].mcnemar_p)}</td>
                  <td className="py-2 pr-3">
                    <PairBars before={u.medqa_acc} after={d.medqa_acc} beforeTone="bg-slate-400" afterTone="bg-clinical-teal" />
                  </td>
                  <td className="py-2 pr-3 tabular-nums">{pct(d.med_refused)} <span className="text-clinical-muted">{ci(d.med_refused)}</span></td>
                  <td className="py-2 tabular-nums">{pct(d.gen_refused)} <span className="text-clinical-muted">{ci(d.gen_refused)}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {Object.keys(stages).length > 0 && (
        <div className="mt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-clinical-muted">
            Where defended attacks were stopped
          </h3>
          <p className="mt-0.5 text-[11px] text-clinical-muted">
            One bar per model: all its attack prompts sent through the gateway, coloured by the numbered stage that
            stopped each one. “Answered” means the prompt got a reply; whether that reply was harmful is what the
            judge decides above.
          </p>
          <div className="mt-2 space-y-1.5">
            {Object.entries(stages).map(([m, cnt]) => {
              const total = Object.values(cnt).reduce((a, b) => a + b, 0) || 1;
              return (
                <div key={m} className="flex items-center gap-3 text-xs">
                  <span className="w-36 shrink-0 font-semibold text-clinical-ink">{m}</span>
                  <div className="flex h-3 flex-1 overflow-hidden rounded bg-slate-100">
                    {STAGES.map((s) =>
                      cnt[s.key] ? (
                        <div
                          key={s.key}
                          className={s.color}
                          style={{ width: `${(100 * cnt[s.key]) / total}%` }}
                          title={`${s.label}: ${cnt[s.key]}`}
                        />
                      ) : null
                    )}
                  </div>
                  <span className="w-10 text-right tabular-nums text-clinical-muted">{total}</span>
                </div>
              );
            })}
          </div>
          <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-clinical-muted">
            {STAGES.map((s) => (
              <span key={s.key} className="flex items-center gap-1">
                <span className={cn("h-2.5 w-2.5 rounded-sm", s.color)} /> {s.label}
              </span>
            ))}
          </div>
        </div>
      )}

      {Object.keys(ablations).length > 0 && (
        <div className="mt-5 overflow-x-auto">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-clinical-muted">Ablations (defended)</h3>
          <table className="mt-2 w-full text-left text-xs">
            <thead className="text-clinical-muted">
              <tr className="border-b border-slate-border">
                <th className="py-1.5 pr-3 font-semibold">Model</th>
                <th className="py-1.5 pr-3 font-semibold">Configuration</th>
                <th className="py-1.5 pr-3 font-semibold">Attack success</th>
                <th className="py-1.5 pr-3 font-semibold">2nd judge</th>
                <th className="py-1.5 pr-3 font-semibold">Medical refused</th>
                <th className="py-1.5 pr-3 font-semibold">MedQA accuracy</th>
                <th className="py-1.5 font-semibold">General refused</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-border">
              {Object.entries(ablations).flatMap(([m, configs]) =>
                Object.entries(configs).map(([name, s]) => (
                  <tr key={`${m}-${name}`} className={cn(name === "full gateway" && "font-semibold")}>
                    <td className="py-1.5 pr-3">{m}</td>
                    <td className="py-1.5 pr-3">{name}</td>
                    <td className="py-1.5 pr-3 tabular-nums">{pct(s.asr_j1)}</td>
                    <td className="py-1.5 pr-3 tabular-nums">{pct(s.asr_j2)}</td>
                    <td className="py-1.5 pr-3 tabular-nums">{pct(s.med_refused)}</td>
                    <td className="py-1.5 pr-3 tabular-nums">{pct(s.medqa_acc)}</td>
                    <td className="py-1.5 tabular-nums">{pct(s.gen_refused)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {showReport && data.markdown && (
        <div className="mt-4 max-h-[60vh] overflow-auto rounded-lg border border-slate-border p-4">
          <Markdown text={data.markdown} className="text-xs" />
        </div>
      )}
    </div>
  );
}
