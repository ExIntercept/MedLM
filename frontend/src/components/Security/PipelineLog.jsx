import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { ChevronRight, Pause, Play, Radio } from "lucide-react";
import { cn } from "../../lib/cn";
import { getPipelineLog } from "../../lib/api";
import FlowMap, { requestStates } from "../Admin/FlowMap";
import { OUTCOME_LABEL, STAGES, STAGE_INFO, guardCategoryLabels, outcomeTone, reasonLabels } from "../../lib/defense";

const POLL_MS = 3000;

const TONE = {
  ok: { dot: "bg-emerald-500", chip: "border-emerald-200 bg-emerald-50 text-emerald-700" },
  warn: { dot: "bg-amber-500", chip: "border-amber-200 bg-amber-50 text-amber-700" },
  stop: { dot: "bg-red-500", chip: "border-red-200 bg-red-50 text-red-700" },
  off: { dot: "bg-slate-300", chip: "border-slate-border bg-slate-bg text-clinical-muted" },
  none: { dot: "border border-dashed border-slate-300 bg-transparent", chip: "border-dashed border-slate-border text-slate-400" },
};

const STATUS = {
  PASS: ["Answered", "ok"], FLAGGED: ["Answered", "ok"], UNGROUNDED: ["Answered", "warn"],
  BLOCKED: ["Blocked", "stop"], OUTPUT_BLOCKED: ["Answer blocked", "stop"], WITHHELD: ["Withheld", "warn"],
  INTERCEPTED: ["Intercepted", "stop"], EMERGENCY_ESCALATION: ["Emergency", "stop"],
  CLARIFY: ["Asked questions", "warn"],
};

const FILTERS = [
  { key: "all", label: "All", test: () => true },
  { key: "stopped", label: "Stopped", test: (r) => ["BLOCKED", "OUTPUT_BLOCKED", "INTERCEPTED", "WITHHELD"].includes(r.status) },
  { key: "flagged", label: "Flagged", test: (r) => r.flagged },
  { key: "sanitized", label: "Sanitized", test: (r) => stepOf(r, "sanitize")?.outcome === "modified" },
  { key: "answered", label: "Answered", test: (r) => ["PASS", "FLAGGED", "UNGROUNDED"].includes(r.status) },
];

function stepOf(rec, stage) {
  return (rec.steps || []).find((s) => s.stage === stage);
}

function StageStrip({ rec }) {
  return (
    <div className="flex items-center gap-1">
      {STAGES.filter((s) => !s.optional || stepOf(rec, s.key)).map((s) => {
        const step = stepOf(rec, s.key);
        const tone = step ? outcomeTone(step.outcome) : "none";
        return (
          <span
            key={s.key}
            className={cn("h-2.5 w-2.5 rounded-full", TONE[tone].dot)}
            title={`${s.label}: ${step ? OUTCOME_LABEL[step.outcome] || step.outcome : "not reached"}`}
          />
        );
      })}
    </div>
  );
}

const fmt = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : v);

/** Key facts for one stage, as label/value pairs. */
function stageFacts(step) {
  const d = step.detail || {};
  const facts = [];
  switch (step.stage) {
    case "sanitize":
      if (d.notes?.length) facts.push(["Changes", d.notes.join(", ")]);
      break;
    case "triage":
    case "guardrails":
      if (d.category) facts.push(["Category", d.category]);
      if (d.rule) facts.push(["Rule", d.rule]);
      break;
    case "detect":
      if (d.category) facts.push(["Category", d.category]);
      if (d.reasons?.length) facts.push(["Triggered by", reasonLabels(d.reasons).join(", ")]);
      if (d.pg_score != null) facts.push(["Layer A injection score", fmt(d.pg_score)]);
      if (d.tm_max_sim != null) facts.push(["Layer B template similarity", fmt(d.tm_max_sim)]);
      if (d.tm_n_components) facts.push(["Layer B role-play components", d.tm_n_components]);
      if (d.harm_score != null) facts.push(["Layer H harm score", fmt(d.harm_score)]);
      if (d.guard_unsafe != null)
        facts.push(["Layer G Llama Guard", d.guard_unsafe ? `unsafe · ${guardCategoryLabels(d.guard_categories || []).join(", ")}` : "safe"]);
      if (d.top_templates?.length)
        facts.push(["Closest jailbreak families", d.top_templates.map((t) => `${t.family} (${fmt(t.sim, 2)})`).join(", ")]);
      if (d.layer_ms)
        facts.push(["Per-layer ms", Object.entries(d.layer_ms).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${Math.round(v)}`).join(" · ")]);
      if (d.disabled?.length) facts.push(["Disabled layers", d.disabled.join(", ")]);
      break;
    case "clarify":
      if (d.family) facts.push(["Symptom", d.family]);
      if (d.questions?.length) facts.push(["Asked", d.questions.join(" · ")]);
      break;
    case "retrieve":
      if (d.titles?.length) facts.push(["Passages", d.titles.join(" · ")]);
      break;
    case "harden":
      if (d.extra_safety_rules) facts.push(["Mitigation", "Extra safety rules added because the request was flagged"]);
      if (d.context_fit?.length) facts.push(["Context fitting", d.context_fit.join(", ")]);
      break;
    case "generate":
      if (d.model) facts.push(["Model", d.model]);
      if (d.chars != null) facts.push(["Answer length", `${d.chars} chars`]);
      break;
    case "verify":
      if (d.score != null) facts.push(["Faithfulness", `${Math.round(d.score * 100)}%`]);
      if (d.removed?.length) facts.push(["Removed (not in the evidence)", d.removed.join(" · ")]);
      break;
    case "output_guard":
      if (d.categories?.length) facts.push(["Hazards", guardCategoryLabels(d.categories).join(", ")]);
      break;
    default:
  }
  return facts;
}

// Look-alike glyphs the sanitizer swaps in for control-token delimiters, and
// invisible characters it strips; both are highlighted so the change is visible.
const DEFANGED = /([‹›⁅⁆])/;
const INVISIBLE = /([​-‏‪-‮⁠-⁤﻿])/;

function Marked({ text, pattern, render }) {
  return text.split(pattern).map((part, i) => (i % 2 === 1 ? render(part, i) : part));
}

function Excerpt({ label, text, tone = "slate", mark }) {
  if (mark === "after") {
    text = (
      <Marked
        text={text}
        pattern={DEFANGED}
        render={(c, i) => (
          <mark key={i} className="rounded-sm bg-amber-200 px-px text-amber-900" title="Neutralised delimiter">{c}</mark>
        )}
      />
    );
  } else if (mark === "before") {
    text = (
      <Marked
        text={text}
        pattern={INVISIBLE}
        render={(c, i) => (
          <mark key={i} className="rounded-sm bg-red-200 px-0.5 text-[9px] text-red-900" title="Invisible character">
            U+{c.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")}
          </mark>
        )}
      />
    );
  }
  return <ExcerptBox label={label} text={text} tone={tone} />;
}

function ExcerptBox({ label, text, tone }) {
  return (
    <div className="mt-1.5">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-clinical-muted">{label}</div>
      <p
        className={cn(
          "mt-0.5 whitespace-pre-wrap break-words rounded border px-2 py-1 font-mono text-[11px]",
          tone === "red" ? "border-red-200 bg-red-50 text-red-700" : "border-slate-border bg-slate-bg text-clinical-ink"
        )}
      >
        {text}
      </p>
    </div>
  );
}

export function RequestDetail({ rec }) {
  return (
    <div className="space-y-4">
      <FlowMap rec={rec} className="rounded-lg border border-slate-border bg-surface p-2" />
      <Timeline rec={rec} />
    </div>
  );
}

function Timeline({ rec }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <ol className="relative space-y-3 border-l border-slate-border pl-5">
        {STAGES.filter((s) => !s.optional || stepOf(rec, s.key)).map((s) => {
          const step = stepOf(rec, s.key);
          const tone = step ? outcomeTone(step.outcome) : "none";
          const facts = step ? stageFacts(step) : [];
          const d = step?.detail || {};
          return (
            <li key={s.key} className="relative">
              <span className={cn("absolute -left-[26px] top-1 h-2.5 w-2.5 rounded-full ring-4 ring-surface", TONE[tone].dot)} />
              <div className="flex flex-wrap items-center gap-2">
                <span className={cn("text-sm font-semibold", step ? "text-clinical-ink" : "text-slate-400")}>{s.label}</span>
                <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-semibold", TONE[tone].chip)}>
                  {step ? OUTCOME_LABEL[step.outcome] || step.outcome : "not reached"}
                </span>
                {step && <span className="text-[11px] tabular-nums text-clinical-muted">{Math.round(step.ms)} ms</span>}
              </div>
              <p className="text-[11px] text-clinical-muted">{STAGE_INFO[s.key]}</p>
              {facts.length > 0 && (
                <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
                  {facts.map(([k, v]) => (
                    <Fragment key={k}>
                      <dt className="text-clinical-muted">{k}</dt>
                      <dd className="break-words text-clinical-ink">{v}</dd>
                    </Fragment>
                  ))}
                </dl>
              )}
              {d.before && <Excerpt label="Before sanitizing" text={d.before} mark="before" />}
              {d.after && <Excerpt label="After sanitizing (changes highlighted)" text={d.after} mark="after" />}
              {d.withheld_draft && <Excerpt label="Withheld draft (never shown)" text={d.withheld_draft} tone="red" />}
              {d.blocked_answer && <Excerpt label="Blocked answer (never shown)" text={d.blocked_answer} tone="red" />}
            </li>
          );
        })}
      </ol>
      <div className="space-y-2">
        <Excerpt label="User message" text={rec.message || "—"} />
        <Excerpt label="Shown to the user" text={rec.response || "—"} />
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
          <dt className="text-clinical-muted">User</dt>
          <dd className="text-clinical-ink">{rec.username} · {rec.role}</dd>
          <dt className="text-clinical-muted">Mode</dt>
          <dd className="text-clinical-ink">{rec.mode}</dd>
          <dt className="text-clinical-muted">Conversation</dt>
          <dd className="text-clinical-ink">#{rec.conversation_id}</dd>
          <dt className="text-clinical-muted">Total</dt>
          <dd className="tabular-nums text-clinical-ink">{Math.round(rec.total_ms)} ms</dd>
        </dl>
      </div>
    </div>
  );
}

export default function PipelineLog({ showMap = true }) {
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [live, setLive] = useState(true);
  const [filter, setFilter] = useState("all");
  const [open, setOpen] = useState(null);
  const [fresh, setFresh] = useState(() => new Set());
  const [stage, setStage] = useState(null);
  const [mapOpen, setMapOpen] = useState(() => {
    try {
      return localStorage.getItem("medlm.pipelineMap") !== "closed";
    } catch {
      return true;
    }
  });
  const toggleMap = () =>
    setMapOpen((v) => {
      try {
        localStorage.setItem("medlm.pipelineMap", v ? "closed" : "open");
      } catch {
        /* storage unavailable */
      }
      return !v;
    });
  const newest = useRef(null);

  const poll = useCallback(async () => {
    try {
      const { logs: incoming = [] } = await getPipelineLog(200, newest.current);
      setError(null);
      if (!incoming.length) return;
      const first = newest.current === null;
      newest.current = incoming[0].id;
      setLogs((prev) => [...incoming, ...prev].slice(0, 500));
      if (!first) {
        const ids = new Set(incoming.map((r) => r.id));
        setFresh(ids);
        setTimeout(() => setFresh(new Set()), 2500);
      }
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    poll();
    if (!live) return undefined;
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [live, poll]);

  const active = FILTERS.find((f) => f.key === filter) || FILTERS[0];
  const atStage = (r) => {
    if (!stage) return true;
    const st = requestStates(r).states[stage];
    return st && (st.tone === "stop" || st.tone === "warn");
  };
  const shown = logs.filter((r) => active.test(r) && atStage(r));

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-1.5">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              onClick={() => setFilter(f.key)}
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs font-semibold transition",
                filter === f.key ? "border-clinical-teal bg-clinical-teal text-white" : "border-slate-border text-clinical-muted hover:bg-slate-bg"
              )}
            >
              {f.label} · {logs.filter(f.test).length}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setLive((v) => !v)}
          className={cn(
            "flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-semibold",
            live ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-slate-border text-clinical-muted"
          )}
        >
          {live ? <Radio size={13} className="animate-pulse" /> : <Play size={13} />}
          {live ? "Live" : "Paused"}
          {live && <Pause size={12} className="opacity-60" />}
        </button>
      </div>

      {error && <p className="mb-2 rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700">{error}</p>}

      {showMap && logs.length > 0 && (
        <div className="mb-4 rounded-lg border border-slate-border p-2">
          <div className="flex items-center justify-between px-1 text-xs">
            <button type="button" onClick={toggleMap} className="flex items-center gap-1 font-semibold text-clinical-ink">
              <ChevronRight size={13} className={cn("transition-transform", mapOpen && "rotate-90")} />
              Where requests went · last {logs.length}
            </button>
            {stage && (
              <button type="button" onClick={() => setStage(null)} className="font-semibold text-clinical-teal-dark hover:underline">
                Clear stage filter ({stage})
              </button>
            )}
          </div>
          {mapOpen && <FlowMap logs={logs} onSelect={setStage} selected={stage} className="mt-1" />}
        </div>
      )}

      {shown.length === 0 ? (
        <p className="py-8 text-center text-sm text-clinical-muted">
          {logs.length === 0 ? "No requests yet. Send a message from any account and it appears here." : "No requests match this filter."}
        </p>
      ) : (
        <div className="max-h-[70vh] overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 z-10 bg-surface text-clinical-muted">
              <tr className="border-b border-slate-border">
                <th className="w-5 py-1.5" />
                <th className="py-1.5 pr-2 font-semibold">Time</th>
                <th className="py-1.5 pr-2 font-semibold">User</th>
                <th className="py-1.5 pr-2 font-semibold">Pipeline</th>
                <th className="py-1.5 pr-2 font-semibold">Outcome</th>
                <th className="py-1.5 pr-2 font-semibold">Message</th>
                <th className="py-1.5 text-right font-semibold">ms</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-border">
              {shown.map((r) => {
                const [label, tone] = STATUS[r.status] || [r.status, "warn"];
                const isOpen = open === r.id;
                return (
                  <Fragment key={r.id}>
                    <tr
                      onClick={() => setOpen(isOpen ? null : r.id)}
                      className={cn(
                        "cursor-pointer align-middle transition-colors hover:bg-slate-bg",
                        fresh.has(r.id) && "bg-clinical-teal-tint"
                      )}
                    >
                      <td className="py-2">
                        <ChevronRight size={13} className={cn("text-clinical-muted transition", isOpen && "rotate-90")} />
                      </td>
                      <td className="whitespace-nowrap py-2 pr-2 text-clinical-muted">
                        {new Date(r.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="whitespace-nowrap py-2 pr-2 text-clinical-ink">
                        {r.username} <span className="text-clinical-muted">· {r.role}</span>
                      </td>
                      <td className="py-2 pr-2"><StageStrip rec={r} /></td>
                      <td className="whitespace-nowrap py-2 pr-2">
                        <span className={cn("rounded-full border px-2 py-0.5 font-semibold", TONE[tone].chip)}>{label}</span>
                        {r.flagged && (
                          <span className={cn("ml-1 rounded-full border px-2 py-0.5 font-semibold", TONE.warn.chip)}>flagged</span>
                        )}
                      </td>
                      <td className="max-w-[340px] py-2 pr-2 text-clinical-ink">
                        <span className="line-clamp-1">{r.message}</span>
                      </td>
                      <td className="py-2 text-right tabular-nums text-clinical-muted">{Math.round(r.total_ms)}</td>
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={7} className="bg-slate-bg/50 px-3 py-4">
                          <RequestDetail rec={r} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-3 text-[11px] text-clinical-muted">
        {[["ok", "passed"], ["warn", "flagged / sanitized"], ["stop", "stopped here"], ["off", "layer off"], ["none", "not reached"]].map(([t, l]) => (
          <span key={t} className="flex items-center gap-1">
            <span className={cn("h-2.5 w-2.5 rounded-full", TONE[t].dot)} /> {l}
          </span>
        ))}
      </div>
    </div>
  );
}
