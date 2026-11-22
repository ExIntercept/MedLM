import { cn } from "../../lib/cn";
import { OUTCOME_LABEL, outcomeTone } from "../../lib/defense";

// Left-to-right map of the gateway. The four detectors fan out of "Detect"
// and back into "Retrieve"; evidence hangs off "Retrieve". A request that is
// stopped early jumps from the stopping stage straight to the response.
const W = 104;
const H = 40;
const MAIN_Y = 150;

const NODES = [
  { id: "input", label: "Message", x: 8, y: MAIN_Y },
  { id: "sanitize", label: "Sanitize", x: 132, y: MAIN_Y },
  { id: "triage", label: "Triage", x: 256, y: MAIN_Y },
  { id: "guardrails", label: "Guardrails", x: 380, y: MAIN_Y },
  { id: "detect", label: "Detect", x: 504, y: MAIN_Y },
  { id: "A", label: "A · Injection", x: 636, y: 28, small: true },
  { id: "B", label: "B · Jailbreak", x: 636, y: 84, small: true },
  { id: "H", label: "H · Harm intent", x: 636, y: 216, small: true },
  { id: "G", label: "G · Llama Guard", x: 636, y: 272, small: true },
  { id: "retrieve", label: "Retrieve", x: 768, y: MAIN_Y },
  { id: "evidence", label: "Evidence", x: 768, y: 250, small: true },
  { id: "harden", label: "Harden", x: 892, y: MAIN_Y },
  { id: "generate", label: "Generate", x: 1016, y: MAIN_Y },
  { id: "verify", label: "Verify", x: 1140, y: MAIN_Y },
  { id: "output_guard", label: "Output guard", x: 1264, y: MAIN_Y },
  { id: "response", label: "Response", x: 1388, y: MAIN_Y },
];
const BY_ID = Object.fromEntries(NODES.map((n) => [n.id, n]));
const VIEW_W = 1388 + W + 8;
const VIEW_H = 330;

const EDGES = [
  ["input", "sanitize"], ["sanitize", "triage"], ["triage", "guardrails"], ["guardrails", "detect"],
  ["detect", "A"], ["detect", "B"], ["detect", "H"], ["detect", "G"],
  ["A", "retrieve"], ["B", "retrieve"], ["H", "retrieve"], ["G", "retrieve"],
  ["retrieve", "evidence"], ["retrieve", "harden"], ["harden", "generate"], ["generate", "verify"],
  ["verify", "output_guard"], ["output_guard", "response"],
];
const MAIN_ORDER = ["sanitize", "triage", "guardrails", "detect", "retrieve", "harden", "generate", "verify", "output_guard"];

// One class set for both themes; dark: variants follow the theme toggle live.
const FILL = {
  stop: "fill-red-500 stroke-red-600 dark:fill-red-500/90 dark:stroke-red-300",
  warn: "fill-amber-100 stroke-amber-500 dark:fill-amber-900/40",
  ok: "fill-emerald-50 stroke-emerald-500 dark:fill-emerald-900/30",
  off: "fill-violet-50 stroke-violet-400 dark:fill-violet-900/30",
  none: "fill-transparent stroke-slate-300 dark:stroke-slate-600",
  neutral: "fill-surface stroke-slate-400 dark:fill-transparent",
};

function edgePath(a, b) {
  const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
  if (b.id === "evidence") return `M ${a.x + W / 2} ${a.y + H} L ${b.x + W / 2} ${b.y}`;
  const dx = Math.max(30, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

/** Arc from a stopping stage over the remaining stages to the response. */
function skipPath(a, b) {
  const x1 = a.x + W / 2, y1 = a.y, x2 = b.x + W / 2, y2 = b.y;
  const top = 6;
  return `M ${x1} ${y1} C ${x1} ${top}, ${x2} ${top}, ${x2} ${y2}`;
}

const DETECTOR_OF_REASON = (r) => (r.startsWith("pg_") ? "A" : r.startsWith("tm_") ? "B" : r.startsWith("harm_") ? "H" : r.startsWith("guard_") ? "G" : null);
const LAYER_KEY = { A: "layer_prompt_guard", B: "layer_template_matcher", H: "layer_harm_classifier", G: "layer_llama_guard" };

/** Per-node state for one request. */
export function requestStates(rec) {
  const steps = Object.fromEntries((rec.steps || []).map((s) => [s.stage, s]));
  const states = { input: { tone: "neutral", sub: rec.role || "" } };
  let stoppedAt = null;
  for (const id of MAIN_ORDER) {
    const s = steps[id];
    if (!s) {
      states[id] = { tone: "none", sub: "not reached" };
      continue;
    }
    const tone = outcomeTone(s.outcome);
    const isStop = tone === "stop" || (id === "output_guard" && s.outcome === "blocked");
    if (isStop && !stoppedAt) stoppedAt = id;
    states[id] = { tone: isStop ? "stop" : tone, sub: `${OUTCOME_LABEL[s.outcome] || s.outcome} · ${Math.round(s.ms)}ms` };
  }
  // Detectors
  const det = steps.detect;
  const d = det?.detail || {};
  const fired = {};
  (d.reasons || []).forEach((r) => {
    const k = DETECTOR_OF_REASON(r);
    if (k) fired[k] = r.endsWith("_block") ? "stop" : fired[k] === "stop" ? "stop" : "warn";
  });
  const scores = {
    A: d.pg_score != null ? d.pg_score.toFixed(2) : null,
    B: d.tm_max_sim != null ? d.tm_max_sim.toFixed(2) : null,
    H: d.harm_score != null ? d.harm_score.toFixed(2) : null,
    G: d.guard_unsafe == null ? null : d.guard_unsafe ? "unsafe" : "safe",
  };
  for (const k of ["A", "B", "H", "G"]) {
    if (!det || det.outcome === "off") states[k] = { tone: det ? "off" : "none", sub: det ? "off" : "not reached" };
    else if ((d.disabled || []).includes(LAYER_KEY[k])) states[k] = { tone: "off", sub: "off" };
    else if (fired[k]) states[k] = { tone: det.outcome === "block" ? fired[k] : "warn", sub: scores[k] ?? "fired" };
    else states[k] = { tone: scores[k] == null ? "none" : "ok", sub: scores[k] ?? "no score" };
  }
  const r = steps.retrieve;
  states.evidence = r && r.outcome !== "off"
    ? { tone: "neutral", sub: r.outcome }
    : { tone: "none", sub: r ? "off" : "not reached" };
  if (steps.generate?.detail?.model) states.generate.sub = shortModel(steps.generate.detail.model);
  states.response = stoppedAt
    ? { tone: "warn", sub: "safe message shown" }
    : { tone: "ok", sub: "answer shown" };
  return { states, stoppedAt };
}

/** Per-node counts across many requests. */
export function aggregateStates(logs) {
  const counts = {};
  const bump = (id, tone) => {
    counts[id] = counts[id] || { ok: 0, warn: 0, stop: 0, off: 0, none: 0, neutral: 0 };
    counts[id][tone] += 1;
  };
  logs.forEach((rec) => {
    const { states } = requestStates(rec);
    Object.entries(states).forEach(([id, s]) => bump(id, s.tone));
  });
  const states = {};
  const traffic = {};
  NODES.forEach(({ id }) => {
    const c = counts[id] || { ok: 0, warn: 0, stop: 0, off: 0, none: 0, neutral: 0 };
    const reached = c.ok + c.warn + c.stop + c.neutral;
    traffic[id] = reached;
    const tone = c.stop ? "stop" : c.warn ? "warn" : reached ? (id === "input" || id === "evidence" ? "neutral" : "ok") : c.off ? "off" : "none";
    const parts = [];
    if (id === "response") parts.push(`${logs.length - c.warn} answered`, c.warn ? `${c.warn} refused` : null);
    else if (!reached) parts.push(c.off ? "off" : "not run");
    else {
      parts.push(`${reached} in`);
      if (c.stop) parts.push(`${c.stop} stopped`);
      if (c.warn) parts.push(`${c.warn} ${id === "sanitize" ? "cleaned" : "flagged"}`);
    }
    states[id] = { tone, sub: parts.filter(Boolean).join(" · "), count: c };
  });
  return { states, traffic };
}

export function shortModel(name = "") {
  const m = name.split(":").slice(-2).join(":").split("/").pop() || name;
  return m.replace(/-GGUF.*$/i, "").replace(/\.gguf$/i, "");
}

/**
 * mode "request": one request (rec).  mode "aggregate": many requests (logs);
 * edge width follows traffic and clicking a node calls onSelect(stageId).
 */
export default function FlowMap({ rec, logs, onSelect, selected, className }) {
  const aggregate = !!logs;
  const { states, stoppedAt } = aggregate ? { ...aggregateStates(logs), stoppedAt: null } : requestStates(rec);
  const traffic = aggregate ? aggregateStates(logs).traffic : null;
  const maxTraffic = aggregate ? Math.max(1, ...Object.values(traffic)) : 1;
  const fills = FILL;

  const edgeState = (a, b) => {
    if (aggregate) {
      const t = Math.min(traffic[a] || 0, traffic[b] || 0);
      return { width: t ? 1 + (4 * t) / maxTraffic : 1, dim: !t };
    }
    const reached = states[a]?.tone !== "none" && states[b]?.tone !== "none" && states[b]?.tone !== "off";
    return { width: reached ? 1.6 : 1, dim: !reached };
  };

  return (
    <div className={cn("overflow-x-auto", className)}>
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="min-w-[900px] w-full" role="img" aria-label="Pipeline flow map">
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="fill-red-400" />
          </marker>
        </defs>
        {EDGES.map(([a, b]) => {
          const e = edgeState(a, b);
          return (
            <path
              key={`${a}-${b}`}
              d={edgePath(BY_ID[a], BY_ID[b])}
              fill="none"
              strokeWidth={e.width}
              className={e.dim ? "stroke-slate-300/60" : "stroke-slate-400"}
              strokeDasharray={e.dim ? "3 4" : undefined}
            />
          );
        })}
        {stoppedAt && (
          <path
            d={skipPath(BY_ID[stoppedAt === "output_guard" ? "output_guard" : stoppedAt], BY_ID.response)}
            fill="none"
            strokeWidth={2}
            strokeDasharray="6 5"
            className="stroke-red-400"
            markerEnd="url(#arrow)"
          />
        )}
        {NODES.map((n) => {
          const s = states[n.id] || { tone: "none", sub: "" };
          const isSel = selected === n.id;
          const clickable = aggregate && onSelect && !["input", "evidence", "response"].includes(n.id);
          const label = n.id === "generate" && !aggregate && s.sub && s.tone !== "none" ? "Generate" : n.label;
          return (
            <g
              key={n.id}
              transform={`translate(${n.x} ${n.y})`}
              onClick={clickable ? () => onSelect(isSel ? null : n.id) : undefined}
              className={clickable ? "cursor-pointer" : undefined}
            >
              <title>{`${n.label}: ${s.sub || ""}`}</title>
              <rect
                width={W}
                height={H}
                rx={9}
                strokeWidth={isSel ? 3 : 1.5}
                strokeDasharray={s.tone === "off" ? "4 3" : undefined}
                className={cn(fills[s.tone] || fills.none, isSel && "stroke-clinical-teal")}
              />
              <text
                x={W / 2}
                y={s.sub ? 16 : 24}
                textAnchor="middle"
                className={cn(
                  "text-[11.5px] font-semibold",
                  s.tone === "stop" ? "fill-white" : s.tone === "none" ? "fill-slate-400" : "fill-clinical-ink"
                )}
              >
                {label}
              </text>
              {s.sub && (
                <text
                  x={W / 2}
                  y={30}
                  textAnchor="middle"
                  className={cn("text-[9.5px]", s.tone === "stop" ? "fill-red-50" : "fill-clinical-muted")}
                >
                  {s.sub.length > 22 ? `${s.sub.slice(0, 21)}…` : s.sub}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-clinical-muted">
        {[
          ["bg-red-500", "stopped here (cause)"],
          ["bg-amber-400", "flagged / sanitized / affected"],
          ["bg-emerald-500", "passed"],
          ["border border-dashed border-violet-400", "layer switched off"],
          ["border border-dashed border-slate-300", "not reached"],
        ].map(([c, l]) => (
          <span key={l} className="flex items-center gap-1">
            <span className={cn("h-2.5 w-2.5 rounded-sm", c)} /> {l}
          </span>
        ))}
        {aggregate && onSelect && <span>· click a stage to filter the requests below</span>}
      </div>
    </div>
  );
}
