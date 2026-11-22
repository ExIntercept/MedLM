import { useState } from "react";
import { ChevronDown, ShieldAlert, ShieldCheck, Siren, Stethoscope, UserRound } from "lucide-react";
import { cn } from "../../lib/cn";
import { Markdown } from "../../lib/markdown";
import { STAGES, guardCategoryLabels, reasonLabels } from "../../lib/defense";
import EvidenceAccordion from "./EvidenceAccordion";

const RED = "bg-red-50 text-red-700 border-red-200";
const AMBER = "bg-amber-50 text-amber-700 border-amber-200";
const GREEN = "bg-emerald-50 text-emerald-700 border-emerald-200";

/** What the user was shown, and which part of the gateway decided it. */
function describeOutcome(audit) {
  const d = audit.defense || {};
  const score = audit.score != null ? `Faithfulness ${(audit.score * 100).toFixed(0)}%` : null;
  switch (audit.status) {
    case "BLOCKED": {
      const by = reasonLabels(d.reasons).filter((r) => !r.startsWith("Control"));
      const cats = guardCategoryLabels(d.guard_categories);
      return {
        tone: RED, blocked: true, label: "Blocked before the model",
        detail: [by.length ? `by ${by.join(", ")}` : "by the security policy",
                 cats.length ? `Llama Guard: ${cats.join(", ")}` : null].filter(Boolean),
      };
    }
    case "OUTPUT_BLOCKED": {
      const cats = guardCategoryLabels(d.output_guard?.categories);
      return {
        tone: RED, blocked: true, label: "Answer blocked by the output guard",
        detail: [cats.length ? `Unsafe: ${cats.join(", ")}` : "The model's answer was judged unsafe"],
      };
    }
    case "WITHHELD":
      return { tone: AMBER, blocked: true, label: "Withheld · not supported by the evidence", detail: [score].filter(Boolean) };
    case "INTERCEPTED":
      return { tone: RED, blocked: true, label: "Intercepted · contraindication guardrail", detail: [] };
    case "EMERGENCY_ESCALATION":
      return { tone: RED, blocked: true, label: "Escalated · emergency triage", detail: [] };
    case "PASS":
      return { tone: GREEN, label: "Verified against evidence", detail: [score].filter(Boolean) };
    default:
      return { tone: AMBER, label: audit.status, detail: [score].filter(Boolean) };
  }
}

function AuditBadge({ audit }) {
  const [open, setOpen] = useState(false);
  if (!audit) return null;
  const d = audit.defense || {};
  const outcome = describeOutcome(audit);
  const Icon = outcome.blocked ? ShieldAlert : ShieldCheck;
  const trace = d.trace || [];
  const flaggedBy = d.flagged ? reasonLabels(d.reasons) : [];
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold", outcome.tone)}
        aria-expanded={open}
      >
        <Icon size={13} />
        {outcome.label}
        {outcome.detail.length > 0 && <span className="font-normal">· {outcome.detail.join(" · ")}</span>}
        {trace.length > 0 && <ChevronDown size={13} className={cn("transition", open && "rotate-180")} />}
      </button>
      {d.flagged && !outcome.blocked && (
        <span className="ml-2 inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] font-semibold text-amber-700">
          Flagged{flaggedBy.length ? ` · ${flaggedBy.join(", ")}` : ""} · answered with extra safety rules
        </span>
      )}
      {open && trace.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1 text-[11px]">
          {STAGES.map((s, i) => {
            const ran = trace.includes(s.key) || (s.key === "harden" && trace.includes("prompt"));
            return (
              <span key={s.key} className="flex items-center gap-1">
                <span
                  className={cn(
                    "rounded border px-1.5 py-0.5",
                    ran ? "border-clinical-teal-tint bg-clinical-teal-tint text-clinical-teal-dark" : "border-slate-border text-slate-400 line-through"
                  )}
                  title={ran ? "ran" : "did not run"}
                >
                  {s.label}
                </span>
                {i < STAGES.length - 1 && <span className="text-slate-300">›</span>}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function TriageBadge({ triage }) {
  if (!triage || triage.level !== "EMERGENCY") return null;
  return (
    <div className="mb-2 inline-flex items-center gap-1.5 rounded-full border border-red-300 bg-red-600 px-3 py-1 text-xs font-bold text-white shadow-sm">
      <Siren size={13} />
      EMERGENCY {triage.category ? `· ${triage.category}` : ""}
    </div>
  );
}

export default function ChatMessage({ role, content, sources, audit, triage, isStreaming }) {
  const isUser = role === "user";
  const isEmergency = triage?.level === "EMERGENCY";
  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-clinical-ink text-surface"
            : isEmergency
            ? "bg-red-100 text-red-600"
            : "bg-clinical-teal-tint text-clinical-teal-dark"
        )}
      >
        {isUser ? <UserRound size={16} /> : isEmergency ? <Siren size={16} /> : <Stethoscope size={16} />}
      </div>
      <div
        className={cn(
          "max-w-[75%] rounded-xl border px-4 py-3 shadow-card",
          isUser
            ? "border-clinical-ink bg-clinical-ink text-surface"
            : isEmergency
            ? "border-red-300 bg-red-50 text-clinical-ink"
            : "border-slate-border bg-surface text-clinical-ink"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{content}</p>
        ) : (
          <>
            <TriageBadge triage={triage} />
            <Markdown text={content} className="text-sm" />
            {isStreaming && content === "" && (
              <span className="inline-flex gap-1 py-1">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-clinical-teal [animation-delay:-0.2s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-clinical-teal [animation-delay:-0.1s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-clinical-teal" />
              </span>
            )}
            <EvidenceAccordion sources={sources} />
            <AuditBadge audit={audit} />
          </>
        )}
      </div>
    </div>
  );
}
