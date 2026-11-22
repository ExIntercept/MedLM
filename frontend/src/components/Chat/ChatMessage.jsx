import { useState } from "react";
import { ChevronDown, ShieldAlert, ShieldCheck, Siren, Stethoscope } from "lucide-react";
import { cn } from "../../lib/cn";
import { Markdown } from "../../lib/markdown";
import { STAGES, guardCategoryLabels, reasonLabels } from "../../lib/defense";
import EvidenceAccordion from "./EvidenceAccordion";
import ClarifyCard from "./ClarifyCard";

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
          {STAGES.filter((s) => !s.optional || trace.includes(s.key)).map((s, i, shown) => {
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
                {i < shown.length - 1 && <span className="text-slate-300">›</span>}
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

export default function ChatMessage({
  role, content, sources, audit, triage, isStreaming, showInternals = false,
  clarify, clarifyActive = false, onAnswer,
}) {
  // A clarification renders its questions as a card; the plain text (kept for
  // history and screen readers) shows only its first and last lines.
  const asking = !isStreaming && clarify?.length > 0;
  const lines = asking ? content.split(/\n/).filter((l) => l.trim()) : [];
  const isUser = role === "user";
  const isEmergency = triage?.level === "EMERGENCY";
  return (
    <div className={cn("flex gap-3", isUser && "justify-end")}>
      {!isUser && (
        <div
          className={cn(
            "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
            isEmergency ? "bg-red-100 text-red-600" : "bg-clinical-teal-tint text-clinical-teal-dark"
          )}
        >
          {isEmergency ? <Siren size={15} /> : <Stethoscope size={15} />}
        </div>
      )}
      <div
        className={cn(
          isUser
            ? "max-w-[80%] rounded-2xl rounded-br-md bg-clinical-teal-tint px-4 py-2.5 text-clinical-ink"
            : isEmergency
            ? "min-w-0 flex-1 rounded-xl border border-red-300 bg-red-50 px-4 py-3 text-clinical-ink"
            : "min-w-0 flex-1 pt-0.5 text-clinical-ink"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{content}</p>
        ) : (
          <>
            <TriageBadge triage={triage} />
            {asking ? (
              <>
                <p className="text-[15px] leading-relaxed">{lines[0]}</p>
                <ClarifyCard questions={clarify} active={clarifyActive && !!onAnswer} onAnswer={onAnswer} />
                <p className="mt-2 text-xs leading-relaxed text-clinical-muted">{lines[lines.length - 1]}</p>
              </>
            ) : (
              <Markdown text={content} className="text-[15px] leading-relaxed" />
            )}
            {isStreaming && content === "" && (
              <span className="inline-flex gap-1 py-1">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-clinical-teal [animation-delay:-0.2s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-clinical-teal [animation-delay:-0.1s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-clinical-teal" />
              </span>
            )}
            <EvidenceAccordion sources={sources} showScores={showInternals} />
            {showInternals && <AuditBadge audit={audit} />}
          </>
        )}
      </div>
    </div>
  );
}
