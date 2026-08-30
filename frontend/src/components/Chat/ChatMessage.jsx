import { Siren, Stethoscope, UserRound } from "lucide-react";
import { cn } from "../../lib/cn";
import { Markdown } from "../../lib/markdown";
import EvidenceAccordion from "./EvidenceAccordion";

function AuditBadge({ audit }) {
  if (!audit) return null;
  const tone =
    audit.status === "PASS"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : audit.status === "INTERCEPTED"
      ? "bg-red-50 text-red-700 border-red-200"
      : "bg-amber-50 text-amber-700 border-amber-200";
  return (
    <div className={cn("mt-3 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold", tone)}>
      {audit.status} · Faithfulness {(audit.score * 100).toFixed(0)}%
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
            ? "bg-clinical-ink text-white"
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
            ? "border-clinical-teal-dark bg-clinical-ink text-white"
            : isEmergency
            ? "border-red-300 bg-red-50 text-clinical-ink"
            : "border-slate-border bg-white text-clinical-ink"
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
