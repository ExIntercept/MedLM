import { FolderClock, Plus } from "lucide-react";
import { cn } from "../../lib/cn";

function formatTimestamp(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ConsultationHistory({ sessions, activeSessionId, onSelect, onNewConsultation }) {
  return (
    <div className="rounded-xl border border-slate-border bg-white p-4 shadow-card">
      <div className="mb-3 flex items-center justify-between border-b border-slate-border pb-3">
        <div className="flex items-center gap-2">
          <FolderClock size={16} className="text-clinical-teal" />
          <h2 className="text-[15px] font-semibold text-clinical-ink">Consultation History</h2>
        </div>
        <span className="text-xs font-semibold text-clinical-muted">
          {sessions.length} Total Session{sessions.length === 1 ? "" : "s"}
        </span>
      </div>

      <button
        type="button"
        onClick={onNewConsultation}
        className="mb-3 flex w-full items-center justify-center gap-1.5 rounded-lg border border-clinical-teal bg-transparent px-3 py-2 text-sm font-semibold text-clinical-teal transition hover:bg-clinical-teal-tint hover:border-clinical-teal-dark hover:text-clinical-teal-dark"
      >
        <Plus size={15} />
        New Consultation
      </button>

      <div className="max-h-80 space-y-1 overflow-y-auto">
        {sessions.length === 0 && (
          <p className="px-1 py-2 text-sm text-clinical-muted">No past consultations yet.</p>
        )}
        {sessions.map((session) => (
          <button
            key={session.id}
            type="button"
            onClick={() => onSelect(session.id)}
            className={cn(
              "block w-full rounded-lg px-3 py-2 text-left text-sm transition",
              session.id === activeSessionId
                ? "bg-clinical-teal-tint text-clinical-teal-dark font-semibold"
                : "text-slate-600 hover:bg-slate-bg"
            )}
          >
            <div className="truncate">{session.subject || `Consultation #${session.id}`}</div>
            <div className="text-xs text-clinical-muted">{formatTimestamp(session.date)}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
