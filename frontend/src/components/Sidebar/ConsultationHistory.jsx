import { cn } from "../../lib/cn";

export function relativeTime(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const mins = Math.round((Date.now() - date.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days} d ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function ConsultationHistory({ sessions, activeSessionId, onSelect }) {
  return (
    <section className="rounded-xl border border-slate-border bg-surface p-2">
      <h2 className="px-2 pb-1 pt-2 text-sm font-semibold text-clinical-ink">Recent</h2>
      <div className="max-h-[45vh] space-y-0.5 overflow-y-auto">
        {sessions.length === 0 && <p className="px-2 py-2 text-xs text-clinical-muted">No consultations yet.</p>}
        {sessions.map((session) => (
          <button
            key={session.id}
            type="button"
            onClick={() => onSelect(session.id)}
            className={cn(
              "block w-full rounded-lg px-2.5 py-2 text-left transition",
              session.id === activeSessionId ? "bg-clinical-teal-tint" : "hover:bg-slate-bg"
            )}
          >
            <div
              className={cn(
                "truncate text-sm",
                session.id === activeSessionId ? "font-medium text-clinical-teal-dark" : "text-clinical-ink"
              )}
            >
              {session.subject || `Consultation #${session.id}`}
            </div>
            <div className="text-[11px] text-clinical-muted">{relativeTime(session.date)}</div>
          </button>
        ))}
      </div>
    </section>
  );
}
