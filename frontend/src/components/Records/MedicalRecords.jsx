import { useEffect, useState } from "react";
import { ArrowLeft, FileText } from "lucide-react";
import ChatMessage from "../Chat/ChatMessage";
import { getConversationDetail } from "../../lib/api";
import { PROFILE_FIELDS } from "../../lib/profile";
import { cn } from "../../lib/cn";
import HealthRecordCard from "./HealthRecordCard";

function formatDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso || "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function MedicalRecords({ sessions, isPatient = false, onRecordSaved }) {
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    getConversationDetail(selectedId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [selectedId]);

  const selected = sessions.find((s) => s.id === selectedId);
  const snapshot = detail?.patient_profile
    ? PROFILE_FIELDS.filter((f) => detail.patient_profile[f.key])
    : [];

  return (
    <div className="flex w-full flex-col gap-4">
      {isPatient && <HealthRecordCard onSaved={onRecordSaved} />}

      <div className="grid gap-4 md:grid-cols-[18rem_1fr]">
        {/* Past consultations: hidden on phones once one is open. */}
        <section
          className={cn(
            "rounded-xl border border-slate-border bg-surface p-2",
            selectedId && "hidden md:block"
          )}
        >
          <h2 className="px-2 pb-1 pt-2 text-sm font-semibold text-clinical-ink">Past consultations</h2>
          <div className="max-h-[65vh] space-y-0.5 overflow-y-auto">
            {sessions.length === 0 && (
              <p className="px-2 py-2 text-xs text-clinical-muted">No consultations yet.</p>
            )}
            {sessions.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => setSelectedId(s.id)}
                className={cn(
                  "block w-full rounded-lg px-2.5 py-2 text-left transition",
                  s.id === selectedId ? "bg-clinical-teal-tint" : "hover:bg-slate-bg"
                )}
              >
                <div
                  className={cn(
                    "truncate text-sm",
                    s.id === selectedId ? "font-medium text-clinical-teal-dark" : "text-clinical-ink"
                  )}
                >
                  {s.subject || `Consultation #${s.id}`}
                </div>
                <div className="text-[11px] text-clinical-muted">{formatDate(s.date)}</div>
              </button>
            ))}
          </div>
        </section>

        <section
          className={cn(
            "flex min-h-[50vh] flex-col rounded-xl border border-slate-border bg-surface",
            !selectedId && "hidden md:flex"
          )}
        >
          {!selectedId ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center text-clinical-muted">
              <FileText size={26} className="text-clinical-teal" />
              <p className="text-sm">Pick a consultation to read it again.</p>
            </div>
          ) : (
            <>
              <header className="flex items-center gap-2 border-b border-slate-border px-4 py-3">
                <button
                  type="button"
                  onClick={() => setSelectedId(null)}
                  aria-label="Back to the list"
                  className="rounded-md p-1 text-clinical-muted transition hover:bg-slate-bg md:hidden"
                >
                  <ArrowLeft size={16} />
                </button>
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-clinical-ink">
                    {selected?.subject || `Consultation #${selectedId}`}
                  </div>
                  <div className="text-[11px] text-clinical-muted">
                    {selected && formatDate(selected.date)} · read-only
                  </div>
                </div>
              </header>

              <div className="flex-1 overflow-y-auto px-4 py-4">
                {loading && <p className="text-sm text-clinical-muted">Loading…</p>}
                {!loading && !detail && (
                  <p className="text-sm text-clinical-muted">This consultation could not be loaded.</p>
                )}
                {!loading && detail && (
                  <div className="mx-auto max-w-3xl space-y-6">
                    {snapshot.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        <span className="mr-1 self-center text-[11px] text-clinical-muted">At the time:</span>
                        {snapshot.map((f) => (
                          <span
                            key={f.key}
                            title={f.label}
                            className="rounded-full border border-slate-border bg-slate-bg px-2.5 py-0.5 text-xs text-clinical-ink"
                          >
                            {detail.patient_profile[f.key]}
                          </span>
                        ))}
                      </div>
                    )}
                    {detail.messages.length === 0 && (
                      <p className="text-sm text-clinical-muted">No messages in this consultation.</p>
                    )}
                    {detail.messages.map((m, i) => (
                      <ChatMessage key={i} role={m.role} content={m.content} />
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
