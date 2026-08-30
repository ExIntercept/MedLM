import { useEffect, useState } from "react";
import { Stethoscope } from "lucide-react";
import ChatMessage from "../Chat/ChatMessage";
import { getConversationDetail } from "../../lib/api";

function formatSessionLabel(session) {
  const date = new Date(session.date);
  const stamp = Number.isNaN(date.getTime())
    ? session.date
    : date.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
  return `📅 ${stamp} — ${session.subject}`;
}

export default function MedicalRecords({ sessions }) {
  const [selectedId, setSelectedId] = useState("");
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

  return (
    <div className="flex w-full flex-col gap-4">
      <div className="rounded-xl border border-slate-border bg-white p-4 shadow-card">
        <h2 className="mb-3 text-[15px] font-semibold text-clinical-ink">Archived Consultation Transcripts</h2>
        <select
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
          className="w-full rounded-lg border border-slate-border bg-slate-bg px-3 py-2 text-sm text-clinical-ink outline-none transition focus:border-clinical-teal focus:bg-white focus:ring-2 focus:ring-clinical-teal/20"
        >
          <option value="">Select a past consultation…</option>
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {formatSessionLabel(s)}
            </option>
          ))}
        </select>
      </div>

      <div className="flex min-h-[60vh] flex-1 flex-col rounded-xl border border-slate-border bg-white p-4 shadow-card">
        {!selectedId && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-clinical-muted">
            <Stethoscope size={28} className="text-clinical-teal" />
            <p className="text-sm">Select a past consultation above to view its full transcript.</p>
          </div>
        )}
        {selectedId && loading && <p className="text-sm text-clinical-muted">Loading transcript…</p>}
        {selectedId && !loading && detail && (
          <div className="flex-1 space-y-5 overflow-y-auto px-1 py-2">
            {detail.messages.length === 0 && (
              <p className="text-sm text-clinical-muted">No messages recorded for this consultation.</p>
            )}
            {detail.messages.map((m, i) => (
              <ChatMessage key={i} role={m.role} content={m.content} />
            ))}
          </div>
        )}

        <div className="mt-3 flex items-end gap-2 rounded-xl border border-slate-border bg-slate-bg p-2">
          <textarea
            disabled
            rows={1}
            placeholder="📁 Viewing archived consultation (Read-only). Switch to Active Consultation to start a new chat."
            className="max-h-[200px] flex-1 cursor-not-allowed resize-none bg-transparent px-2 py-2 text-sm text-clinical-muted outline-none placeholder:text-clinical-muted"
          />
          <button
            type="button"
            disabled
            className="flex shrink-0 cursor-not-allowed items-center gap-1.5 rounded-lg bg-slate-300 px-4 py-2 text-sm font-semibold text-white opacity-60"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
