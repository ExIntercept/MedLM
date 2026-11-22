import { useEffect, useState } from "react";
import { Check, HeartPulse, Loader2, Pencil } from "lucide-react";
import { getHealthRecord, saveHealthRecord } from "../../lib/api";
import { PROFILE_FIELDS, normalizeProfile } from "../../lib/profile";

const RECORD_FIELDS = PROFILE_FIELDS.filter((f) => !f.visitOnly);

function Items({ value }) {
  const items = (value || "").split(",").map((s) => s.trim()).filter(Boolean);
  if (!items.length) return <span className="text-clinical-muted">Not recorded</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {items.map((it) => (
        <span key={it} className="rounded-md border border-slate-border bg-slate-bg px-1.5 py-0.5 text-xs text-clinical-ink">
          {it}
        </span>
      ))}
    </span>
  );
}

/** The patient's own long-term record: what they've told the assistant, editable. */
export default function HealthRecordCard({ onSaved }) {
  const [record, setRecord] = useState(null);
  const [updatedAt, setUpdatedAt] = useState(null);
  const [draft, setDraft] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealthRecord()
      .then(({ profile, updated_at }) => {
        setRecord(normalizeProfile(profile));
        setUpdatedAt(updated_at);
      })
      .catch((e) => setError(e.message));
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const { profile } = await saveHealthRecord(draft);
      const next = normalizeProfile(profile);
      setRecord(next);
      setUpdatedAt(new Date().toISOString());
      setDraft(null);
      onSaved?.(next);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-xl border border-slate-border bg-surface p-4 shadow-card">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="flex items-center gap-2 text-[15px] font-semibold text-clinical-ink">
            <HeartPulse size={16} className="text-clinical-teal" /> My health record
          </h2>
          <p className="mt-0.5 text-xs text-clinical-muted">
            Built from what you&apos;ve told the assistant across your consultations
            {updatedAt && ` · last updated ${new Date(updatedAt).toLocaleString()}`}
          </p>
        </div>
        {record && !draft && (
          <button
            type="button"
            onClick={() => setDraft(record)}
            className="flex items-center gap-1.5 rounded-lg border border-slate-border px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-bg"
          >
            <Pencil size={13} /> Edit
          </button>
        )}
        {draft && (
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setDraft(null)}
              className="rounded-lg border border-slate-border px-3 py-1.5 text-xs font-semibold text-clinical-muted hover:bg-slate-bg"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={save}
              disabled={saving}
              className="flex items-center gap-1.5 rounded-lg bg-clinical-teal px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-60"
            >
              {saving ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />} Save
            </button>
          </div>
        )}
      </div>

      {error && <p className="mb-2 rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700">{error}</p>}
      {!record && !error && <Loader2 size={16} className="animate-spin text-clinical-teal" />}

      {record && (
        <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
          {RECORD_FIELDS.map((f) => (
            <div key={f.key}>
              <dt className="text-[11px] font-semibold uppercase tracking-wide text-clinical-muted">{f.label}</dt>
              <dd className="mt-1 text-sm text-clinical-ink">
                {draft ? (
                  <input
                    type="text"
                    value={draft[f.key] ?? ""}
                    placeholder={f.placeholder}
                    onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}
                    className="w-full rounded-lg border border-slate-border bg-slate-bg px-3 py-1.5 text-sm text-clinical-ink outline-none focus:border-clinical-teal focus:bg-surface"
                  />
                ) : ["age", "sex"].includes(f.key) ? (
                  record[f.key] || <span className="text-clinical-muted">Not recorded</span>
                ) : (
                  <Items value={record[f.key]} />
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {draft && <p className="mt-3 text-[11px] text-clinical-muted">Separate multiple items with commas.</p>}
    </div>
  );
}
