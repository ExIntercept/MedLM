import { useState } from "react";
import { Check, Pencil } from "lucide-react";
import { cn } from "../../lib/cn";
import { PROFILE_FIELDS } from "../../lib/profile";

const SHORT = {
  age: (v) => `${v} yrs`,
  sex: (v) => v,
  duration: (v) => `for ${v}`,
  allergies: (v) => `Allergy: ${v}`,
  past_medications: (v) => `Previously: ${v}`,
};

/**
 * The details the assistant knows. Compact by default (a few chips); "Edit"
 * opens the full form. Fields picked up from the chat briefly highlight.
 */
export default function PatientIntakeForm({ profile, onChange, isPatient = true, recentlyUpdated = [] }) {
  const [editing, setEditing] = useState(false);
  const filled = PROFILE_FIELDS.filter((f) => (profile[f.key] || "").trim());

  return (
    <section data-tour="details" className="rounded-xl border border-slate-border bg-surface p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-clinical-ink">{isPatient ? "Your details" : "Case details"}</h2>
        <button
          type="button"
          onClick={() => setEditing((v) => !v)}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-clinical-teal-dark transition hover:bg-clinical-teal-tint"
        >
          {editing ? <Check size={13} /> : <Pencil size={12} />}
          {editing ? "Done" : "Edit"}
        </button>
      </div>

      {!editing && (
        <div className="mt-3">
          {filled.length === 0 ? (
            <p className="text-xs leading-relaxed text-clinical-muted">
              {isPatient
                ? "Nothing yet. Things you mention, like your age, conditions, medicines or allergies, are saved here automatically."
                : "Details of the case you describe appear here for this consultation."}
            </p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {filled.map((f) => (
                <span
                  key={f.key}
                  title={f.label}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-xs transition-colors",
                    recentlyUpdated.includes(f.key)
                      ? "border-clinical-teal bg-clinical-teal-tint text-clinical-teal-dark"
                      : "border-slate-border bg-slate-bg text-clinical-ink"
                  )}
                >
                  {(SHORT[f.key] || ((v) => v))(profile[f.key])}
                </span>
              ))}
            </div>
          )}
          {recentlyUpdated.length > 0 && (
            <p className="mt-2 text-[11px] font-medium text-clinical-teal-dark">Updated from your last message</p>
          )}
        </div>
      )}

      {editing && (
        <div className="mt-3 space-y-2.5">
          {PROFILE_FIELDS.map((field) => (
            <label key={field.key} className="block">
              <span className="mb-1 block text-xs text-clinical-muted">{field.label}</span>
              <input
                type="text"
                value={profile[field.key] ?? ""}
                placeholder={field.placeholder}
                onChange={(e) => onChange({ ...profile, [field.key]: e.target.value })}
                className="w-full rounded-lg border border-slate-border bg-slate-bg px-3 py-1.5 text-sm text-clinical-ink outline-none transition focus:border-clinical-teal focus:bg-surface focus:ring-2 focus:ring-clinical-teal/20"
              />
            </label>
          ))}
          {isPatient && <p className="text-[11px] text-clinical-muted">Changes are saved to your record.</p>}
        </div>
      )}
    </section>
  );
}
