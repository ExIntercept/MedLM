import { ClipboardList, Sparkles } from "lucide-react";
import { PROFILE_FIELDS } from "../../lib/profile";

export default function PatientIntakeForm({ profile, onChange, isPatient = true, recentlyUpdated = [] }) {
  return (
    <div className="rounded-xl border border-slate-border bg-surface p-4 shadow-card">
      <div className="mb-3 border-b border-slate-border pb-3">
        <div className="flex items-center gap-2">
          <ClipboardList size={16} className="text-clinical-teal" />
          <h2 className="text-[15px] font-semibold text-clinical-ink">
            {isPatient ? "Your health details" : "Patient intake"}
          </h2>
        </div>
        <p className="mt-1 text-[11px] leading-snug text-clinical-muted">
          {isPatient
            ? "Filled in from what you tell the assistant and saved to your record. Edit anything that's wrong."
            : "Filled in from the case as you describe it. Kept with this consultation only."}
        </p>
      </div>
      <div className="space-y-3">
        {PROFILE_FIELDS.map((field) => {
          const fresh = recentlyUpdated.includes(field.key);
          return (
            <label key={field.key} className="block">
              <span className="mb-1 flex items-center gap-1 text-xs font-medium text-clinical-muted">
                {field.label}
                {fresh && (
                  <span className="inline-flex items-center gap-0.5 rounded bg-clinical-teal-tint px-1 text-[10px] font-semibold text-clinical-teal-dark">
                    <Sparkles size={10} /> updated
                  </span>
                )}
              </span>
              <input
                type="text"
                value={profile[field.key] ?? ""}
                placeholder={field.placeholder}
                onChange={(e) => onChange({ ...profile, [field.key]: e.target.value })}
                className={
                  "w-full rounded-lg border bg-slate-bg px-3 py-2 text-sm text-clinical-ink outline-none transition focus:border-clinical-teal focus:bg-surface focus:ring-2 focus:ring-clinical-teal/20 " +
                  (fresh ? "border-clinical-teal" : "border-slate-border")
                }
              />
            </label>
          );
        })}
      </div>
    </div>
  );
}
