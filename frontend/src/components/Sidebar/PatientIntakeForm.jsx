import { ClipboardList } from "lucide-react";

const FIELDS = [
  { key: "age", label: "Age", placeholder: "e.g. 34" },
  { key: "sex", label: "Biological Sex", placeholder: "e.g. Female" },
  { key: "duration", label: "Symptom Duration", placeholder: "e.g. 2 weeks" },
  { key: "conditions", label: "Existing Conditions", placeholder: "e.g. Type 2 diabetes" },
  { key: "medications", label: "Medications", placeholder: "e.g. Metformin 500mg" },
];

export default function PatientIntakeForm({ profile, onChange }) {
  return (
    <div className="rounded-xl border border-slate-border bg-white p-4 shadow-card">
      <div className="mb-3 flex items-center gap-2 border-b border-slate-border pb-3">
        <ClipboardList size={16} className="text-clinical-teal" />
        <h2 className="text-[15px] font-semibold text-clinical-ink">Patient Intake Form</h2>
      </div>
      <div className="space-y-3">
        {FIELDS.map((field) => (
          <label key={field.key} className="block">
            <span className="mb-1 block text-xs font-medium text-clinical-muted">{field.label}</span>
            <input
              type="text"
              value={profile[field.key] ?? ""}
              placeholder={field.placeholder}
              onChange={(e) => onChange({ ...profile, [field.key]: e.target.value })}
              className="w-full rounded-lg border border-slate-border bg-slate-bg px-3 py-2 text-sm text-clinical-ink outline-none transition focus:border-clinical-teal focus:bg-white focus:ring-2 focus:ring-clinical-teal/20"
            />
          </label>
        ))}
      </div>
    </div>
  );
}
