import { useEffect, useState } from "react";
import { Pill, ShieldAlert, ShieldCheck } from "lucide-react";
import { checkMedications } from "../../lib/api";

export default function Prescriptions({ profile }) {
  const medications = (profile.medications || "").trim();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!medications) {
      setResult(null);
      return;
    }
    setLoading(true);
    checkMedications(medications, profile.conditions)
      .then(setResult)
      .catch(() => setResult(null))
      .finally(() => setLoading(false));
  }, [medications, profile.conditions]);

  return (
    <div className="rounded-xl border border-slate-border bg-white p-4 shadow-card">
      <div className="mb-3 flex items-center gap-2 border-b border-slate-border pb-3">
        <Pill size={16} className="text-clinical-teal" />
        <h2 className="text-[15px] font-semibold text-clinical-ink">Active Medications &amp; Prescriptions</h2>
      </div>
      <p className="mb-3 text-xs text-clinical-muted">
        Sourced from the Patient Intake panel under Active Consultation. Automatically screened against known
        hard-rule contraindications.
      </p>

      {!medications && (
        <p className="text-sm text-clinical-muted">
          No medications currently recorded. Add them in the Patient Intake panel under Active Consultation.
        </p>
      )}

      {medications && loading && <p className="text-sm text-clinical-muted">Checking…</p>}

      {medications && !loading && result && (
        <div className="space-y-3">
          <div>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-clinical-muted">
              Confirmed Medications
            </h3>
            <ul className="list-disc space-y-1 pl-5 text-sm text-clinical-ink">
              {result.medications.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </div>

          {result.alert ? (
            <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-700">
              <ShieldAlert size={16} className="mt-0.5 shrink-0" />
              <div>
                <div className="font-semibold">Automated Contraindication Alert</div>
                <p>{result.alert}</p>
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2.5 text-sm text-emerald-700">
              <ShieldCheck size={16} className="mt-0.5 shrink-0" />
              <p>
                No known hard-rule contraindications detected against the recorded conditions/medications. This
                automated check is not a substitute for pharmacist or physician medication reconciliation.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
