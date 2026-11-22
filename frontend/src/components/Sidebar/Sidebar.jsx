import { Plus } from "lucide-react";
import PatientIntakeForm from "./PatientIntakeForm";
import ConsultationHistory from "./ConsultationHistory";

export default function Sidebar({
  profile,
  onProfileChange,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewConsultation,
  isPatient,
  recentlyUpdated,
}) {
  return (
    <aside className="flex w-full flex-col gap-3 md:w-72 md:shrink-0">
      <button
        type="button"
        data-tour="new-consultation"
        onClick={onNewConsultation}
        className="flex w-full items-center justify-center gap-1.5 rounded-xl bg-clinical-teal px-3 py-2.5 text-sm font-semibold text-white shadow-card transition hover:bg-clinical-teal-hover"
      >
        <Plus size={16} />
        New consultation
      </button>
      <PatientIntakeForm
        profile={profile}
        onChange={onProfileChange}
        isPatient={isPatient}
        recentlyUpdated={recentlyUpdated}
      />
      <ConsultationHistory sessions={sessions} activeSessionId={activeSessionId} onSelect={onSelectSession} />
    </aside>
  );
}
