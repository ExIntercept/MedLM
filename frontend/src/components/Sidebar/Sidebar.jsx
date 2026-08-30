import PatientIntakeForm from "./PatientIntakeForm";
import ConsultationHistory from "./ConsultationHistory";

export default function Sidebar({
  profile,
  onProfileChange,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewConsultation,
}) {
  return (
    <aside className="flex w-full flex-col gap-4 md:w-80 md:shrink-0">
      <PatientIntakeForm profile={profile} onChange={onProfileChange} />
      <ConsultationHistory
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={onSelectSession}
        onNewConsultation={onNewConsultation}
      />
    </aside>
  );
}
