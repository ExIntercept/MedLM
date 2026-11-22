import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import Header from "./components/Header";
import { tabsFor } from "./lib/roles";
import Sidebar from "./components/Sidebar/Sidebar";
import ChatThread from "./components/Chat/ChatThread";
import ChatInput from "./components/Chat/ChatInput";
import MedicalRecords from "./components/Records/MedicalRecords";
import Prescriptions from "./components/Prescriptions/Prescriptions";
import EvidenceExplorer from "./components/Evidence/EvidenceExplorer";
import EvaluationDashboard from "./components/EvaluationDashboard";
import AdminConsole from "./components/Admin/AdminConsole";
import LoginPage from "./components/Auth/LoginPage";
import { useChatStream } from "./hooks/useChatStream";
import { useAuth } from "./context/AuthContext";
import { createConversation, getConversationDetail, getConversations, getHealthRecord, saveHealthRecord } from "./lib/api";
import { EMPTY_PROFILE, PROFILE_FIELDS, normalizeProfile } from "./lib/profile";
import { cn } from "./lib/cn";
import { useTheme } from "./lib/theme";


function Workspace() {
  const { user, logout } = useAuth();
  const { theme, toggle: toggleTheme } = useTheme();
  const role = user?.role || "patient";
  const isClinician = role === "clinician" || role === "admin";
  const [activeTab, setActiveTab] = useState("consultation");
  const [profile, setProfile] = useState(EMPTY_PROFILE);
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [mode, setMode] = useState("patient");
  const isPatient = role === "patient";
  const [recentlyUpdated, setRecentlyUpdated] = useState([]);
  const saveTimer = useRef(null);
  const profileRef = useRef(profile);
  profileRef.current = profile;

  // Patients start from their saved health record; clinicians from a blank case.
  const loadRecord = useCallback(async () => {
    if (!isPatient) return EMPTY_PROFILE;
    try {
      const { profile: rec } = await getHealthRecord();
      return normalizeProfile({ ...rec, duration: "" });
    } catch {
      return EMPTY_PROFILE;
    }
  }, [isPatient]);

  useEffect(() => {
    loadRecord().then(setProfile);
  }, [loadRecord]);

  // A patient's own edits are saved to their record shortly after they stop typing.
  const handleProfileChange = (next) => {
    setProfile(next);
    if (!isPatient) return;
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => saveHealthRecord(next).catch(() => {}), 800);
  };

  const refreshConversations = useCallback(() => {
    getConversations()
      .then(setConversations)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  const { messages, isStreaming, error, send, stop, reset } = useChatStream({
    conversationId,
    onConversationId: (id) => {
      setConversationId(id);
      refreshConversations();
    },
    // Auto-fills the Patient Intake sidebar from fields the backend extracts
    // out of the message the user just sent (age/sex/duration/conditions/meds).
    // Only overwrites keys the extractor actually found.
    // The server sends the whole merged profile after each screened turn.
    onProfileUpdate: (fields) => {
      const prev = profileRef.current;
      const next = normalizeProfile(fields);
      const changed = PROFILE_FIELDS.map((f) => f.key).filter((k) => (prev[k] || "") !== (next[k] || ""));
      setProfile(next);
      if (changed.length) {
        setRecentlyUpdated(changed);
        setTimeout(() => setRecentlyUpdated([]), 4000);
      }
    },
  });

  const handleNewConsultation = async () => {
    const start = await loadRecord();
    setProfile(start);
    try {
      const { conversation_id } = await createConversation("New Consultation", start);
      setConversationId(conversation_id);
      reset([]);
      refreshConversations();
    } catch {
      setConversationId(null);
      reset([]);
    }
    setActiveTab("consultation");
  };

  const handleSelectConversation = (id) => {
    setConversationId(id);
    reset([]);
    getConversationDetail(id)
      .then((d) => d.patient_profile && setProfile(normalizeProfile(d.patient_profile)))
      .catch(() => {});
  };

  // Patients always use patient mode; the server enforces this too.
  const chatMode = isClinician ? mode : "patient";
  const visibleTab = tabsFor(role).some((t) => t.key === activeTab) ? activeTab : "consultation";

  const handleSend = (message) => {
    send({ message, patientProfile: profile, mode: chatMode });
  };

  const handleSignOut = () => {
    logout();
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-4 p-4 md:p-6">
      <Header
        username={user?.username || "User"}
        role={role}
        activeTab={visibleTab}
        onTabChange={setActiveTab}
        onSignOut={handleSignOut}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {visibleTab === "consultation" && (
        <div className="flex flex-1 flex-col gap-4 md:flex-row">
          <Sidebar
            profile={profile}
            onProfileChange={handleProfileChange}
            isPatient={isPatient}
            recentlyUpdated={recentlyUpdated}
            sessions={conversations}
            activeSessionId={conversationId}
            onSelectSession={handleSelectConversation}
            onNewConsultation={handleNewConsultation}
          />

          <main className="flex min-h-[70vh] flex-1 flex-col rounded-xl border border-slate-border bg-surface p-4 shadow-card">
            <div className="mb-3 flex items-center justify-between border-b border-slate-border pb-3">
              <h2 className="text-[15px] font-semibold text-clinical-ink">Consultation</h2>
              {isClinician && (
              <div className="flex overflow-hidden rounded-lg border border-slate-border text-xs font-semibold">
                {[
                  { key: "patient", label: "Patient" },
                  { key: "clinician", label: "Clinician" },
                ].map((opt) => (
                  <button
                    key={opt.key}
                    type="button"
                    onClick={() => setMode(opt.key)}
                    className={cn(
                      "px-3 py-1.5 transition",
                      mode === opt.key
                        ? "bg-clinical-teal text-white"
                        : "bg-surface text-clinical-muted hover:bg-slate-bg"
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              )}
            </div>

            <ChatThread messages={messages} isStreaming={isStreaming} showInternals={isClinician} />

            {error && (
              <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                {error}
              </p>
            )}

            <div className="mt-3">
              <ChatInput onSend={handleSend} onStop={stop} disabled={isStreaming} isStreaming={isStreaming} />
            </div>
          </main>
        </div>
      )}

      {visibleTab === "records" && (
        <MedicalRecords sessions={conversations} isPatient={isPatient} onRecordSaved={(p) => setProfile((prev) => normalizeProfile({ ...p, duration: prev.duration }))} />
      )}
      {visibleTab === "prescriptions" && <Prescriptions profile={profile} />}
      {visibleTab === "evidence" && <EvidenceExplorer mode={chatMode} />}
      {visibleTab === "evaluation" && <EvaluationDashboard />}
      {visibleTab === "security" && <AdminConsole />}
    </div>
  );
}

export default function App() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-bg text-clinical-muted">
        <Loader2 size={22} className="animate-spin text-clinical-teal" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return <Workspace />;
}
