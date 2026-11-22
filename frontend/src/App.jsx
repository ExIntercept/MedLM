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
    <div className="flex min-h-screen flex-col">
      <Header
        username={user?.username || "User"}
        role={role}
        activeTab={visibleTab}
        onTabChange={setActiveTab}
        onSignOut={handleSignOut}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-4 px-4 py-5 md:px-6">
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

            <main className="flex h-[calc(100vh-7.5rem)] min-h-[480px] flex-1 flex-col rounded-xl border border-slate-border bg-surface px-3 pb-3 pt-2 md:px-5">
              {isClinician && (
                <div className="flex items-center justify-end gap-2 pt-1 text-xs text-clinical-muted">
                  Answer style
                  <div className="flex rounded-lg bg-slate-bg p-0.5 font-medium">
                    {[
                      { key: "patient", label: "Plain language" },
                      { key: "clinician", label: "Clinical" },
                    ].map((opt) => (
                      <button
                        key={opt.key}
                        type="button"
                        onClick={() => setMode(opt.key)}
                        className={cn(
                          "rounded-md px-2.5 py-1 transition",
                          mode === opt.key ? "bg-surface text-clinical-ink shadow-card" : "text-clinical-muted hover:text-clinical-ink"
                        )}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <ChatThread
                messages={messages}
                isStreaming={isStreaming}
                showInternals={isClinician}
                mode={isPatient ? "patient" : "clinician"}
                onPick={handleSend}
              />

              {error && (
                <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>
              )}

              <div className="mx-auto w-full max-w-3xl">
                <ChatInput
                  onSend={handleSend}
                  onStop={stop}
                  disabled={isStreaming}
                  isStreaming={isStreaming}
                  placeholder={isPatient ? "Describe how you're feeling, or ask a question…" : "Ask a clinical question or describe a case…"}
                />
              </div>
            </main>
          </div>
        )}

        {visibleTab === "records" && (
          <MedicalRecords
            sessions={conversations}
            isPatient={isPatient}
            onRecordSaved={(p) => setProfile((prev) => normalizeProfile({ ...p, duration: prev.duration }))}
          />
        )}
        {visibleTab === "prescriptions" && <Prescriptions profile={profile} />}
        {visibleTab === "evidence" && <EvidenceExplorer mode={chatMode} />}
        {visibleTab === "evaluation" && <EvaluationDashboard />}
        {visibleTab === "security" && <AdminConsole />}
      </div>
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
