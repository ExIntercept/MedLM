import { useEffect, useRef } from "react";
import { Stethoscope } from "lucide-react";
import ChatMessage from "./ChatMessage";

const STARTERS = {
  patient: [
    "I've had a headache for three days. What could be causing it?",
    "Can I take ibuprofen with my blood pressure tablets?",
    "What are the early signs of dehydration?",
    "How can I sleep better with back pain?",
  ],
  clinician: [
    "First-line treatment for uncomplicated hypertension in adults?",
    "Interactions to check before starting a patient on warfarin",
    "Red flags in a patient presenting with headache",
    "Metformin dosing in reduced kidney function",
  ],
};

export default function ChatThread({ messages, isStreaming, showInternals = false, mode = "patient", onPick }) {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Instant scroll while tokens stream in; smooth otherwise.
    messagesEndRef.current?.scrollIntoView({ behavior: isStreaming ? "auto" : "smooth", block: "end" });
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    const isPatient = mode === "patient";
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-4 py-10 text-center">
        <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-clinical-teal-tint text-clinical-teal-dark">
          <Stethoscope size={24} />
        </span>
        <h2 className="mt-4 text-xl font-semibold text-clinical-ink">
          {isPatient ? "How can I help today?" : "What would you like to look into?"}
        </h2>
        <p className="mt-1 max-w-md text-sm text-clinical-muted">
          {isPatient
            ? "Describe how you're feeling or ask a health question. Answers are checked against medical guidelines."
            : "Ask a clinical question or describe a case. Answers cite the guidelines they rely on."}
        </p>
        {onPick && (
          <div className="mt-6 grid w-full max-w-xl gap-2 sm:grid-cols-2">
            {STARTERS[isPatient ? "patient" : "clinician"].map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onPick(s)}
                className="rounded-xl border border-slate-border bg-surface px-3.5 py-3 text-left text-sm text-clinical-ink transition hover:border-clinical-teal hover:bg-clinical-teal-tint"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {isPatient && (
          <p className="mt-6 max-w-md text-[11px] leading-relaxed text-clinical-muted">
            Not a replacement for a doctor. In an emergency call your local emergency number. Conversations may be
            reviewed by administrators for safety, and every review is logged.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-1 py-4 md:px-4">
      <div className="mx-auto w-full max-w-3xl space-y-6">
      {messages.map((message, i) => (
        <ChatMessage
          key={i}
          role={message.role}
          content={message.content}
          sources={message.sources}
          audit={message.audit}
          triage={message.triage}
          showInternals={showInternals}
          isStreaming={isStreaming && i === messages.length - 1 && message.role === "assistant"}
        />
      ))}
      <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
