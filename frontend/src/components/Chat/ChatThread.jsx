import { useEffect, useRef } from "react";
import { Stethoscope } from "lucide-react";
import ChatMessage from "./ChatMessage";

export default function ChatThread({ messages, isStreaming }) {
  const messagesEndRef = useRef(null);

  const scrollToBottom = (behavior) => {
    messagesEndRef.current?.scrollIntoView({ behavior, block: "end" });
  };

  useEffect(() => {
    // Smooth scroll settles fine for discrete new messages, but firing it on
    // every single streamed token (many times a second) makes the animation
    // keep restarting and never actually catch up — jump instantly instead
    // while a response is actively streaming.
    scrollToBottom(isStreaming ? "auto" : "smooth");
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return (
      <div className="flex max-h-[calc(100vh-220px)] flex-1 flex-col items-center justify-center gap-3 px-8 text-center">
        <Stethoscope size={32} className="text-clinical-teal" />
        <h2 className="text-lg font-bold text-clinical-ink">
          Hello, I&apos;m your AI Clinical Intake Assistant 👋
        </h2>
        <p className="max-w-md text-sm text-clinical-muted">
          I&apos;m here to help you understand your symptoms, organize your health information, and guide you on
          safe next steps.
        </p>
        <p className="max-w-md text-sm italic text-clinical-muted">
          You&apos;re in a confidential session. Please describe what you are experiencing below.
        </p>
      </div>
    );
  }

  return (
    <div className="max-h-[calc(100vh-220px)] flex-1 space-y-4 overflow-y-auto p-4">
      {messages.map((message, i) => (
        <ChatMessage
          key={i}
          role={message.role}
          content={message.content}
          sources={message.sources}
          audit={message.audit}
          triage={message.triage}
          isStreaming={isStreaming && i === messages.length - 1 && message.role === "assistant"}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}
