import { useCallback, useRef, useState } from "react";
import { streamChat } from "../lib/api";
import { useToast } from "../context/ToastContext";

/**
 * Drives one clinical consultation thread: sends a message to
 * POST /api/chat/stream and renders retrieved sources + streamed tokens +
 * the post-generation audit as they arrive over SSE.
 */
export function useChatStream({ conversationId, onConversationId, onProfileUpdate }) {
  const [messages, setMessages] = useState([]); // [{ role, content, sources?, audit?, triage? }]
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);
  const { showToast } = useToast();

  const send = useCallback(
    async ({ message, patientProfile, mode }) => {
      setError(null);
      setIsStreaming(true);

      const history = messages.map(({ role, content }) => ({ role, content }));
      setMessages((prev) => [
        ...prev,
        { role: "user", content: message },
        { role: "assistant", content: "", sources: [], audit: null, triage: null },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;

      const patchAssistant = (patch) => {
        setMessages((prev) => {
          const next = [...prev];
          const idx = next.length - 1;
          next[idx] = { ...next[idx], ...patch };
          return next;
        });
      };

      try {
        await streamChat(
          {
            conversation_id: conversationId,
            message,
            patient_profile: patientProfile,
            history,
            mode,
          },
          ({ event, data }) => {
            if (event === "conversation") {
              onConversationId?.(Number(data));
            } else if (event === "sources") {
              try {
                patchAssistant({ sources: JSON.parse(data) });
              } catch (e) {
                console.warn("Skipping unparseable 'sources' SSE payload:", data, e);
              }
            } else if (event === "profile_update") {
              try {
                onProfileUpdate?.(JSON.parse(data));
              } catch (e) {
                console.warn("Skipping unparseable 'profile_update' SSE payload:", data, e);
              }
            } else if (event === "triage_status") {
              try {
                patchAssistant({ triage: JSON.parse(data) });
              } catch (e) {
                console.warn("Skipping unparseable 'triage_status' SSE payload:", data, e);
              }
            } else if (event === "token") {
              setMessages((prev) => {
                const next = [...prev];
                const idx = next.length - 1;
                next[idx] = { ...next[idx], content: next[idx].content + data };
                return next;
              });
            } else if (event === "done") {
              try {
                patchAssistant({ audit: JSON.parse(data) });
              } catch (e) {
                console.warn("Skipping unparseable 'done' SSE payload:", data, e);
              }
            } else if (event === "error") {
              setError(data);
              showToast(data || "The clinical assistant ran into an error mid-response.", "error");
            }
          },
          { signal: controller.signal }
        );
      } catch (err) {
        if (err.name !== "AbortError") {
          const msg = err.message || "Connection to the clinical API failed.";
          setError(msg);
          showToast(msg, "error");
        }
      } finally {
        setIsStreaming(false);
      }
    },
    [messages, conversationId, onConversationId, onProfileUpdate, showToast]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const reset = useCallback((initialMessages = []) => {
    setMessages(initialMessages);
    setError(null);
  }, []);

  return { messages, isStreaming, error, send, stop, reset };
}
