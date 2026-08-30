import { useEffect, useRef, useState } from "react";
import { Send, Square } from "lucide-react";
import { cn } from "../../lib/cn";

export default function ChatInput({ onSend, onStop, disabled, isStreaming }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex items-end gap-2 rounded-xl border border-slate-border bg-white p-2 shadow-card">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        placeholder="Describe symptoms or a clinical case... (Enter to send, Shift+Enter for a new line)"
        className="max-h-[200px] flex-1 resize-none bg-transparent px-2 py-2 text-sm text-clinical-ink outline-none placeholder:text-clinical-muted"
      />
      {isStreaming ? (
        <button
          type="button"
          onClick={onStop}
          className="flex shrink-0 items-center gap-1.5 rounded-lg border border-slate-border bg-white px-4 py-2 text-sm font-semibold text-slate-600 transition hover:bg-slate-bg"
        >
          <Square size={14} />
          Stop
        </button>
      ) : (
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          className={cn(
            "flex shrink-0 items-center gap-1.5 rounded-lg bg-clinical-teal px-4 py-2 text-sm font-semibold text-white transition hover:bg-clinical-teal-dark",
            (disabled || !value.trim()) && "cursor-not-allowed opacity-50 hover:bg-clinical-teal"
          )}
        >
          <Send size={14} />
          Send
        </button>
      )}
    </div>
  );
}
