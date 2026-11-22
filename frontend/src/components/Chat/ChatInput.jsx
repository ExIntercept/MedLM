import { useEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";
import { cn } from "../../lib/cn";

export default function ChatInput({ onSend, onStop, disabled, isStreaming, placeholder = "Type your message…" }) {
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

  const canSend = !disabled && value.trim();
  return (
    <div>
      <div className="flex items-end gap-2 rounded-2xl border border-slate-border bg-surface p-2 shadow-card transition focus-within:border-clinical-teal focus-within:ring-2 focus-within:ring-clinical-teal/15">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder={placeholder}
          aria-label="Message"
          className="max-h-[200px] flex-1 resize-none bg-transparent px-2 py-2 text-[15px] text-clinical-ink outline-none placeholder:text-clinical-muted"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-border text-clinical-ink transition hover:bg-slate-bg"
          >
            <Square size={14} />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            aria-label="Send"
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-clinical-teal text-white transition hover:bg-clinical-teal-hover",
              !canSend && "cursor-not-allowed opacity-40 hover:bg-clinical-teal"
            )}
          >
            <ArrowUp size={18} />
          </button>
        )}
      </div>
      <p className="mt-1.5 hidden text-center text-[11px] text-clinical-muted sm:block">
        Enter to send · Shift + Enter for a new line
      </p>
    </div>
  );
}
