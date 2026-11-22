import { useEffect, useRef, useState } from "react";
import { ArrowUp, ImagePlus, Square, X } from "lucide-react";
import { cn } from "../../lib/cn";

export default function ChatInput({ onSend, onStop, disabled, isStreaming, placeholder = "Type your message…" }) {
  const [value, setValue] = useState("");
  const [image, setImage] = useState(null); // { dataUrl, name }
  const [imageError, setImageError] = useState("");
  const textareaRef = useRef(null);
  const fileRef = useRef(null);

  const pickImage = (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      setImageError("Use a PNG, JPEG or WebP image.");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setImageError("The image must be 5 MB or smaller.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setImage({ dataUrl: reader.result, name: file.name });
      setImageError("");
    };
    reader.readAsDataURL(file);
  };

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const trimmed = value.trim() || (image ? "What does this image show?" : "");
    if (!trimmed || disabled) return;
    onSend(trimmed, image?.dataUrl);
    setValue("");
    setImage(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const canSend = !disabled && (value.trim() || image);
  return (
    <div>
      {image && (
        <div className="mb-2 flex items-center gap-2 rounded-xl border border-slate-border bg-surface p-2">
          <img src={image.dataUrl} alt="" className="h-12 w-12 rounded-md object-cover" />
          <span className="min-w-0 flex-1 truncate text-xs text-clinical-muted">
            {image.name} · read by the vision model and checked before the assistant sees it
          </span>
          <button type="button" onClick={() => setImage(null)} aria-label="Remove image"
            className="rounded-md p-1 text-clinical-muted hover:bg-slate-bg hover:text-clinical-ink">
            <X size={14} />
          </button>
        </div>
      )}
      {imageError && <p className="mb-1 text-xs text-red-600">{imageError}</p>}
      <div className="flex items-end gap-2 rounded-2xl border border-slate-border bg-surface p-2 shadow-card transition focus-within:border-clinical-teal focus-within:ring-2 focus-within:ring-clinical-teal/15">
        <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp" className="hidden" onChange={pickImage} />
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
          aria-label="Attach an image"
          title="Attach an image (X-ray, scan, photo)"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-clinical-muted transition hover:bg-slate-bg hover:text-clinical-ink disabled:opacity-40"
        >
          <ImagePlus size={18} />
        </button>
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
