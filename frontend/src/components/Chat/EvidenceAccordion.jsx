import { useState } from "react";
import { BookMarked, ChevronDown } from "lucide-react";
import { cn } from "../../lib/cn";

export default function EvidenceAccordion({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 overflow-hidden rounded-lg border border-slate-border">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between bg-slate-bg px-3 py-2 text-left text-sm font-semibold text-clinical-ink"
      >
        <span className="flex items-center gap-2">
          <BookMarked size={15} className="text-clinical-teal" />
          Clinical Evidence &amp; Guideline Sources ({sources.length})
        </span>
        <ChevronDown size={16} className={cn("transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ul className="divide-y divide-slate-border bg-white">
          {sources.map((src, i) => (
            <li key={`${src.chunk_id}-${i}`} className="px-3 py-2.5 text-sm">
              <div className="font-medium text-clinical-ink">{src.title}</div>
              <div className="mt-0.5 text-xs text-clinical-muted">
                {src.source || "Unknown Source"} · chunk {src.chunk_id}
                {typeof src.score === "number" && ` · rerank score ${src.score.toFixed(3)}`}
              </div>
              {src.excerpt && (
                <p className="mt-1.5 border-l-2 border-clinical-teal pl-2 text-xs text-slate-600">
                  {src.excerpt}
                  {src.excerpt.length >= 280 ? "…" : ""}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
