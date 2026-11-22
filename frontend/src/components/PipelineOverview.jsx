import { useState } from "react";
import { ChevronDown, Workflow } from "lucide-react";
import { cn } from "../lib/cn";
import { PHASES, STAGES } from "../lib/defense";

const PHASE_TONE = {
  request: "border-clinical-teal/40 bg-clinical-teal-tint/60",
  answer: "border-slate-border bg-slate-bg",
  check: "border-amber-300/60 bg-amber-50/60 dark:bg-amber-500/10",
};

/**
 * The MedLM gateway at a glance: nine numbered stages in three phases, the same
 * names the detector log, pipeline log, settings and paper use.
 */
export default function PipelineOverview({ defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-xl border border-slate-border bg-surface p-4 shadow-card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start justify-between gap-2 text-left"
        aria-expanded={open}
      >
        <div>
          <h2 className="flex items-center gap-2 text-[15px] font-semibold text-clinical-ink">
            <Workflow size={16} className="text-clinical-teal" />
            How the MedLM gateway works
          </h2>
          <p className="mt-0.5 text-xs text-clinical-muted">
            The <b className="font-semibold">gateway</b> is all nine stages together, in this order. Phase 1 can stop a
            message before the model sees it; phase 3 can withhold an answer before the user sees it.
          </p>
        </div>
        <ChevronDown size={16} className={cn("mt-1 shrink-0 text-clinical-muted transition", open && "rotate-180")} />
      </button>

      {open && (
        <>
          <div className="mt-3 grid gap-3 lg:grid-cols-[1.3fr_1fr_0.8fr]">
            {PHASES.map((p, i) => (
              <div key={p.key} className={cn("rounded-lg border p-3", PHASE_TONE[p.key])}>
                <div className="text-[11px] font-semibold uppercase tracking-wide text-clinical-muted">
                  Phase {i + 1} · {p.label}
                </div>
                <div className="text-[11px] text-clinical-muted">{p.hint}</div>
                <ol className="mt-2 space-y-1.5">
                  {STAGES.filter((s) => s.phase === p.key).map((s) => (
                    <li key={s.key} className="flex gap-2 text-xs">
                      <span
                        className={cn(
                          "mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold",
                          s.n ? "bg-clinical-teal text-white" : "border border-dashed border-clinical-muted text-clinical-muted"
                        )}
                      >
                        {s.n ?? "+"}
                      </span>
                      <span className="min-w-0">
                        <span className="font-semibold text-clinical-ink">{s.name}</span>
                        <span className="block text-clinical-muted">{s.plain}</span>
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
          <dl className="mt-3 grid gap-x-4 gap-y-1 text-[11px] text-clinical-muted sm:grid-cols-2">
            <div><dt className="inline font-semibold text-clinical-ink">Gateway</dt> <dd className="inline">— the whole pipeline above, placed in front of any medical model.</dd></div>
            <div><dt className="inline font-semibold text-clinical-ink">Guardrails</dt> <dd className="inline">— only stage 3, the fixed drug-safety rules.</dd></div>
            <div><dt className="inline font-semibold text-clinical-ink">Llama Guard</dt> <dd className="inline">— one safety model, used twice: on the message (stage 4, Layer G) and on the answer (stage 9).</dd></div>
            <div><dt className="inline font-semibold text-clinical-ink">Undefended / defended</dt> <dd className="inline">— the model called directly, or through the gateway.</dd></div>
          </dl>
        </>
      )}
    </div>
  );
}
