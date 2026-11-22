import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { X } from "lucide-react";
import { cn } from "../../lib/cn";

/**
 * First-run walkthrough. Each step spotlights an element marked with
 * data-tour="<target>" and explains it in a small card beside it; steps
 * without a target (or whose element is not on screen) show as a centred card.
 */

const STEPS = {
  patient: [
    { title: "Welcome to MedLM", body: "Ask health questions and get answers that are checked against medical guidelines. Here's a quick look around. It takes under a minute." },
    { target: "chat-input", title: "Ask anything", body: "Describe how you feel or ask a health question. If you mention a new symptom, I'll ask two or three quick questions first so the advice fits you. You can always skip them." },
    { target: "starters", title: "Not sure where to start?", body: "Tap one of these examples to try it out." },
    { target: "details", title: "Your details", body: "Your age, conditions, medicines and allergies are picked up from what you write and saved here, so you don't have to repeat them. Tap Edit to correct anything." },
    { target: "new-consultation", title: "Start fresh any time", body: "Begin a new consultation here. Your recent ones are listed underneath." },
    { target: "tab-records", title: "Records & History", body: "Your saved health record and every past consultation, to read back whenever you need." },
    { target: "tab-prescriptions", title: "Medication Check", body: "Checks the medicines in your details against known dangerous combinations with your conditions." },
    { target: "account", title: "Theme and account", body: "Switch between light and dark, sign out, or replay this tour from your account menu." },
    { title: "You're all set", body: "Answers are checked against guidelines, but MedLM is not a replacement for a doctor. In an emergency, call your local emergency number." },
  ],
  clinician: [
    { title: "Welcome to MedLM", body: "A clinical assistant that answers from retrieved guidelines and screens every request for attacks before it reaches the model. Here's a quick look around." },
    { target: "chat-input", title: "Ask or describe a case", body: "Ask a clinical question or describe a case. Each answer lists the guideline passages it used and how well it is supported by them." },
    { target: "answer-style", title: "Answer style", body: "Plain language for patient-friendly wording, Clinical for dosing, differentials and terminology." },
    { target: "details", title: "Case details", body: "Age, sex, conditions, medicines and allergies picked up from the case you describe. Edit them to steer the answer." },
    { target: "tab-evidence", title: "Evidence Explorer", body: "Search the guideline library directly and see the passages ranked as the assistant sees them." },
    { target: "tab-prescriptions", title: "Medication Check", body: "Screens the case's medicines against hard contraindication rules." },
    { target: "tab-evaluation", title: "Model Evaluation", body: "Benchmark results: attack success and answer quality for each model, with and without the defenses." },
    { target: "tab-records", title: "Records & History", body: "Past consultations, read-only." },
    { target: "account", title: "Theme and account", body: "Switch between light and dark, sign out, or replay this tour from your account menu." },
    { title: "You're all set", body: "Tip: open the badge under any answer to see which pipeline stages ran." },
  ],
};
STEPS.admin = [
  ...STEPS.clinician.slice(0, -2),
  { target: "status", title: "System status", body: "All services at a glance: the medical model, evidence search and the two Llama Guard checks. Click for details." },
  { target: "tab-security", title: "Admin Console", body: "Watch every request move through the pipeline live, see which stage stopped an attack, switch the medical model, turn gateway stages on or off and review users. Opening a user's conversation is recorded in the access log." },
  STEPS.clinician[STEPS.clinician.length - 2],
  { title: "You're all set", body: "Tip: in Admin Console, click any request in Live pipeline to see its map and stage-by-stage timeline." },
];

export function tourSteps(role) {
  return STEPS[role] || STEPS.patient;
}

const KEY = (username) => `medlm.tour.v1.${username || "anon"}`;

export function tourSeen(username) {
  try {
    return localStorage.getItem(KEY(username)) === "done";
  } catch {
    return true; // storage unavailable: don't nag on every load
  }
}

export function markTourSeen(username) {
  try {
    localStorage.setItem(KEY(username), "done");
  } catch {
    /* storage unavailable */
  }
}

/** The first visible element for a target (tabs exist twice: wide and narrow layouts). */
function findTarget(name) {
  if (!name) return null;
  return [...document.querySelectorAll(`[data-tour="${name}"]`)].find((el) => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }) || null;
}

const PAD = 6;
const CARD_W = 320;
const GAP = 12;

export default function Tour({ role, onClose }) {
  const steps = tourSteps(role);
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState(null);
  const step = steps[index];
  const last = index === steps.length - 1;

  const measure = useCallback(() => {
    const el = findTarget(step.target);
    setRect(el ? el.getBoundingClientRect() : null);
  }, [step.target]);

  useLayoutEffect(() => {
    const el = findTarget(step.target);
    if (el) el.scrollIntoView({ block: "nearest", behavior: "instant" });
    measure();
  }, [measure, step.target]);

  useEffect(() => {
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [measure]);

  const next = useCallback(() => (last ? onClose() : setIndex((i) => i + 1)), [last, onClose]);
  const back = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      // Enter on a focused button already clicks it; only handle it elsewhere.
      else if (e.key === "ArrowRight" || (e.key === "Enter" && e.target.tagName !== "BUTTON")) next();
      else if (e.key === "ArrowLeft") back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, back, onClose]);

  // Card placement: below the target if it fits, else above; centred otherwise.
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const cardW = Math.min(CARD_W, vw - 32);
  let cardStyle = { width: cardW, left: (vw - cardW) / 2, top: Math.max(16, vh / 2 - 120) };
  if (rect) {
    const left = Math.min(Math.max(16, rect.left + rect.width / 2 - cardW / 2), vw - cardW - 16);
    const below = rect.bottom + PAD + GAP;
    cardStyle = vh - below > 200
      ? { width: cardW, left, top: below }
      : { width: cardW, left, bottom: vh - rect.top + PAD + GAP };
  }

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-labelledby="tour-title">
      {rect ? (
        <div
          className="pointer-events-none fixed rounded-xl ring-2 ring-clinical-teal transition-all duration-200"
          style={{
            left: rect.left - PAD,
            top: rect.top - PAD,
            width: rect.width + PAD * 2,
            height: rect.height + PAD * 2,
            boxShadow: "0 0 0 9999px rgba(8, 14, 13, 0.55)",
          }}
        />
      ) : (
        <div className="fixed inset-0 bg-[rgba(8,14,13,0.55)]" />
      )}

      <div
        className="fixed rounded-2xl border border-slate-border bg-surface p-4 shadow-elevated transition-all duration-200"
        style={cardStyle}
      >
        <div className="flex items-start justify-between gap-3">
          <h2 id="tour-title" className="text-[15px] font-semibold text-clinical-ink">{step.title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the tour"
            className="-mr-1 -mt-1 rounded-md p-1 text-clinical-muted transition hover:bg-slate-bg hover:text-clinical-ink"
          >
            <X size={15} />
          </button>
        </div>
        <p className="mt-1.5 text-sm leading-relaxed text-clinical-muted">{step.body}</p>
        <div className="mt-4 flex items-center justify-between gap-2">
          <div className="flex gap-1" aria-label={`Step ${index + 1} of ${steps.length}`}>
            {steps.map((_, i) => (
              <span
                key={i}
                className={cn("h-1.5 rounded-full transition-all", i === index ? "w-4 bg-clinical-teal" : "w-1.5 bg-slate-border")}
              />
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            {index === 0 ? (
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-clinical-muted transition hover:bg-slate-bg hover:text-clinical-ink"
              >
                Skip
              </button>
            ) : (
              <button
                type="button"
                onClick={back}
                className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-clinical-muted transition hover:bg-slate-bg hover:text-clinical-ink"
              >
                Back
              </button>
            )}
            <button
              type="button"
              onClick={next}
              autoFocus
              className="whitespace-nowrap rounded-lg bg-clinical-teal px-3.5 py-1.5 text-sm font-semibold text-white transition hover:bg-clinical-teal-hover"
            >
              {index === 0 ? "Show me around" : last ? "Get started" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
