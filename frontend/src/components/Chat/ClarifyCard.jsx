import { useState } from "react";
import { ArrowRight } from "lucide-react";
import { cn } from "../../lib/cn";

// How each answer is phrased in the reply the patient sends.
const ANSWER_LABEL = {
  onset: "Started",
  severity: "Severity",
  warning: "Warning signs",
  background: "Conditions or medicines",
  age: "Age",
};

/**
 * The assistant's questions before it answers a new symptom complaint.
 * Choice questions are tap-able chips; others take a short typed answer.
 * Sends one message with every answer given, or "Just answer" to skip.
 */
export default function ClarifyCard({ questions, active, onAnswer }) {
  const [answers, setAnswers] = useState({});
  const set = (id, value) => setAnswers((prev) => ({ ...prev, [id]: value }));
  const given = questions.filter((q) => (answers[q.id] || "").trim());

  const submit = () => {
    if (!given.length) return;
    onAnswer(given.map((q) => `${ANSWER_LABEL[q.id] || q.text}: ${answers[q.id].trim()}`).join("; "));
  };

  return (
    <div className="mt-3 space-y-3 rounded-xl border border-slate-border bg-surface p-4">
      {questions.map((q, i) => (
        <div key={q.id}>
          <p className="text-sm font-medium text-clinical-ink">
            <span className="mr-1.5 text-clinical-muted">{i + 1}.</span>
            {q.text}
          </p>
          {q.kind === "choice" ? (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {q.options.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  disabled={!active}
                  onClick={() => set(q.id, answers[q.id] === opt ? "" : opt)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-sm transition disabled:cursor-default",
                    answers[q.id] === opt
                      ? "border-clinical-teal bg-clinical-teal text-white"
                      : "border-slate-border text-clinical-ink enabled:hover:border-clinical-teal enabled:hover:bg-clinical-teal-tint"
                  )}
                >
                  {opt}
                </button>
              ))}
            </div>
          ) : (
            <input
              type="text"
              disabled={!active}
              value={answers[q.id] || ""}
              onChange={(e) => set(q.id, e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder={q.id === "age" ? "e.g. 34" : "e.g. asthma, metformin, or none"}
              className="mt-2 w-full rounded-lg border border-slate-border bg-slate-bg px-3 py-1.5 text-sm text-clinical-ink outline-none transition focus:border-clinical-teal focus:bg-surface focus:ring-2 focus:ring-clinical-teal/20 disabled:opacity-60"
            />
          )}
        </div>
      ))}
      {active && (
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <button
            type="button"
            onClick={submit}
            disabled={!given.length}
            className="flex items-center gap-1.5 rounded-lg bg-clinical-teal px-3.5 py-1.5 text-sm font-semibold text-white transition hover:bg-clinical-teal-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send answers
            <ArrowRight size={14} />
          </button>
          <button
            type="button"
            onClick={() => onAnswer("Just answer")}
            className="rounded-lg px-3 py-1.5 text-sm font-medium text-clinical-muted transition hover:bg-slate-bg hover:text-clinical-ink"
          >
            Skip, just answer
          </button>
        </div>
      )}
    </div>
  );
}
