// Human-readable names for the gateway's stages, detector reasons and
// Llama Guard hazard categories (src/security/*).

// The one vocabulary for the pipeline, used by every screen and by the paper.
// "Gateway" means all nine stages together; each stage has a number, a short
// label (chips, maps, legends) and a full name (overview, settings).
export const PHASES = [
  { key: "request", label: "Check the request", hint: "Before the model is called; can stop the message" },
  { key: "answer", label: "Prepare and answer", hint: "Evidence is found and the protected model answers" },
  { key: "check", label: "Check the answer", hint: "Before the patient sees it; can withhold the answer" },
];

export const STAGES = [
  { key: "sanitize", n: 1, phase: "request", label: "Sanitize", name: "Input sanitization",
    plain: "Removes hidden characters and model control tokens" },
  { key: "triage", n: 2, phase: "request", label: "Triage", name: "Emergency triage",
    plain: "Sends emergencies to urgent-care advice instead of the model" },
  { key: "guardrails", n: 3, phase: "request", label: "Guardrails", name: "Contraindication guardrails",
    plain: "Hard rules for dangerous drug combinations" },
  { key: "detect", n: 4, phase: "request", label: "Detectors", name: "Attack detectors (A · B · H · G)",
    plain: "A injection, B jailbreak templates, H harmful clinical intent, G Llama Guard" },
  { key: "clarify", n: null, phase: "request", label: "Follow-up", name: "Follow-up questions",
    plain: "Patient chat only: asks a few questions before answering a new symptom", optional: true },
  { key: "retrieve", n: 5, phase: "answer", label: "Retrieve", name: "Evidence retrieval",
    plain: "Finds relevant passages in the medical guideline corpus" },
  { key: "harden", n: 6, phase: "answer", label: "Harden", name: "Prompt hardening",
    plain: "Wraps the message in tagged delimiters and restates the safety rules" },
  { key: "generate", n: 7, phase: "answer", label: "Generate", name: "Generation",
    plain: "The protected medical model writes the answer" },
  { key: "verify", n: 8, phase: "check", label: "Grounding check", name: "Evidence-grounding check",
    plain: "Withholds answers (or removes claims) the evidence does not support" },
  { key: "output_guard", n: 9, phase: "check", label: "Output guard", name: "Output guard",
    plain: "Llama Guard checks the answer itself for harmful content" },
];

export const STAGE_BY_KEY = Object.fromEntries(STAGES.map((s) => [s.key, s]));

const REASONS = {
  pg_score_block: "Injection classifier (Layer A)",
  pg_score_flag: "Injection classifier (Layer A)",
  tm_sim_block: "Jailbreak template matcher (Layer B)",
  tm_sim_flag: "Jailbreak template matcher (Layer B)",
  tm_components_block: "Jailbreak components (Layer B)",
  tm_components_flag: "Jailbreak components (Layer B)",
  harm_block: "Harmful-intent classifier (Layer H)",
  harm_flag: "Harmful-intent classifier (Layer H)",
  guard_block: "Llama Guard (Layer G)",
  control_token_detected: "Control tokens neutralised",
};

export const GUARD_CATEGORIES = {
  S1: "Violent crimes", S2: "Non-violent crimes", S3: "Sex-related crimes",
  S4: "Child sexual exploitation", S5: "Defamation", S6: "Specialized advice",
  S7: "Privacy", S8: "Intellectual property", S9: "Indiscriminate weapons",
  S10: "Hate", S11: "Suicide & self-harm", S12: "Sexual content",
  S13: "Elections", S14: "Code interpreter abuse",
};

export function reasonLabels(reasons = []) {
  return [...new Set(reasons.map((r) => REASONS[r] || r))];
}

export function guardCategoryLabels(categories = []) {
  return categories.map((c) => GUARD_CATEGORIES[c] || c);
}

// Switches in pipeline order; `stage` is the stage number they belong to.
export const TOGGLES = [
  { key: "layer_sanitize", stage: 1, label: "Input sanitization", hint: "Invisible characters and model control tokens" },
  { key: "layer_triage", stage: 2, label: "Emergency triage", hint: "Escalates acute emergencies without calling the model" },
  { key: "layer_guardrails", stage: 3, label: "Contraindication guardrails", hint: "Hard rules for dangerous drug combinations" },
  { key: "defense_enabled", stage: 4, label: "Attack detectors", hint: "Master switch for Layers A, B, H and G below" },
  { key: "layer_prompt_guard", stage: 4, sub: true, label: "Layer A · Injection classifier", hint: "DeBERTa prompt-injection model" },
  { key: "layer_template_matcher", stage: 4, sub: true, label: "Layer B · Jailbreak templates", hint: "Similarity to known jailbreaks" },
  { key: "layer_harm_classifier", stage: 4, sub: true, label: "Layer H · Harmful clinical intent", hint: "Plainly worded harmful requests" },
  { key: "layer_llama_guard", stage: 4, sub: true, label: "Layer G · Llama Guard", hint: "Hazard categories on the user's message" },
  { key: "layer_prompt_hardening", stage: 6, label: "Prompt hardening", hint: "Tagged delimiters and restated rules" },
  { key: "layer_output_guard", stage: 9, label: "Output guard", hint: "Llama Guard on the model's answer" },
];

/** What each stage does, for the admin pipeline view. */
export const STAGE_INFO = Object.fromEntries(STAGES.map((s) => [s.key, s.plain]));

/** Stage names used by the benchmark tables (scripts/paper_tables.py), mapped to pipeline stages. */
export const STOP_STAGES = [
  { key: "emergency triage", stage: "triage", color: "bg-teal-300" },
  { key: "guardrails", stage: "guardrails", color: "bg-teal-400" },
  { key: "input detectors", stage: "detect", color: "bg-clinical-teal" },
  { key: "grounding check", stage: "verify", color: "bg-amber-400" },
  { key: "output guard", stage: "output_guard", color: "bg-violet-500" },
  { key: "model refused", label: "Model refused", color: "bg-slate-400" },
  { key: "answered", label: "Answered", color: "bg-slate-200" },
].map((s) => {
  const st = s.stage && STAGE_BY_KEY[s.stage];
  return { ...s, label: st ? `${st.n} · ${st.label}` : s.label };
});

/** Colour tone for a stage outcome: ok | warn | stop | off. */
export function outcomeTone(outcome = "") {
  const o = outcome.toLowerCase();
  if (["block", "blocked", "intercepted", "emergency", "withheld"].includes(o)) return "stop";
  if (["flag", "modified", "case_discussion", "unsafe_allowed", "flagged", "ungrounded", "asked", "redacted"].includes(o)) return "warn";
  if (["off", "skipped"].includes(o) || o.startsWith("0 passages")) return "off";
  return "ok";
}

export const OUTCOME_LABEL = {
  clean: "clean", modified: "sanitized", clear: "clear", emergency: "escalated",
  case_discussion: "case discussion", intercepted: "intercepted", allow: "allowed",
  flag: "flagged", block: "blocked", hardened: "hardened", answered: "answered",
  pass: "grounded", flagged: "partly grounded", redacted: "unsupported lines removed", ungrounded: "ungrounded", withheld: "withheld",
  safe: "safe", unsafe_allowed: "unsafe (non-blocking)", blocked: "blocked", off: "off", skipped: "skipped",
  asked: "questions asked", not_needed: "not needed",
};
