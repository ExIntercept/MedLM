// Human-readable names for the gateway's stages, detector reasons and
// Llama Guard hazard categories (src/security/*).

export const STAGES = [
  { key: "sanitize", label: "Sanitize" },
  { key: "triage", label: "Triage" },
  { key: "guardrails", label: "Guardrails" },
  { key: "detect", label: "Detect" },
  { key: "clarify", label: "Clarify", optional: true }, // patient chat only
  { key: "retrieve", label: "Retrieve" },
  { key: "harden", label: "Harden" },
  { key: "generate", label: "Generate" },
  { key: "verify", label: "Verify" },
  { key: "output_guard", label: "Output guard" },
];

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

export const TOGGLES = [
  { key: "defense_enabled", label: "Attack detection", hint: "Master switch for Layers A, B, H and G" },
  { key: "layer_sanitize", label: "Input sanitization", hint: "Invisible characters and model control tokens" },
  { key: "layer_triage", label: "Emergency triage", hint: "Escalates acute emergencies without calling the model" },
  { key: "layer_guardrails", label: "Contraindication guardrails", hint: "Hard clinical safety rules" },
  { key: "layer_prompt_guard", label: "Layer A · Injection classifier", hint: "DeBERTa prompt-injection model" },
  { key: "layer_template_matcher", label: "Layer B · Jailbreak templates", hint: "Similarity to known jailbreaks" },
  { key: "layer_harm_classifier", label: "Layer H · Harmful clinical intent", hint: "Plainly worded harmful requests" },
  { key: "layer_llama_guard", label: "Layer G · Llama Guard (input)", hint: "Hazard taxonomy on the user's message" },
  { key: "layer_prompt_hardening", label: "Prompt hardening", hint: "Nonce-tagged delimiters and restated rules" },
  { key: "layer_output_guard", label: "Output guard", hint: "Llama Guard on the model's answer" },
];

/** What each stage does, for the admin pipeline view. */
export const STAGE_INFO = {
  sanitize: "Strips invisible characters and model control tokens from the message and profile",
  triage: "Escalates acute emergencies without calling the model",
  guardrails: "Hard contraindication rules for dangerous drug requests",
  detect: "Layers A (injection classifier), B (jailbreak templates), H (harmful intent), G (Llama Guard)",
  retrieve: "Hybrid search over the medical guideline corpus",
  harden: "Nonce-tagged prompt with restated rules; extra rules when the request was flagged",
  generate: "The medical model writes an answer",
  verify: "Checks each claim against the retrieved evidence; withholds unsupported answers",
  output_guard: "Llama Guard judges the model's answer before it is shown",
};

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
