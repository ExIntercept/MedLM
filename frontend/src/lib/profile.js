// Fields of the intake form / health record, shared by the sidebar and Records tab.
export const PROFILE_FIELDS = [
  { key: "age", label: "Age", placeholder: "e.g. 34" },
  { key: "sex", label: "Biological sex", placeholder: "e.g. Female" },
  { key: "duration", label: "Symptom duration", placeholder: "e.g. 2 weeks", visitOnly: true },
  { key: "conditions", label: "Conditions", placeholder: "e.g. Type 2 diabetes, asthma" },
  { key: "medications", label: "Current medications", placeholder: "e.g. Metformin 500mg" },
  { key: "past_medications", label: "Previous medications", placeholder: "e.g. Atorvastatin" },
  { key: "allergies", label: "Allergies", placeholder: "e.g. Penicillin" },
];

export const EMPTY_PROFILE = Object.fromEntries(PROFILE_FIELDS.map((f) => [f.key, ""]));

/** Fill every field, so inputs stay controlled. */
export function normalizeProfile(p) {
  return { ...EMPTY_PROFILE, ...Object.fromEntries(Object.entries(p || {}).map(([k, v]) => [k, v ?? ""])) };
}
