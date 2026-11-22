/** Tabs each role sees. The server also enforces the split for clinician mode, evidence search and the security console. */
export const TABS = [
  { key: "consultation", label: "Consultation", roles: ["patient", "clinician", "admin"] },
  { key: "records", label: "Records & History", roles: ["patient", "clinician", "admin"] },
  { key: "prescriptions", label: "Medication Check", roles: ["patient", "clinician", "admin"] },
  { key: "evidence", label: "Evidence Explorer", roles: ["clinician", "admin"] },
  { key: "evaluation", label: "Model Evaluation", roles: ["clinician", "admin"] },
  { key: "security", label: "Admin Console", roles: ["admin"] },
];

export function tabsFor(role) {
  return TABS.filter((t) => t.roles.includes(role || "patient"));
}
