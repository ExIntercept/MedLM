import { useState } from "react";
import { Loader2, Lock, ShieldCheck, User } from "lucide-react";
import { cn } from "../../lib/cn";
import { useAuth } from "../../context/AuthContext";
import { useToast } from "../../context/ToastContext";

const inputClass =
  "w-full rounded-lg border border-slate-border bg-slate-bg px-3 py-2 text-sm text-clinical-ink outline-none transition focus:border-clinical-teal focus:bg-white focus:ring-2 focus:ring-clinical-teal/20";

export default function LoginPage() {
  const { login, register } = useAuth();
  const { showToast } = useToast();
  const [mode, setMode] = useState("login"); // "login" | "signup"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError(null);
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(username.trim(), password);
      } else {
        await register(username.trim(), password, email.trim());
        showToast("Account created — welcome to MedIntake AI.", "success");
      }
    } catch (err) {
      setFormError(err.message || "Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-bg p-4">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-clinical-teal-tint text-clinical-teal-dark">
            <ShieldCheck size={24} />
          </div>
          <h1 className="text-xl font-bold text-clinical-ink">MedIntake AI</h1>
          <p className="mt-1 text-sm text-clinical-muted">Secure clinical decision support &amp; intake portal</p>
        </div>

        <div className="rounded-xl border border-slate-border bg-white p-6 shadow-elevated">
          <div className="mb-5 flex overflow-hidden rounded-lg border border-slate-border text-sm font-semibold">
            {[
              { key: "login", label: "Log In" },
              { key: "signup", label: "Create Account" },
            ].map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => {
                  setMode(tab.key);
                  setFormError(null);
                }}
                className={cn(
                  "flex-1 px-3 py-2 transition",
                  mode === tab.key ? "bg-clinical-teal text-white" : "bg-white text-clinical-muted hover:bg-slate-bg"
                )}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-3">
            <label className="block">
              <span className="mb-1 flex items-center gap-1.5 text-xs font-medium text-clinical-muted">
                <User size={13} />
                Username
              </span>
              <input
                type="text"
                required
                minLength={3}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className={inputClass}
              />
            </label>

            {mode === "signup" && (
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-clinical-muted">Email (optional)</span>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className={inputClass}
                />
              </label>
            )}

            <label className="block">
              <span className="mb-1 flex items-center gap-1.5 text-xs font-medium text-clinical-muted">
                <Lock size={13} />
                Password
              </span>
              <input
                type="password"
                required
                minLength={6}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={mode === "signup" ? "Choose a secure password (min 6 chars)" : "Enter your password"}
                className={inputClass}
              />
            </label>

            {formError && (
              <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{formError}</p>
            )}

            <button
              type="submit"
              disabled={submitting}
              className={cn(
                "flex w-full items-center justify-center gap-2 rounded-lg bg-clinical-teal px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-clinical-teal-dark",
                submitting && "cursor-not-allowed opacity-70"
              )}
            >
              {submitting && <Loader2 size={15} className="animate-spin" />}
              {mode === "login" ? "Log In to Clinical Workspace" : "Create Account"}
            </button>
          </form>
        </div>

        <p className="mt-4 text-center text-xs text-clinical-muted">
          Private, local-first clinical decision support. Your data stays on this deployment.
        </p>
      </div>
    </div>
  );
}
