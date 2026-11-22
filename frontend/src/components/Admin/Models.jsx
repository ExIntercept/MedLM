import { useEffect, useState } from "react";
import { CheckCircle2, Cpu, Loader2 } from "lucide-react";
import { cn } from "../../lib/cn";
import { getAdminModels, setAdminModel } from "../../lib/api";
import { Card } from "./AdminConsole";

export default function Models() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);
  const [notice, setNotice] = useState(null);

  const load = () =>
    getAdminModels()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(e.message));

  useEffect(() => {
    load();
  }, []);

  const choose = async (model) => {
    setBusy(model);
    setNotice(null);
    try {
      await setAdminModel(model);
      await load();
      setNotice("Switched. The next message from any user goes to this model, through the same defenses.");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Card
      title={
        <h2 className="flex items-center gap-2 text-[15px] font-semibold text-clinical-ink">
          <Cpu size={16} className="text-clinical-teal" /> Medical model
        </h2>
      }
      action={<span className="text-xs text-clinical-muted">The gateway and all nine stages stay the same; only the model behind them changes</span>}
    >
      {error && <p className="mb-3 rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700">{error}</p>}
      {notice && <p className="mb-3 rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs text-emerald-700">{notice}</p>}
      {!data && !error && <Loader2 size={18} className="animate-spin text-clinical-teal" />}
      {data && !data.server_reachable && (
        <p className="mb-3 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-xs text-amber-700">
          The model server is not reachable, so installed models can&apos;t be checked.
        </p>
      )}
      {data && (
        <div className="grid gap-3 md:grid-cols-2">
          {data.models.map((m) => (
            <div
              key={m.model}
              className={cn(
                "flex flex-col gap-2 rounded-lg border p-3",
                m.active ? "border-clinical-teal bg-clinical-teal-tint" : "border-slate-border bg-slate-bg"
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold text-clinical-ink">
                    {m.label}
                    {m.default && <span className="rounded bg-surface px-1.5 py-0.5 text-[10px] font-semibold text-clinical-muted">default</span>}
                  </div>
                  <div className="text-xs text-clinical-muted">{m.note}</div>
                </div>
                {m.active ? (
                  <span className="flex items-center gap-1 text-xs font-semibold text-clinical-teal-dark">
                    <CheckCircle2 size={14} /> In use
                  </span>
                ) : (
                  <button
                    type="button"
                    disabled={!m.installed || busy !== null}
                    onClick={() => choose(m.model)}
                    className="flex items-center gap-1.5 rounded-lg bg-clinical-teal px-3 py-1.5 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {busy === m.model && <Loader2 size={12} className="animate-spin" />} Use this model
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-clinical-muted">
                <span>Family: {m.family}</span>
                <span>Context: {m.context_window.toLocaleString()} tokens</span>
                <span>Called via: {m.provider === "completions" ? "raw completions + our template" : "chat API"}</span>
                <span className={m.installed ? "text-emerald-600" : "text-red-600"}>{m.installed ? "installed" : "not installed"}</span>
              </div>
            </div>
          ))}
        </div>
      )}
      <p className="mt-3 text-[11px] text-clinical-muted">
        The switch lasts until the server restarts (then the model in .env is used). Each answer in the pipeline log records which model wrote it.
      </p>
    </Card>
  );
}
