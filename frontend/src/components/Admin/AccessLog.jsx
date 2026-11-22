import { useEffect, useState } from "react";
import { getAccessLog } from "../../lib/api";
import { shortModel } from "./FlowMap";
import { Card } from "./AdminConsole";

const ACTION = {
  view_user: "Viewed user's record and consultations",
  view_conversation: "Opened a conversation",
  switch_model: "Switched the medical model",
};

export default function AccessLog() {
  const [logs, setLogs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAccessLog()
      .then((d) => setLogs(d.logs || []))
      .catch((e) => setError(e.message));
  }, []);

  return (
    <Card title="Access log" action={<span className="text-xs text-clinical-muted">Every time an admin looks at a user&apos;s data or changes the model</span>}>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {logs && logs.length === 0 && <p className="text-sm text-clinical-muted">No admin access recorded yet.</p>}
      {logs && logs.length > 0 && (
        <div className="max-h-[70vh] overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-surface text-clinical-muted">
              <tr className="border-b border-slate-border">
                <th className="py-1.5 pr-3 font-semibold">Time</th>
                <th className="py-1.5 pr-3 font-semibold">Admin</th>
                <th className="py-1.5 pr-3 font-semibold">Action</th>
                <th className="py-1.5 pr-3 font-semibold">User</th>
                <th className="py-1.5 font-semibold">Detail</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-border">
              {logs.map((r, i) => (
                <tr key={i}>
                  <td className="whitespace-nowrap py-1.5 pr-3 text-clinical-muted">{new Date(r.timestamp).toLocaleString()}</td>
                  <td className="py-1.5 pr-3 text-clinical-ink">{r.admin}</td>
                  <td className="py-1.5 pr-3 text-clinical-ink">{ACTION[r.action] || r.action}</td>
                  <td className="py-1.5 pr-3 text-clinical-ink">{r.subject_user || "—"}</td>
                  <td className="py-1.5 text-clinical-muted">
                    {r.conversation_id != null && `conversation #${r.conversation_id}`}
                    {r.model && `${shortModel(r.previous || "")} → ${shortModel(r.model)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
