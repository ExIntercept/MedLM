import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ChevronDown, Eye, Loader2, Search } from "lucide-react";
import { cn } from "../../lib/cn";
import { getAdminConversation, getAdminUser, getAdminUsers } from "../../lib/api";
import { Markdown } from "../../lib/markdown";
import { PROFILE_FIELDS } from "../../lib/profile";
import { RequestDetail } from "../Security/PipelineLog";
import FlowMap from "./FlowMap";
import { Card } from "./AdminConsole";

const ROLE_TONE = {
  admin: "bg-violet-50 text-violet-700 border-violet-200",
  clinician: "bg-sky-50 text-sky-700 border-sky-200",
  patient: "bg-slate-bg text-clinical-muted border-slate-border",
};

const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

function UserList({ onOpen }) {
  const [users, setUsers] = useState(null);
  const [error, setError] = useState(null);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("last_active");

  useEffect(() => {
    getAdminUsers()
      .then((d) => setUsers(d.users || []))
      .catch((e) => setError(e.message));
  }, []);

  const rows = useMemo(() => {
    const list = (users || []).filter((u) => u.username.toLowerCase().includes(q.toLowerCase()));
    return [...list].sort((a, b) =>
      sort === "stopped" ? b.stopped - a.stopped : String(b.last_active || "").localeCompare(String(a.last_active || ""))
    );
  }, [users, q, sort]);

  return (
    <Card
      title="Users"
      action={
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 rounded-lg border border-slate-border bg-slate-bg px-2 py-1 text-xs">
            <Search size={12} className="text-clinical-muted" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find user" className="w-28 bg-transparent text-clinical-ink outline-none" />
          </label>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value)}
            className="rounded-lg border border-slate-border bg-slate-bg px-2 py-1 text-xs text-clinical-ink"
          >
            <option value="last_active">Most recent</option>
            <option value="stopped">Most stopped</option>
          </select>
        </div>
      }
    >
      <p className="mb-3 text-xs text-clinical-muted">
        Activity only. Opening a user shows their record and conversations, and is recorded in the access log.
      </p>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {!users && !error && <Loader2 size={18} className="animate-spin text-clinical-teal" />}
      {users && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-clinical-muted">
              <tr className="border-b border-slate-border">
                <th className="py-1.5 pr-3 font-semibold">User</th>
                <th className="py-1.5 pr-3 font-semibold">Role</th>
                <th className="py-1.5 pr-3 font-semibold">Last active</th>
                <th className="py-1.5 pr-3 text-right font-semibold">Consultations</th>
                <th className="py-1.5 pr-3 text-right font-semibold">Requests</th>
                <th className="py-1.5 pr-3 text-right font-semibold">Stopped</th>
                <th className="py-1.5 pr-3 text-right font-semibold">Flagged</th>
                <th className="py-1.5 pr-3 text-right font-semibold">Sanitized</th>
                <th className="py-1.5" />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-border">
              {rows.map((u) => {
                const suspicious = u.requests >= 3 && u.stopped / u.requests >= 0.5;
                return (
                  <tr key={u.id} className="hover:bg-slate-bg">
                    <td className="py-2 pr-3 font-semibold text-clinical-ink">
                      {u.username}
                      {suspicious && (
                        <span className="ml-2 rounded-full border border-red-200 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700">
                          mostly blocked
                        </span>
                      )}
                    </td>
                    <td className="py-2 pr-3">
                      <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-semibold", ROLE_TONE[u.role])}>{u.role}</span>
                    </td>
                    <td className="whitespace-nowrap py-2 pr-3 text-clinical-muted">{when(u.last_active)}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{u.consultations}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{u.requests}</td>
                    <td className={cn("py-2 pr-3 text-right tabular-nums", u.stopped && "font-semibold text-red-600")}>{u.stopped}</td>
                    <td className={cn("py-2 pr-3 text-right tabular-nums", u.flagged && "text-amber-600")}>{u.flagged}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{u.sanitized}</td>
                    <td className="py-2 text-right">
                      <button
                        type="button"
                        onClick={() => onOpen(u.id)}
                        className="inline-flex items-center gap-1 rounded-lg border border-slate-border px-2 py-1 font-semibold text-slate-600 hover:bg-surface"
                      >
                        <Eye size={12} /> Open
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

/** Pair each assistant reply with the pipeline run that produced it. */
function runFor(message, runs) {
  const head = (message.content || "").slice(0, 120);
  return runs.find((r) => (r.response || "").slice(0, 120) === head);
}

function Conversation({ id }) {
  const [conv, setConv] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(null);

  useEffect(() => {
    setConv(null);
    getAdminConversation(id)
      .then(setConv)
      .catch((e) => setError(e.message));
  }, [id]);

  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (!conv) return <Loader2 size={18} className="animate-spin text-clinical-teal" />;

  const runs = conv.runs || [];
  return (
    <div className="space-y-4">
      {runs.length > 0 && (
        <div className="rounded-lg border border-slate-border p-2">
          <div className="mb-1 px-1 text-xs font-semibold text-clinical-ink">This conversation through the pipeline · {runs.length} turns</div>
          <FlowMap logs={runs} />
        </div>
      )}
      <div className="space-y-3">
        {conv.messages.map((m, i) => {
          const run = m.role === "assistant" ? runFor(m, runs) : null;
          const key = `${i}`;
          return (
            <div key={key} className={cn("rounded-lg border px-3 py-2", m.role === "user" ? "border-slate-border bg-slate-bg" : "border-slate-border bg-surface")}>
              <div className="mb-1 flex items-center justify-between gap-2 text-[11px] text-clinical-muted">
                <span className="font-semibold uppercase tracking-wide">{m.role === "user" ? conv.owner : "Assistant"}</span>
                <span>{when(m.created_at)}</span>
              </div>
              {m.role === "user" ? (
                <p className="whitespace-pre-wrap text-sm text-clinical-ink">{m.content}</p>
              ) : (
                <Markdown text={m.content} className="text-sm" />
              )}
              {m.role === "assistant" && (
                <div className="mt-2">
                  {run ? (
                    <>
                      <button
                        type="button"
                        onClick={() => setOpen(open === key ? null : key)}
                        className="flex items-center gap-1 text-xs font-semibold text-clinical-teal-dark hover:underline"
                      >
                        <ChevronDown size={13} className={cn("transition", open === key && "rotate-180")} />
                        How the pipeline handled this ({run.status}, {Math.round(run.total_ms)} ms)
                      </button>
                      {open === key && (
                        <div className="mt-2 rounded-lg border border-slate-border bg-slate-bg/50 p-3">
                          <RequestDetail rec={run} />
                        </div>
                      )}
                    </>
                  ) : (
                    <span className="text-[11px] text-clinical-muted">
                      {m.meta?.status ? `Status ${m.meta.status}` : "No pipeline record (sent before logging was added)"}
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function UserDetail({ id, onBack }) {
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);
  const [convId, setConvId] = useState(null);

  useEffect(() => {
    getAdminUser(id)
      .then((u) => {
        setUser(u);
        setConvId(u.conversations[0]?.id ?? null);
      })
      .catch((e) => setError(e.message));
  }, [id]);

  return (
    <div className="flex flex-col gap-4">
      <button type="button" onClick={onBack} className="flex w-fit items-center gap-1 text-sm font-semibold text-clinical-teal-dark hover:underline">
        <ArrowLeft size={14} /> All users
      </button>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {!user && !error && <Loader2 size={18} className="animate-spin text-clinical-teal" />}
      {user && (
        <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
          <div className="flex flex-col gap-4">
            <Card title={user.username}>
              <div className="space-y-1 text-xs">
                <div><span className="text-clinical-muted">Role:</span> <span className="font-semibold">{user.role}</span></div>
                <div><span className="text-clinical-muted">Joined:</span> {when(user.created_at)}</div>
              </div>
              <h3 className="mb-1 mt-3 text-[11px] font-semibold uppercase tracking-wide text-clinical-muted">Health record</h3>
              <dl className="space-y-1 text-xs">
                {PROFILE_FIELDS.filter((f) => !f.visitOnly).map((f) => (
                  <div key={f.key}>
                    <dt className="inline text-clinical-muted">{f.label}: </dt>
                    <dd className="inline text-clinical-ink">{user.health_record?.[f.key] || "—"}</dd>
                  </div>
                ))}
              </dl>
            </Card>
            <Card title={`Consultations (${user.conversations.length})`}>
              {user.conversations.length === 0 && <p className="text-xs text-clinical-muted">None.</p>}
              <ul className="max-h-[50vh] space-y-1 overflow-auto">
                {user.conversations.map((c) => (
                  <li key={c.id}>
                    <button
                      type="button"
                      onClick={() => setConvId(c.id)}
                      className={cn(
                        "w-full rounded-lg px-2 py-1.5 text-left text-xs",
                        convId === c.id ? "bg-clinical-teal-tint text-clinical-teal-dark" : "hover:bg-slate-bg"
                      )}
                    >
                      <div className="truncate font-semibold">{c.title || `Consultation #${c.id}`}</div>
                      <div className="text-[11px] text-clinical-muted">{when(c.created_at)} · {c.messages} messages</div>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
          <Card title="Conversation">{convId ? <Conversation id={convId} /> : <p className="text-sm text-clinical-muted">Pick a consultation.</p>}</Card>
        </div>
      )}
    </div>
  );
}

export default function Users() {
  const [userId, setUserId] = useState(null);
  return userId ? <UserDetail id={userId} onBack={() => setUserId(null)} /> : <UserList onOpen={setUserId} />;
}
