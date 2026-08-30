import { useEffect, useState } from "react";
import { Download, FileText, Loader2, TrendingUp, X } from "lucide-react";
import { cn } from "../lib/cn";
import { Markdown } from "../lib/markdown";
import { getEvaluationReport, getEvaluationSummary } from "../lib/api";

function pct(v) {
  return v === null || v === undefined ? "N/A" : `${(v * 100).toFixed(1)}%`;
}

function KpiCard({ label, value, sub, tone = "teal" }) {
  const toneClasses = {
    teal: "border-clinical-teal-tint bg-clinical-teal-tint text-clinical-teal-dark",
    slate: "border-slate-border bg-slate-bg text-clinical-ink",
  };
  return (
    <div className={cn("rounded-xl border p-4 shadow-card", toneClasses[tone] || toneClasses.slate)}>
      <div className="text-xs font-semibold uppercase tracking-wide text-clinical-muted">{label}</div>
      <div className="mt-1 text-2xl font-bold text-clinical-ink">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-clinical-muted">{sub}</div>}
    </div>
  );
}

function fmtBool(v) {
  if (v === null || v === undefined) return "—";
  return v ? "✅" : "❌";
}

function fmtScore(v) {
  return v === null || v === undefined ? "—" : v.toFixed(2);
}

function downloadMarkdown(markdown) {
  const blob = new Blob([markdown], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "BENCHMARK_REPORT.md";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function EvaluationDashboard() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [reportMarkdown, setReportMarkdown] = useState(null);
  const [reportOpen, setReportOpen] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    getEvaluationSummary()
      .then(setSummary)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const openReport = async () => {
    setReportOpen(true);
    if (reportMarkdown) return;
    setReportLoading(true);
    try {
      setReportMarkdown(await getEvaluationReport());
    } catch (err) {
      setReportMarkdown(`_Failed to load report: ${err.message}_`);
    } finally {
      setReportLoading(false);
    }
  };

  const handleExport = async () => {
    let markdown = reportMarkdown;
    if (!markdown) {
      try {
        markdown = await getEvaluationReport();
      } catch (err) {
        return;
      }
    }
    downloadMarkdown(markdown);
  };

  if (loading) {
    return (
      <div className="flex min-h-[40vh] flex-col items-center justify-center gap-2 rounded-xl border border-slate-border bg-white p-8 shadow-card text-clinical-muted">
        <Loader2 size={24} className="animate-spin text-clinical-teal" />
        <p className="text-sm">Loading evaluation results…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-800 shadow-card">
        <p className="font-semibold">No evaluation results available</p>
        <p className="mt-1">{error}</p>
        <p className="mt-2 text-xs text-amber-700">
          Run <code className="rounded bg-amber-100 px-1 py-0.5">python scripts/run_benchmark.py</code> from the
          project root, then reload this tab.
        </p>
      </div>
    );
  }

  const m = summary.metrics || {};
  const cases = summary.cases || [];

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-xl border border-slate-border bg-white p-4 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="flex items-center gap-2 text-[15px] font-semibold text-clinical-ink">
              <TrendingUp size={16} className="text-clinical-teal" />
              Model Evaluation &amp; Proof
            </h2>
            <p className="mt-0.5 text-xs text-clinical-muted">
              {summary.num_cases} test cases · model <code>{summary.model}</code> · generated{" "}
              {summary.generated_at ? new Date(summary.generated_at).toLocaleString() : "—"}
            </p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={openReport}
              className="flex items-center gap-1.5 rounded-lg border border-slate-border bg-white px-3 py-2 text-sm font-semibold text-slate-600 transition hover:bg-slate-bg"
            >
              <FileText size={14} />
              View BENCHMARK_REPORT.md
            </button>
            <button
              type="button"
              onClick={handleExport}
              className="flex items-center gap-1.5 rounded-lg bg-clinical-teal px-3 py-2 text-sm font-semibold text-white transition hover:bg-clinical-teal-dark"
            >
              <Download size={14} />
              Export
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
        <KpiCard label="Retrieval Hit Rate @5" value={pct(m.retrieval_hit_rate_at_5)} sub={`@3: ${pct(m.retrieval_hit_rate_at_3)}`} />
        <KpiCard
          label="Hybrid RAG Faithfulness"
          value={pct(m.faithfulness_rag_avg)}
          sub={`vs ${pct(m.faithfulness_baseline_avg)} Baseline`}
        />
        <KpiCard label="Contraindication Catch Rate" value={pct(m.contraindication_catch_rate)} />
        <KpiCard label="Emergency Escalation Accuracy" value={pct(m.emergency_escalation_accuracy)} />
        <KpiCard label="Key-Phrase Grounding" value={pct(m.key_phrase_coverage_avg)} />
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-border bg-white shadow-card">
        <div className="border-b border-slate-border px-4 py-3">
          <h3 className="text-[15px] font-semibold text-clinical-ink">Baseline vs. Hybrid RAG — Per-Case Breakdown</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-border bg-slate-bg text-xs font-semibold uppercase tracking-wide text-clinical-muted">
                <th className="px-3 py-2">Case ID</th>
                <th className="px-3 py-2">Cat</th>
                <th className="px-3 py-2">Emergency</th>
                <th className="px-3 py-2">Contraindication</th>
                <th className="px-3 py-2">Hit@3</th>
                <th className="px-3 py-2">Hit@5</th>
                <th className="px-3 py-2">Groundedness (RAG)</th>
                <th className="px-3 py-2">Groundedness (Baseline)</th>
                <th className="px-3 py-2">Key-Phrase Coverage</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => (
                <tr key={c.case_id} className="border-b border-slate-border last:border-0 hover:bg-slate-bg">
                  <td className="px-3 py-2 font-mono text-xs text-clinical-ink">{c.case_id}</td>
                  <td className="px-3 py-2 text-clinical-muted">{c.category}</td>
                  <td className="px-3 py-2">{fmtBool(c.emergency_detected)}</td>
                  <td className="px-3 py-2">{fmtBool(c.contraindication_detected)}</td>
                  <td className="px-3 py-2">{fmtBool(c.retrieval_hit_at_3)}</td>
                  <td className="px-3 py-2">{fmtBool(c.retrieval_hit_at_5)}</td>
                  <td className="px-3 py-2">{fmtScore(c.groundedness_rag)}</td>
                  <td className="px-3 py-2">{fmtScore(c.groundedness_baseline)}</td>
                  <td className="px-3 py-2">{pct(c.key_phrase_coverage)}</td>
                </tr>
              ))}
              {cases.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-3 py-6 text-center text-sm text-clinical-muted">
                    No per-case data in this benchmark run.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {reportOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => setReportOpen(false)}>
          <div
            className="max-h-[85vh] w-full max-w-3xl overflow-hidden rounded-xl border border-slate-border bg-white shadow-elevated"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-border px-4 py-3">
              <h3 className="text-[15px] font-semibold text-clinical-ink">BENCHMARK_REPORT.md</h3>
              <button type="button" onClick={() => setReportOpen(false)} className="text-clinical-muted hover:text-clinical-ink">
                <X size={18} />
              </button>
            </div>
            <div className="max-h-[calc(85vh-56px)] overflow-y-auto p-4">
              {reportLoading ? (
                <p className="text-sm text-clinical-muted">Loading report…</p>
              ) : (
                <Markdown text={reportMarkdown} className="text-sm" />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
