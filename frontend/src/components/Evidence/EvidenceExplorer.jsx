import { useState } from "react";
import { ChevronDown, Info, Search } from "lucide-react";
import { cn } from "../../lib/cn";
import { searchEvidence } from "../../lib/api";

const SOURCE_BADGES = {
  StatPearls: { emoji: "📚", label: "StatPearls" },
  OpenFDA: { emoji: "🏛️", label: "OpenFDA" },
  MedQuAD: { emoji: "📋", label: "MedQuAD" },
};

function sourceBadge(source) {
  return SOURCE_BADGES[source] || { emoji: "📄", label: source || "Guideline" };
}

/** Normalizes StatPearls-style " -- Section" separators and collapses any
 * excessive hyphen runs or exact repeated segments (e.g. some titles repeat
 * the same phrase either side of a delimiter). */
function cleanTitle(title) {
  if (!title) return "Guideline";
  const normalized = title
    .replace(/\s*-{2,}\s*/g, " — ")
    .replace(/\s{2,}/g, " ")
    .trim();
  const parts = normalized.split(" — ").map((p) => p.trim());
  const deduped = parts.filter((part, i) => i === 0 || part.toLowerCase() !== parts[i - 1].toLowerCase());
  return deduped.join(" — ");
}

const EXCERPT_PREVIEW_LENGTH = 220;

function ResultCard({ result }) {
  const [expanded, setExpanded] = useState(false);
  const badge = sourceBadge(result.source);
  const title = cleanTitle(result.title);
  const excerpt = result.excerpt || "";
  const isLong = excerpt.length > EXCERPT_PREVIEW_LENGTH;
  const shown = expanded || !isLong ? excerpt : `${excerpt.slice(0, EXCERPT_PREVIEW_LENGTH).trimEnd()}…`;

  return (
    <li className="rounded-lg border border-slate-border p-3">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="inline-flex items-center gap-1 rounded-full border border-slate-border bg-slate-bg px-2 py-0.5 text-[11px] font-semibold text-slate-600">
          {badge.emoji} {badge.label}
        </span>
      </div>
      <div className="font-semibold text-clinical-ink">{title}</div>
      <p className="mt-1.5 border-l-2 border-clinical-teal pl-2 text-xs leading-relaxed text-slate-600">{shown}</p>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1.5 flex items-center gap-1 text-xs font-semibold text-clinical-teal-dark hover:underline"
        >
          <ChevronDown size={13} className={cn("transition-transform", expanded && "rotate-180")} />
          {expanded ? "Collapse" : "Expand full guideline"}
        </button>
      )}
    </li>
  );
}

export default function EvidenceExplorer({ mode = "clinician" }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const runSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setSearched(true);
    try {
      setResults(await searchEvidence(query.trim(), mode));
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-slate-border bg-white p-4 shadow-card">
      <h2 className="mb-3 text-[15px] font-semibold text-clinical-ink">
        Search StatPearls, OpenFDA &amp; MedQuAD Guideline Chunks
      </h2>

      {mode === "patient" && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2.5 text-sm text-sky-800">
          <Info size={16} className="mt-0.5 shrink-0" />
          <p>
            <strong>ℹ️ Clinical Reference Library:</strong> These are verified medical excerpts used by our
            clinical evaluation engine to cross-reference symptoms and medications.
          </p>
        </div>
      )}

      <div className="mb-4 flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runSearch()}
          placeholder="e.g. metformin renal dosing, warfarin pregnancy, potassium chloride infusion..."
          className="flex-1 rounded-lg border border-slate-border bg-slate-bg px-3 py-2 text-sm text-clinical-ink outline-none transition focus:border-clinical-teal focus:bg-white focus:ring-2 focus:ring-clinical-teal/20"
        />
        <button
          type="button"
          onClick={runSearch}
          className="flex items-center gap-1.5 rounded-lg bg-clinical-teal px-4 py-2 text-sm font-semibold text-white transition hover:bg-clinical-teal-dark"
        >
          <Search size={14} />
          Search
        </button>
      </div>

      {!searched && (
        <p className="text-sm text-clinical-muted">
          Enter a clinical topic, drug name, or guideline keyword to search the corpus.
        </p>
      )}
      {searched && loading && <p className="text-sm text-clinical-muted">Searching…</p>}
      {searched && !loading && results.length === 0 && (
        <p className="text-sm text-clinical-muted">No matching guideline chunks found.</p>
      )}

      <ul className="space-y-3">
        {results.map((r, i) => (
          <ResultCard key={`${r.chunk_id}-${i}`} result={r} />
        ))}
      </ul>
    </div>
  );
}
