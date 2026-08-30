// Minimal, dependency-free Markdown -> React renderer.
// Supports the subset the clinical LLM actually emits: headings (### / ## / #),
// bold (**text**), unordered list items (- item), horizontal rules (---), and
// paragraphs. Deliberately avoids dangerouslySetInnerHTML.

function renderInline(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>;
  });
}

export function Markdown({ text, className }) {
  if (!text) return null;
  const lines = text.replace(/\r\n/g, "\n").split("\n");

  const blocks = [];
  let listBuffer = [];

  const flushList = (key) => {
    if (listBuffer.length) {
      blocks.push(
        <ul key={`ul-${key}`} className="list-disc pl-5 space-y-1 my-2">
          {listBuffer.map((item, i) => (
            <li key={i}>{renderInline(item, `li-${key}-${i}`)}</li>
          ))}
        </ul>
      );
      listBuffer = [];
    }
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    if (trimmed === "") {
      flushList(idx);
      return;
    }
    if (/^---+$/.test(trimmed)) {
      flushList(idx);
      blocks.push(<hr key={`hr-${idx}`} className="my-3 border-slate-border" />);
      return;
    }
    const heading = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushList(idx);
      const level = heading[1].length;
      const content = heading[2];
      const sizes = {
        1: "text-lg font-bold mt-3 mb-1",
        2: "text-base font-bold mt-3 mb-1",
        3: "text-[15px] font-semibold mt-3 mb-1 text-clinical-teal-dark",
        4: "text-sm font-semibold mt-2 mb-1",
      };
      blocks.push(
        <div key={`h-${idx}`} className={sizes[level] || sizes[4]}>
          {renderInline(content, `h-${idx}`)}
        </div>
      );
      return;
    }
    const listItem = trimmed.match(/^[-*]\s+(.*)$/);
    if (listItem) {
      listBuffer.push(listItem[1]);
      return;
    }

    flushList(idx);
    blocks.push(
      <p key={`p-${idx}`} className="my-1 leading-relaxed">
        {renderInline(trimmed, `p-${idx}`)}
      </p>
    );
  });
  flushList("end");

  return <div className={className}>{blocks}</div>;
}
