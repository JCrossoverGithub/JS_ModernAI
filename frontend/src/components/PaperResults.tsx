import { useState } from "react";
import {
  Download,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Users,
  Calendar,
  Quote,
  Building,
} from "lucide-react";
import type { Paper } from "../types";

function PaperCard({ paper, index, number }: { paper: Paper; index: number; number: number }) {
  const [expanded, setExpanded] = useState(false);

  const authorsStr =
    paper.authors.length > 3
      ? paper.authors.slice(0, 3).join(", ") + ` +${paper.authors.length - 3}`
      : paper.authors.join(", ");

  return (
    <div
      className="group bg-white/[0.03] border border-white/[0.06] rounded-xl hover:border-white/[0.1] transition-all animate-fade-in"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <div className="p-4">
        {/* Title + links */}
        <div className="flex items-start gap-3 mb-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0 mt-0.5">
            <span className="text-[11px] font-bold text-indigo-400">{number}</span>
          </div>
          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-medium text-white leading-snug">
              {paper.url ? (
                <a
                  href={paper.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-indigo-300 transition-colors"
                >
                  {paper.title}
                </a>
              ) : (
                paper.title
              )}
            </h4>
          </div>
        </div>

        {/* Meta row */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pl-10 mb-2">
          {paper.authors.length > 0 && (
            <span className="flex items-center gap-1 text-[11px] text-slate-400">
              <Users size={10} />
              {authorsStr}
            </span>
          )}
          {paper.year && (
            <span className="flex items-center gap-1 text-[11px] text-slate-400">
              <Calendar size={10} />
              {paper.year}
            </span>
          )}
          {paper.venue && (
            <span className="flex items-center gap-1 text-[11px] text-slate-400">
              <Building size={10} />
              {paper.venue}
            </span>
          )}
          {paper.citationCount > 0 && (
            <span className="flex items-center gap-1 text-[11px] text-slate-400">
              <Quote size={10} />
              {paper.citationCount.toLocaleString()} citations
            </span>
          )}
        </div>

        {/* Abstract (expandable) */}
        {paper.abstract && (
          <div className="pl-10">
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300 transition-colors mb-1"
            >
              {expanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              {expanded ? "Hide abstract" : "Show abstract"}
            </button>
            {expanded && (
              <p className="text-xs text-slate-400 leading-relaxed animate-fade-in">
                {paper.abstract}
              </p>
            )}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex items-center gap-2 pl-10 mt-3">
          {(paper.downloadUrl || paper.url) && (
            <a
              href={paper.downloadUrl || paper.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-500/15 hover:bg-indigo-500/25 border border-indigo-500/20 hover:border-indigo-500/40 text-indigo-300 rounded-lg text-xs font-medium transition-all active:scale-95"
            >
              <Download size={12} />
              {paper.downloadUrl ? "Download PDF" : "Open Paper"}
            </a>
          )}
          {paper.url && paper.downloadUrl && (
            <a
              href={paper.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-slate-300 rounded-lg text-xs font-medium transition-all active:scale-95"
            >
              <ExternalLink size={12} />
              View Paper
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export default function PaperResults({ papers }: { papers: Paper[] }) {
  const [showAll, setShowAll] = useState(false);
  const displayed = showAll ? papers : papers.slice(0, 5);
  const hasMore = papers.length > 5;

  return (
    <div className="mt-4 space-y-3">
      <div className="flex items-center gap-2 mb-3">
        <div className="h-px flex-1 bg-white/[0.04]" />
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider px-2">
          {papers.length} Research Papers Found
        </span>
        <div className="h-px flex-1 bg-white/[0.04]" />
      </div>

      <div className="space-y-2">
        {displayed.map((paper, i) => (
          <PaperCard key={`${paper.title}-${i}`} paper={paper} index={i} number={i + 1} />
        ))}
      </div>

      {hasMore && !showAll && (
        <button
          onClick={() => setShowAll(true)}
          className="w-full py-2 text-xs text-slate-400 hover:text-white hover:bg-white/[0.04] rounded-lg transition-colors"
        >
          Show {papers.length - 5} more papers
        </button>
      )}
    </div>
  );
}
