import { useState } from "react";
import { GREEN, INK, ORANGE, TEAL } from "./theme";

// Small renderers for the case drawer: majority boundary and symbol analysis.

const BOUNDARY_STATUS_STYLES = {
  boundary_found: { color: ORANGE.shade1, background: "rgba(209, 110, 76, 0.12)", border: "rgba(209, 110, 76, 0.4)" },
  majority_only: { color: GREEN.shade2, background: "rgba(133, 165, 95, 0.12)", border: "rgba(133, 165, 95, 0.35)" },
  ambiguous: { color: ORANGE.shade2, background: "rgba(115, 49, 28, 0.12)", border: "rgba(115, 49, 28, 0.35)" },
  no_majority: { color: ORANGE.shade2, background: "rgba(115, 49, 28, 0.12)", border: "rgba(115, 49, 28, 0.35)" },
};

function boundaryStatusStyle(status) {
  if (status && BOUNDARY_STATUS_STYLES[status]) {
    return BOUNDARY_STATUS_STYLES[status];
  }
  return { color: INK.mid, background: "rgba(63, 63, 63, 0.1)", border: "rgba(63, 63, 63, 0.3)" };
}

/** Status chip plus a one-line explanation of what the models actually received. */
export function MajorityBoundaryChip({ boundary }) {
  if (!boundary) {
    return (
      <div style={{ fontSize: 10, marginBottom: 6, opacity: 0.6, fontStyle: "italic" }}>
        No majority boundary computed yet for this case.
      </div>
    );
  }

  const style = boundaryStatusStyle(boundary.status);
  const truncated = boundary.status === "boundary_found";
  const explanation = truncated
    ? `Paragraphs after ${(boundary.endIndex ?? 0) + 1} were cut before prompting${
        boundary.splitBoundary ? `; paragraph ${(boundary.endIndex ?? 0) + 1} was cut mid-paragraph` : ""
      }.`
    : boundary.status === "majority_only"
      ? "No separate opinion detected; the whole opinion was sent to the models."
      : "Boundary could not be resolved, so all paragraphs were kept and sent to the models.";

  return (
    <div style={{ fontSize: 10, marginBottom: 6, display: "grid", gap: 3 }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
        <span
          style={{
            padding: "2px 7px",
            borderRadius: 999,
            color: style.color,
            background: style.background,
            border: `1px solid ${style.border}`,
            fontWeight: 700,
          }}
        >
          {boundary.status || "unknown"}
        </span>
        {boundary.keptParagraphCount !== null && boundary.inputParagraphCount !== null && (
          <span style={{ opacity: 0.8 }}>
            {boundary.keptParagraphCount} / {boundary.inputParagraphCount} paragraphs sent to models
          </span>
        )}
      </div>
      {boundary.marker && (
        <div style={{ opacity: 0.85, fontStyle: "italic" }}>Marker: &ldquo;{boundary.marker}&rdquo;</div>
      )}
      <div style={{ opacity: 0.75 }}>{explanation}</div>
    </div>
  );
}

/** Labeled rule marking where the majority opinion stops. */
export function MajorityBoundaryDivider({ marker }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        margin: "4px 0",
        fontSize: 9,
        fontWeight: 700,
        color: ORANGE.shade1,
      }}
    >
      <span style={{ flex: "0 0 auto" }}>
        MAJORITY ENDS{marker ? ` — ${marker}` : ""}
      </span>
      <span style={{ flex: 1, height: 0, borderTop: "2px dashed rgba(209, 110, 76, 0.7)" }} />
    </div>
  );
}

/**
 * Paragraph body split at the majority boundary: the kept head renders normally and the
 * cut tail renders dimmed, with the divider between them.
 */
export function MajorityParagraphBody({ paragraph, boundary, isBoundaryParagraph, isPastBoundary }) {
  if (isPastBoundary) {
    return <div style={{ opacity: 0.4 }}>{paragraph}</div>;
  }
  if (!isBoundaryParagraph) {
    return <div>{paragraph}</div>;
  }
  if (!boundary.splitBoundary || boundary.endCharOffset === null) {
    return <div style={{ opacity: 0.4 }}>{paragraph}</div>;
  }
  return (
    <div>
      <span>{paragraph.slice(0, boundary.endCharOffset)}</span>
      <MajorityBoundaryDivider marker={boundary.marker} />
      <span style={{ opacity: 0.4 }}>{paragraph.slice(boundary.endCharOffset)}</span>
    </div>
  );
}

export function SymbolAnalysisCard({ sa, nodeInfo }) {
  const [excerptOpen, setExcerptOpen] = useState(true);
  const [chainOpen, setChainOpen] = useState(true);

  const TRUTH_CHIP = {
    true:        { label: "True",        color: GREEN.shade2, bg: "rgba(133,165,95,0.12)", border: "rgba(133,165,95,0.35)" },
    false:       { label: "False",       color: ORANGE.shade2, bg: "rgba(115,49,28,0.12)", border: "rgba(115,49,28,0.35)" },
    unaddressed: { label: "Unaddressed", color: INK.muted, bg: "rgba(114,114,114,0.1)", border: "rgba(114,114,114,0.3)" },
  };
  const chip = TRUTH_CHIP[sa.truth_value] ?? TRUTH_CHIP.unaddressed;

  return (
    <div
      style={{
        border: "1px solid var(--study-rule, #d8d8d8)",
        borderRadius: 6,
        padding: "7px 9px",
        background: "var(--study-panel, #fff)",
      }}
    >
      {/* ── Header: symbol id + truth chip + short label ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", marginBottom: 3 }}>
        <span
          style={{
            fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
            fontSize: 10,
            padding: "2px 6px",
            borderRadius: 4,
            background: "rgba(89,147,160,0.08)",
            border: "1px solid rgba(89,147,160,0.25)",
            color: TEAL.shade1,
          }}
        >
          {sa.symbol_id}
        </span>
        {nodeInfo && nodeInfo.short && nodeInfo.short !== sa.symbol_id && (
          <span style={{ fontSize: 10, fontWeight: 700, color: INK.mid }}>
            {nodeInfo.short}
          </span>
        )}
        <span
          style={{
            fontSize: 10,
            padding: "2px 7px",
            borderRadius: 999,
            color: chip.color,
            background: chip.bg,
            border: `1px solid ${chip.border}`,
            fontWeight: 700,
          }}
        >
          {chip.label}
        </span>
      </div>

      {/* ── Node definition from grammar ── */}
      {nodeInfo && nodeInfo.text && (
        <div
          style={{
            fontSize: 10,
            lineHeight: 1.4,
            marginBottom: 5,
            padding: "4px 7px",
            borderRadius: 4,
            background: "rgba(63,63,63,0.05)",
            border: "1px solid rgba(63,63,63,0.12)",
            color: INK.muted,
            fontStyle: "italic",
          }}
        >
          <span style={{ fontWeight: 700, fontStyle: "normal", marginRight: 4 }}>Definition:</span>
          {nodeInfo.text}
        </div>
      )}

      {/* ── Issue ── */}
      {sa.issue_summary && (
        <div
          style={{
            fontSize: 10,
            lineHeight: 1.4,
            marginBottom: 5,
            padding: "4px 7px",
            borderRadius: 4,
            background: "rgba(89,147,160,0.06)",
            border: "1px solid rgba(89,147,160,0.18)",
            color: INK.mid,
          }}
        >
          <span style={{ fontWeight: 700, marginRight: 4 }}>Issue:</span>
          {sa.issue_summary}
        </div>
      )}

      {/* ── Facts ── */}
      {sa.facts_summary && (
        <div
          style={{
            fontSize: 10,
            lineHeight: 1.4,
            marginBottom: 5,
            padding: "4px 7px",
            borderRadius: 4,
            background: "rgba(89,147,160,0.06)",
            border: "1px solid rgba(89,147,160,0.18)",
            color: INK.mid,
          }}
        >
          <span style={{ fontWeight: 700, marginRight: 4 }}>Facts:</span>
          {sa.facts_summary}
        </div>
      )}

      {/* ── Application summary ── */}
      {sa.application_summary && (
        <div style={{ fontSize: 10, lineHeight: 1.4, marginBottom: 5 }}>
          {sa.application_summary}
        </div>
      )}

      {/* ── Reasoning chain ── */}
      {sa.reasoning_chain && (
        <div style={{ marginBottom: 5 }}>
          <button
            type="button"
            onClick={() => setChainOpen((v) => !v)}
            style={{
              fontSize: 10,
              color: TEAL.shade1,
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              marginBottom: 3,
            }}
          >
            {chainOpen ? "▾ Hide reasoning chain" : "▸ Show reasoning chain"}
          </button>
          {chainOpen && (
            <div
              style={{
                fontSize: 10,
                lineHeight: 1.5,
                padding: "5px 8px",
                borderRadius: 4,
                borderLeft: `3px solid ${TEAL.base}`,
                background: "rgba(89,147,160,0.08)",
                whiteSpace: "pre-wrap",
              }}
            >
              {sa.reasoning_chain}
            </div>
          )}
        </div>
      )}

      {/* ── Verbatim excerpts ── */}
      {Array.isArray(sa.relevant_texts) && sa.relevant_texts.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setExcerptOpen((v) => !v)}
            style={{
              fontSize: 10,
              color: TEAL.shade1,
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              marginBottom: 3,
            }}
          >
            {excerptOpen ? "▾ Hide excerpts" : `▸ Show ${sa.relevant_texts.length} excerpt(s)`}
          </button>
          {excerptOpen && (
            <div style={{ display: "grid", gap: 4 }}>
              {sa.relevant_texts.map((txt, ti) => (
                <div
                  key={ti}
                  style={{
                    fontSize: 10,
                    lineHeight: 1.35,
                    padding: "4px 7px",
                    borderRadius: 4,
                    borderLeft: `3px solid ${TEAL.base}`,
                    background: "rgba(89,147,160,0.08)",
                    fontStyle: "italic",
                  }}
                >
                  &ldquo;{txt}&rdquo;
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
