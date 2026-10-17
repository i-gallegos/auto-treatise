import { useMemo, useState } from "react";
import { facetValueLabel } from "./caseKeys";
import { shareFill } from "./courtStats";

const FACET_HINTS = {
  court_level: "coarse rung on the hierarchy (datasets/courts.py COURT_LEVELS)",
  circuit: "federal appellate and district courts only; state courts are not in a circuit",
  jurisdiction: "US for nationwide courts, USPS code otherwise",
  court_id: "CourtListener court slug — one per court",
  judge_author_party:
    "appointing party of the judges who wrote the opinion. Every signer counts, "
    + "as does a lone judge, and every judge who joined a per curiam. "
    + "Concurrences and dissents do not. One shared party is that party; "
    + "different parties are mixed. Unknown if an author cannot be identified "
    + "or matched. State and Article I courts are not Article III",
};

function Swatch({ share }) {
  return (
    <span
      className="study-court-swatch"
      style={{ background: shareFill(share) ?? "#fff" }}
    />
  );
}

/**
 * @param {"tree"|"graph"} scope which corpus the option counts come from.
 * @param {string} [emptyHint] copy when the selection is empty.
 * @param {boolean} [showRamp] share-fill legend; off on tabs that do not shade nodes.
 */
export function CourtHighlightPanel({
  options,
  selection,
  onToggle,
  onClear,
  scope = "tree",
  caseIndex,
  disabled = false,
  disabledReason = "",
  emptyHint = "Pick courts or panels to shade the tree by each node's share of the selection",
  showRamp = true,
  extraRef = null,
}) {
  const [openFacet, setOpenFacet] = useState(null);
  const selectedChips = useMemo(() => {
    const out = [];
    for (const facet of options) {
      for (const entry of facet.values) {
        if (selection.has(entry.key)) {
          out.push({
            key: entry.key,
            label: facetValueLabel(facet.id, entry.value, caseIndex),
            count: entry.count,
          });
        }
      }
    }
    return out;
  }, [options, selection, caseIndex]);

  if (disabled) {
    return (
      <div className="study-court-panel">
        <div className="study-court-head">
          <span className="study-court-title">{emptyHint}</span>
          <span className="study-muted">{disabledReason}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="study-court-panel">
      <div className="study-court-head">
        <span className="study-court-title">{emptyHint}</span>
        {selectedChips.length > 0 && (
          <>
            <span className="study-court-chips">
              {selectedChips.map((chip) => (
                <button
                  key={chip.key}
                  type="button"
                  className="study-court-chip"
                  onClick={() => onToggle(chip.key)}
                  title="remove"
                >
                  {chip.label} <span className="study-muted">({chip.count})</span> ✕
                </button>
              ))}
            </span>
            <button type="button" className="study-court-clear" onClick={onClear}>
              Clear
            </button>
          </>
        )}
      </div>

      <div className="study-court-facets">
        {options.map((facet) => {
          const isOpen = openFacet === facet.id;
          const selectedHere = facet.values.filter((v) => selection.has(v.key)).length;

          return (
            <div key={facet.id} className="study-court-facet">
              <button
                type="button"
                className={`study-court-facet-head${isOpen ? " open" : ""}`}
                onClick={() => setOpenFacet(isOpen ? null : facet.id)}
                title={FACET_HINTS[facet.id] ?? ""}
              >
                {facet.label}
                <span className="study-muted">
                  {" "}
                  {facet.values.length}
                  {selectedHere ? ` · ${selectedHere} on` : ""}
                </span>
              </button>

              {isOpen && (
                <div className="study-court-values">
                  {facet.values.length === 0 && (
                    <div className="study-muted study-court-empty">No values</div>
                  )}
                  {facet.values.map((entry) => (
                    <button
                      key={entry.key}
                      type="button"
                      className={`study-court-value${selection.has(entry.key) ? " on" : ""}`}
                      onClick={() => onToggle(entry.key)}
                    >
                      <span>{facetValueLabel(facet.id, entry.value, caseIndex)}</span>
                      <span className="study-court-value-tail">
                        <span className="study-muted">{entry.count}</span>
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
        <div className="study-court-extra" ref={extraRef} />
      </div>

      {/* Floats over the tree's top-left corner so it takes no vertical space.
          The share ramp only means something where nodes are shaded by share;
          in the graph the filter is a ring on the matching cases, not a fill. */}
      {selection.size > 0 && showRamp && (scope === "tree" ? (
        <span className="study-court-ramp">
          0%
          {[0.15, 0.4, 0.7, 1].map((share) => (
            <Swatch key={share} share={share} />
          ))}
          100%
        </span>
      ) : (
        <span className="study-court-ramp study-muted">
          matches ringed, the rest dimmed
        </span>
      ))}
    </div>
  );
}

export default CourtHighlightPanel;
