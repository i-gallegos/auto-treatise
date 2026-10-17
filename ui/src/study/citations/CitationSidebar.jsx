import { splitSideLabel, splitSideOf } from "../splitMembership";
import { displayNodeId } from "../nodeIds";

const num = (x) => (x === null || x === undefined ? "—" : x);

export function SplitRows({ node, splitIndex, caseIndex = null }) {
  if (!splitIndex || !node) return null;
  const side = splitSideOf(
    splitIndex,
    { caseId: node.case_id, opinionId: node.opinion_id },
    caseIndex,
  );
  return (
    <>
      <dt>Split</dt>
      <dd>
        <span className={`study-split-chip${side ? ` study-split-chip--${side}` : ""}`}>
          {splitSideLabel(side) ?? "not in split"}
        </span>
      </dd>
    </>
  );
}

export function CaseDetail({ node, leafLabel, extra = null, splitIndex = null, caseIndex = null }) {
  if (!node) return null;
  return (
    <div className="study-cit-detail">
      <div className="study-cit-detail-name">{node.name}</div>
      <dl>
        <dt>Court</dt><dd>{node.court_name || node.court || "?"}</dd>
        <dt>Decided</dt><dd>{node.date ?? "?"}</dd>
        <SplitRows node={node} splitIndex={splitIndex} caseIndex={caseIndex} />
        <dt>Level</dt><dd>{node.court_level ?? "?"}</dd>
        <dt>Circuit</dt><dd>{node.circuit ?? "—"}</dd>
        <dt>Jurisdiction</dt><dd>{node.jurisdiction ?? "—"}</dd>
        <dt>Status</dt><dd>{node.status ?? "?"}</dd>
        <dt>Leaf</dt><dd>{node.leaf ? (leafLabel ?? displayNodeId(node.leaf)) : "Unassigned"}</dd>
        <dt>Cites (corpus)</dt><dd>{node.out_deg}</dd>
        <dt>Cited by</dt><dd>{node.in_deg}</dd>
        <dt>Cites (total)</dt><dd>{num(node.n_cited_total)}</dd>
        <dt>Case_id</dt><dd>{num(node.case_id)}</dd>
        <dt>Opinion</dt><dd>{node.opinion_id}</dd>
      </dl>
      {extra}
    </div>
  );
}

/** "federal_appellate" becomes "Federal Appellate"; other labels are as is. */
function legendText(label) {
  const text = String(label);
  if (!text.includes("_")) return text;
  return text
    .split("_")
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

/** Legend rows with counts, click to mute a group. */
export function Legend({ groups, hidden, onToggle, colorFor, labelFor, noLeafColor }) {
  return (
    <div className="study-cit-legend">
      {groups.map(([key, n]) => (
        <button
          key={key}
          type="button"
          className={`study-cit-legend-row${hidden.has(key) ? " off" : ""}`}
          onClick={() => onToggle(key)}
        >
          <span
            className="study-cit-swatch"
            style={{
              background: key === "__none__"
                ? noLeafColor
                : (colorFor(key) ?? noLeafColor),
            }}
          />
          <span>
            {key === "__none__" ? "Unassigned" : legendText(labelFor(key))}{" "}
            <span className="study-muted">({n})</span>
          </span>
        </button>
      ))}
    </div>
  );
}
