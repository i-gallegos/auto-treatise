import { originalLabelSuffix } from "./labelOverlay";
import { displayNodeId } from "./nodeIds";

/**
 * Collapsed-by-default assignment box: leaf label in the summary, branch
 * reasoning behind a nested disclosure. `showPredicted` is Annotation-tab
 * only (the CSV's predicted label); the drawer omits that row.
 * `showOriginalLabels` is Annotation-tab only: overlay headings keep the
 * original short name in parentheses. The drawer omits that too.
 *
 * The Annotation tab also passes `branchVerdicts`/`onBranchVerdict`, which turns
 * each branch in the lineage into a place to record whether the secondary
 * sources confirm the tree's choice *at that branch*. The reason it belongs
 * here rather than in the judgement form: the treatises reach the first or
 * second split and stop, so agreement is not one answer per case but a depth,
 * and a depth is only meaningful next to the branch it refers to. The drawer
 * passes neither prop and renders exactly as before.
 */

//: Both boxes clear = the sources never reach this branch, which is the
//: majority of branches and the reason this is not a single checkbox.
const BRANCH_CHOICES = [
  { verdict: "agree", label: "Agrees", title: "The sources place the case on this side of the split" },
  { verdict: "disagree", label: "Disagrees", title: "The sources place the case on the other side" },
];

/** How far down the path the sources actually go, as a sentence fragment. */
function reachSummary(lineage, verdicts) {
  const marked = lineage
    .map((step, i) => ({ depth: Number.isFinite(step.depth) ? step.depth : i }))
    .filter(({ depth }) => verdicts?.[depth]);
  if (marked.length === 0) return "";
  const deepest = Math.max(...marked.map((m) => m.depth));
  const n = marked.length;
  return `sources reach depth ${deepest + 1} of ${lineage.length}`
    + (n === marked.length && n > 1 ? ` · ${n} branches marked` : "");
}

function LabelWithOriginal({ label, original, showOriginal }) {
  const suffix = showOriginal ? originalLabelSuffix({ label, original_label: original }) : "";
  return (
    <>
      {label}
      {suffix ? <span className="ann-label-original"> ({suffix})</span> : null}
    </>
  );
}

export function TreeLineageBox({
  assignment,
  showPredicted = false,
  showOriginalLabels = false,
  className = "",
  branchVerdicts = null,
  onBranchVerdict = null,
  lineageOpen = false,
  //: Summary heading. Held-out reviews pass the annotator's name, because the
  //: path they judge is its label rather than the tree's routing.
  title = "Auto-treatise assignment",
}) {
  const lineage = Array.isArray(assignment?.lineage) ? assignment.lineage : [];
  const leafLabel = assignment?.leaf_label || lineage[lineage.length - 1]?.label || "";
  if (!leafLabel && !assignment?.pred_leaf_label && lineage.length === 0) {
    return null;
  }

  const editable = typeof onBranchVerdict === "function";
  const reach = branchVerdicts ? reachSummary(lineage, branchVerdicts) : "";

  const footerParts = [];
  if (assignment?.run_id) {
    footerParts.push(`${assignment.assignment_source || "tree"} · ${assignment.run_id}`);
    if (assignment.n_low_confidence_steps && assignment.n_low_confidence_steps !== "0") {
      footerParts.push(`${assignment.n_low_confidence_steps} low-confidence step(s)`);
    }
  }

  return (
    // Still closed by default with the branch controls live. The reason the box
    // folds away at all is that the tree's answer should not be read before the
    // reviewer has formed one from the sources — and that argument does not
    // weaken because there is now something to tick inside.
    <details className={`ann-tree-box${className ? ` ${className}` : ""}`}>
      <summary className="ann-tree-summary">
        {title}
        {leafLabel && (
          <span className="ann-tree-summary-leaf">
            <LabelWithOriginal
              label={leafLabel}
              original={assignment.leaf_original_label || lineage[lineage.length - 1]?.original_label}
              showOriginal={showOriginalLabels}
            />
          </span>
        )}
        {reach && <span className="ann-tree-reach">{reach}</span>}
      </summary>
      <div className="ann-tree-row">
        <span className="ann-tree-key">Tree label</span>
        <span>
          {assignment.leaf_label ? (
            <LabelWithOriginal
              label={assignment.leaf_label}
              original={assignment.leaf_original_label}
              showOriginal={showOriginalLabels}
            />
          ) : "—"}
          {assignment.leaf_id && <code className="ann-node-id">{displayNodeId(assignment.leaf_id)}</code>}
        </span>
      </div>
      {showPredicted && (
        <div className="ann-tree-row">
          <span className="ann-tree-key">Predicted</span>
          <span>{assignment.pred_leaf_label || "—"}</span>
        </div>
      )}
      {assignment.leaf_summary && (
        <p className="ann-tree-desc">{assignment.leaf_summary}</p>
      )}
      {assignment.leaf_description && assignment.leaf_description !== assignment.leaf_summary && (
        <p className="ann-tree-desc">{assignment.leaf_description}</p>
      )}

      {lineage.length > 0 && (
        // `lineageOpen` is opt-in and nothing passes it yet; it exists so a
        // caller that has already shown the reviewer the sources can skip a
        // click without that becoming the default everywhere.
        <details className="ann-lineage" open={lineageOpen}>
          <summary className="ann-lineage-summary">
            Full lineage — {lineage.length} branch
            {lineage.length === 1 ? "" : "es"}
            {assignment.min_confidence && ` · min confidence ${assignment.min_confidence}`}
            {assignment.complete_path === "no" && " · incomplete path"}
          </summary>

          {editable && (
            <p className="ann-lineage-hint">
              Mark each branch the sources actually address. Leave a branch blank
              where they are silent — that is the finding, not a gap.
            </p>
          )}

          <ol className="ann-lineage-list">
            {lineage.map((step, i) => {
              const depth = Number.isFinite(step.depth) ? step.depth : i;
              const verdict = branchVerdicts?.[depth] ?? "";
              return (
                <li
                  className={`ann-step${verdict ? ` ann-step-${verdict}` : ""}`}
                  key={`${depth}-${step.node_id}`}
                >
                  <div className="ann-step-head">
                    <code className="ann-node-id">{displayNodeId(step.node_id)}</code>
                    <span className={`ann-side ann-side-${step.side}`}>
                      {step.side}
                    </span>
                    {step.confidence && (
                      <span
                        className={`ann-conf ann-conf-${step.confidence.toLowerCase()}`}
                      >
                        {step.confidence}
                      </span>
                    )}
                    <span className="ann-step-label">
                      <LabelWithOriginal
                        label={step.label}
                        original={step.original_label}
                        showOriginal={showOriginalLabels}
                      />
                    </span>
                  </div>
                  {step.axis && (
                    <p className="ann-step-axis">{step.axis}</p>
                  )}
                  {step.summary && (
                    <p className="ann-tree-desc">{step.summary}</p>
                  )}
                  {step.why && <p className="ann-step-why">{step.why}</p>}
                  {step.extra && <p className="ann-step-why">{step.extra}</p>}

                  {editable && (
                    <div className="ann-step-verdict">
                      <span className="ann-step-verdict-key">
                        d{depth} — secondary sources
                      </span>
                      {BRANCH_CHOICES.map((choice) => (
                        <label
                          className="dbb-check-label ann-step-check"
                          key={choice.verdict}
                          title={choice.title}
                        >
                          <input
                            type="checkbox"
                            // Exclusive by construction: ticking one clears the
                            // other, so the three states are agree, disagree
                            // and neither, with no way to express both.
                            checked={verdict === choice.verdict}
                            aria-label={`Depth ${depth} ${choice.label}`}
                            onChange={(e) => onBranchVerdict(
                              { ...step, depth },
                              e.target.checked ? choice.verdict : "",
                            )}
                          />
                          {choice.label}
                        </label>
                      ))}
                      {!verdict && (
                        <span className="ann-step-verdict-none">Not addressed</span>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ol>

          {footerParts.length > 0 && (
            <p className="ann-lineage-foot">{footerParts.join(" · ")}</p>
          )}
        </details>
      )}
    </details>
  );
}
