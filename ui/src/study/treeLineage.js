const CONFIDENCE_RANK = { high: 2, medium: 1, low: 0 };

/**
 * Follow one case down the tree.
 *
 * Returns null when there is no graph or case id; otherwise a walk that may
 * be incomplete if the case was never assigned (or dropped mid-path).
 */
export function walkRunLineage(graph, caseId) {
  const id = String(caseId ?? "").trim();
  if (!graph?.nodes || !id) return null;
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const steps = [];
  let node = byId.get(graph.root_id || "n0");
  const seen = new Set();
  while (node) {
    if (node.is_leaf) {
      return {
        complete: true,
        leaf_id: node.id,
        leaf_label: node.label || "",
        leaf_original_label: node.original_label || "",
        leaf_description: node.description || "",
        leaf_summary: node.summary || "",
        steps,
      };
    }
    if (seen.has(node.id)) break;
    seen.add(node.id);
    const record = node.assignments?.[id];
    if (!record) {
      return {
        complete: false,
        leaf_id: "",
        leaf_label: "",
        leaf_original_label: "",
        leaf_description: "",
        leaf_summary: "",
        steps,
      };
    }
    const childId = record.side === "left" ? node.left_id : node.right_id;
    const child = byId.get(childId);
    steps.push({
      depth: steps.length,
      node_id: node.id,
      side: String(record.side || "").toLowerCase(),
      confidence: record.confidence || "",
      label: child?.label || "",
      original_label: child?.original_label || "",
      summary: child?.summary || "",
      axis: node.split_criterion_short || "",
      why: record.rationale || "",
      extra: "",
    });
    if (!child) break;
    node = child;
  }
  return {
    complete: false,
    leaf_id: "",
    leaf_label: "",
    leaf_original_label: "",
    leaf_description: "",
    leaf_summary: "",
    steps,
  };
}

export function summarizeWalk(walked) {
  const steps = walked?.steps ?? [];
  const confidences = steps.map((s) => s.confidence).filter(Boolean);
  const minConfidence = confidences.length
    ? confidences.reduce((a, b) => (
      (CONFIDENCE_RANK[a] ?? 0) <= (CONFIDENCE_RANK[b] ?? 0) ? a : b
    ))
    : "";
  return {
    min_confidence: minConfidence,
    n_low_confidence_steps: String(
      confidences.filter((c) => String(c).toLowerCase() === "low").length,
    ),
    complete_path: walked?.complete ? "yes" : "no",
  };
}

/**
 * Node ids on a finished (or partial) walk, leaf included.
 *
 * The steps name the *split* nodes the case passed through; the leaf is only
 * on `leaf_id`. Callers that ask "is this case in n0.L.L?" need both.
 */
export function pathNodeIds(walked) {
  const ids = new Set();
  if (!walked) return ids;
  for (const step of walked.steps ?? []) {
    const nodeId = String(step?.node_id ?? "").trim();
    if (nodeId) ids.add(nodeId);
  }
  const leafId = String(walked.leaf_id ?? "").trim();
  if (leafId) ids.add(leafId);
  return ids;
}

/** Treatise-shaped object the lineage box consumes. */
export function assignmentFromWalk(walked, extras = {}) {
  if (!walked) return null;
  return {
    leaf_id: walked.leaf_id,
    leaf_label: walked.leaf_label,
    leaf_original_label: walked.leaf_original_label || "",
    leaf_description: walked.leaf_description,
    leaf_summary: walked.leaf_summary,
    lineage: walked.steps,
    ...summarizeWalk(walked),
    ...extras,
  };
}
