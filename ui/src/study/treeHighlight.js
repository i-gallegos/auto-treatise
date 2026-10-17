import { asKey } from "./caseKeys";
import {
  categoryShare,
  matchInNode,
  selectionCategoryTotal,
  shareFill,
  shareTextColor,
} from "./courtStats";
import {
  CITER_COLOR,
  CITER_GLOW,
  COURT_MATCH_COLOR,
  citerNodeFill,
} from "./citations/useCitationData";

/**
 * `case_id -> [node ids]`. A case appears in its leaf and in every ancestor,
 * because `corpus_level.json` repeats the case list at each level, so the path is
 * read straight off the nodes rather than walked up the tree.
 */
export function nodeIdsByCaseId(nodes) {
  const out = new Map();
  for (const node of nodes ?? []) {
    for (const raw of node.cases ?? []) {
      const caseId = asKey(raw);
      if (!caseId) continue;
      if (!out.has(caseId)) out.set(caseId, []);
      out.get(caseId).push(node.id);
    }
  }
  return out;
}

/** How many of each node's cases cite the selected case directly. */
export function citerCountsByNode(nodes, citerCaseIds) {
  const counts = new Map();
  let max = 0;
  if (!citerCaseIds?.size) return { counts, max };
  for (const node of nodes ?? []) {
    let count = 0;
    for (const raw of node.cases ?? []) {
      if (citerCaseIds.has(asKey(raw))) count += 1;
    }
    if (count > 0) {
      counts.set(node.id, count);
      max = Math.max(max, count);
    }
  }
  return { counts, max };
}

export function buildTreeAnnotations({
  nodes,
  courtStats,
  courtSelection,
  selectedCaseId = null,
  citerCaseIds = null,
  rootId = null,
}) {
  const out = {};
  const caseActive = !!selectedCaseId || !!citerCaseIds?.size;
  const { counts, max } = citerCountsByNode(nodes, citerCaseIds);
  const categoryTotal = courtSelection?.size > 0
    ? selectionCategoryTotal(courtStats, courtSelection, rootId)
    : 0;

  for (const node of nodes ?? []) {
    const annotation = {};

    // While either overlay is active every node is annotated, selected or not:
    // unselected nodes are explicitly muted so the default leaf-green /
    // split-indigo palette stops competing with the overlay fill. Otherwise a
    // node matching nothing still looks coloured, and the filter reads as noise.
    if (caseActive) {
      const count = counts.get(node.id) ?? 0;
      if (count > 0) {
        annotation.fill = citerNodeFill(count, max);
        annotation.border = CITER_COLOR;
        annotation.badge = `${count} citing`;
        annotation.title =
          `${count} case(s) in this node cite the selected case directly`;
      } else {
        annotation.muted = true;
      }
    } else if (courtSelection?.size > 0) {
      const report = matchInNode(courtStats?.get(node.id), courtSelection);
      if (report && report.matched > 0) {
        const share = categoryShare(report.matched, categoryTotal);
        annotation.fill = shareFill(share);
        annotation.border = COURT_MATCH_COLOR;
        annotation.textColor = shareTextColor(share);
        annotation.badge =
          `${report.matched} / ${categoryTotal} · ${Math.round(100 * share)}%`;
        annotation.title =
          `${report.matched} of ${categoryTotal} selected-court cases are in this node `
          + `(${report.matched} of ${report.total} of the node's cases)`;
      } else {
        annotation.muted = true;
        // Explicit zero, so "nothing here" is distinguishable from "not computed".
        if (report) annotation.badge = `0 / ${categoryTotal}`;
      }
    }

    if (selectedCaseId && (node.cases ?? []).some((c) => asKey(c) === selectedCaseId)) {
      annotation.outline = CITER_COLOR;
      annotation.outlineGlow = CITER_GLOW;
      if (!annotation.badge) annotation.badge = "selected case";
      annotation.title = "the selected case is in this node";
    }

    if (Object.keys(annotation).length > 0) out[node.id] = annotation;
  }

  if (selectedCaseId) {
    for (const nodeId of nodeIdsByCaseId(nodes).get(selectedCaseId) ?? []) {
      out[nodeId] = { ...(out[nodeId] ?? {}), dotted: CITER_COLOR };
    }
  }
  return out;
}

