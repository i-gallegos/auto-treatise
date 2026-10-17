import { citersOfAny } from "./reachability";

/**
 * The subgraph induced by `keep`. Degrees are recounted from those edges, so a
 * training-case view does not size a node by citations from outside the tree.
 */
export function inducedSubgraph(nodes, links, keep) {
  const ids = new Set();
  const kept = [];
  for (const node of nodes ?? []) {
    if (!keep(node)) continue;
    ids.add(node.i);
    kept.push(node);
  }
  const inn = new Map();
  const out = new Map();
  const keptLinks = [];
  for (const edge of links ?? []) {
    if (!ids.has(edge.s) || !ids.has(edge.t)) continue;
    keptLinks.push(edge);
    if (!inn.has(edge.t)) inn.set(edge.t, []);
    inn.get(edge.t).push(edge.s);
    if (!out.has(edge.s)) out.set(edge.s, []);
    out.get(edge.s).push(edge.t);
  }
  return {
    nodes: kept.map((node) => ({
      ...node,
      in_deg: inn.get(node.i)?.length ?? 0,
      out_deg: out.get(node.i)?.length ?? 0,
    })),
    links: keptLinks,
    inDeg: inn,
    outDeg: out,
  };
}

/** Highest in-degree in the payload — the top of the slider. */
export function maxInDegree(nodes) {
  let max = 0;
  for (const node of nodes ?? []) max = Math.max(max, node.in_deg ?? 0);
  return max;
}

/**
 * @returns {null | {
 *   threshold: number,
 *   hubs: Set<number>, citers: Set<number>, visible: Set<number>,
 *   hubCount: number, citerCount: number, hiddenCount: number, total: number,
 * }}
 *   `null` when the threshold is off (0 or 1 — every case with any citation at
 *   all is already the whole graph), so callers can treat "no filter" as absence
 *   rather than a special case.
 */
export function degreeSelection(nodes, adjacency, threshold) {
  const wanted = Number(threshold);
  if (!Number.isFinite(wanted) || wanted <= 1) return null;

  const hubs = new Set();
  for (const node of nodes ?? []) {
    if ((node.in_deg ?? 0) >= wanted) hubs.add(node.i);
  }

  const citers = citersOfAny(adjacency, hubs);
  const visible = new Set([...hubs, ...citers]);

  return {
    threshold: wanted,
    hubs,
    citers,
    visible,
    hubCount: hubs.size,
    citerCount: citers.size,
    hiddenCount: Math.max(0, (nodes?.length ?? 0) - visible.size),
    total: nodes?.length ?? 0,
  };
}

/** One-line description of what the threshold is currently showing. */
export function describeDegreeSelection(selection) {
  if (!selection) return null;
  const { threshold, hubCount, citerCount, hiddenCount, total } = selection;
  if (hubCount === 0) {
    return `No case in this corpus is cited ${threshold} or more times, so nothing `
      + "is shown. Lower the threshold.";
  }
  return `${hubCount} case${hubCount === 1 ? "" : "s"} cited ${threshold}+ times, `
    + `${citerCount} faded case${citerCount === 1 ? "" : "s"} citing them directly, `
    + `${hiddenCount} of ${total} hidden.`;
}
