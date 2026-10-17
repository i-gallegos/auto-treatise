import { useCallback, useMemo } from "react";
import { facetValueLabel, resolve } from "../caseKeys";
import { buildAdjacency } from "./reachability";
import { ORANGE, TEAL } from "../theme";
import { groupIdOf } from "./groupLayout";
import { RANK_LABELS, applyChronologyRanks } from "./rankBands";

export {
  RANK_AXIS_LINES,
  RANK_LABELS,
  axisLinesForRank,
  displayRank,
  applyChronologyRanks,
} from "./rankBands";

const FORK_COLORS = [TEAL.base, ORANGE.base];

/**
 * Colours for the case-selection highlight. They have to agree across the
 * chronology scatter, the case tree and the sidebar legend, so they live in one
 * place.
 */
/** The case the user picked. */
export const SELECTED_COLOR = "#b91c1c";

/**
 * Cases (and tree nodes) selected by the court/jurisdiction filter. The same hue
 * the share ramp in `courtStats.shareFill` ends on, so "this node holds the Ninth
 * Circuit's cases" and "this case is Ninth Circuit" are visibly the same statement
 * across the two panes.
 */
export const COURT_MATCH_COLOR = TEAL.base;

/** Cases that cite it — one hop. Theme orange against the selected case's red. */
export const CITER_COLOR = ORANGE.base;
export const CITER_GLOW = "rgba(209, 110, 76, 0.20)";

/**
 * Fill for a tree node holding `n` of the citing cases. Graded by how many land
 * there, which is the only quantity left now that the relation is one hop: there
 * is no distance to encode, so the fill encodes concentration instead.
 */
export function citerNodeFill(count, maxCount = 1) {
  if (!(count > 0)) return null;
  const t = Math.pow(Math.min(1, count / Math.max(1, maxCount)), 0.6);
  const mix = (from, to) => Math.round(from + (to - from) * t);
  // ORANGE.tint1 #f4e1da -> ORANGE.base #d16e4c
  return `rgb(${mix(244, 209)}, ${mix(225, 110)}, ${mix(218, 76)})`;
}

export const CATEGORICAL = [
  "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2",
  "#edc948", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
  "#8cd17d", "#b6992d",
];

export const RANK_COLORS = {
  4: "#1b3f66", 3: "#2f6fb2", 2: "#76b7b2", 1: "#c3cdd6", 0: "#dfe3e8",
};

/**
 * What the node colours can mean. Offered identically by every view in both tabs,
 * so "colour by circuit" is one control with one meaning rather than a per-tab
 * variation. `tree leaf` needs a tree run and is disabled without one.
 */
export const COLOR_OPTIONS = [
  { id: "leaf", label: "Tree leaf", needsTree: true },
  { id: "court_level", label: "Court level" },
  { id: "circuit", label: "Circuit" },
  { id: "jurisdiction", label: "Jurisdiction" },
  { id: "judge_author_party", label: "Author appointing party" },
  { id: "rank", label: "Authority rank" },
];

export const NO_LEAF_COLOR = "#c9ccd1";

/**
 * Which child of `colorAt` this case took.
 *
 * `path` is root-to-leaf. The key is the next id after `colorAt`. A case that
 * never passes through `colorAt` is keyed by the branch it took off the parent
 * instead, so the bypass side of a history fork stays a color of its own.
 */
export function forkKey(path, colorAt) {
  if (!colorAt || !Array.isArray(path) || path.length === 0) return null;
  const at = path.indexOf(colorAt);
  if (at >= 0) {
    if (at + 1 < path.length) return path[at + 1];
    return colorAt;
  }
  const cut = colorAt.lastIndexOf(".");
  if (cut <= 0) return null;
  const parent = colorAt.slice(0, cut);
  const parentAt = path.indexOf(parent);
  if (parentAt >= 0 && parentAt + 1 < path.length) return path[parentAt + 1];
  return null;
}

const NO_NODES = [];
const NO_LINKS = [];

export function useCitationData(payload) {
  const hasData = !!payload && Array.isArray(payload.nodes) && payload.nodes.length > 0;
  const hasTree = hasData && !!payload.tree;

  const chronology = useMemo(
    () => (hasData ? applyChronologyRanks(payload) : { nodes: NO_NODES, timeAxis: null }),
    [hasData, payload],
  );
  const nodes = chronology.nodes;
  const links = useMemo(() => (hasData ? payload.links : NO_LINKS), [hasData, payload]);

  const byId = useMemo(() => new Map(nodes.map((n) => [n.i, n])), [nodes]);

  /** Both ids resolve to the same node, because artifacts disagree about which they carry. */
  const indexByOpinionId = useMemo(() => {
    const out = new Map();
    for (const node of nodes) {
      if (node.opinion_id !== null && node.opinion_id !== undefined) {
        out.set(String(node.opinion_id), node.i);
      }
    }
    return out;
  }, [nodes]);

  const indexByCaseId = useMemo(() => {
    const out = new Map();
    for (const node of nodes) {
      if (node.case_id !== null && node.case_id !== undefined) {
        out.set(String(node.case_id), node.i);
      }
    }
    return out;
  }, [nodes]);

  const treeNodes = useMemo(() => {
    if (!hasTree) return new Map();
    return new Map(payload.tree.nodes.map((n) => [n.id, n]));
  }, [hasTree, payload]);

  const leafColor = useMemo(
    () => new Map((hasData ? payload.leaves ?? [] : []).map((l) => [l.id, l.color])),
    [hasData, payload],
  );

  /** Edge indices per node, for the "light up my neighbours" highlight. */
  const { outAdj, inAdj } = useMemo(() => {
    const out = new Map();
    const inn = new Map();
    (links ?? []).forEach((e, k) => {
      if (!out.has(e.s)) out.set(e.s, []);
      if (!inn.has(e.t)) inn.set(e.t, []);
      out.get(e.s).push(k);
      inn.get(e.t).push(k);
    });
    return { outAdj: out, inAdj: inn };
  }, [links]);

  /** Node-index adjacency, for reachability. */
  const adjacency = useMemo(() => buildAdjacency(links), [links]);

  const indexFor = useCallback(
    ({ opinionId, caseId } = {}) => {
      if (opinionId !== null && opinionId !== undefined) {
        const hit = indexByOpinionId.get(String(opinionId));
        if (hit !== undefined) return hit;
      }
      if (caseId !== null && caseId !== undefined) {
        const hit = indexByCaseId.get(String(caseId));
        if (hit !== undefined) return hit;
      }
      return null;
    },
    [indexByOpinionId, indexByCaseId],
  );

  return {
    hasData,
    hasTree,
    nodes,
    links,
    byId,
    treeNodes,
    leafColor,
    outAdj,
    inAdj,
    adjacency,
    indexFor,
    stats: payload?.stats ?? {},
    meta: payload?.meta ?? {},
    timeAxis: chronology.timeAxis,
    timeAxisTree: payload?.time_axis_tree ?? null,
    figures: Array.isArray(payload?.figures) ? payload.figures : [],
    noLeafColor: payload?.no_leaf_color ?? NO_LEAF_COLOR,
  };
}

/** Grouping/colour helpers, shared by both views and the legend. */
export function useCitationColors({
  nodes, leafColor, treeNodes, colorBy, noLeafColor, caseIndex = null,
  colorAt = null, groups = null,
}) {
  const groupKey = useCallback(
    (n) => {
      if (!n) return null;
      if (groups?.length) return groupIdOf(n.path, groups);
      if (colorAt) return forkKey(n.path, colorAt);
      if (colorBy === "leaf") return n.leaf || null;
      if (colorBy === "rank") return String(n.rank ?? 0);
      if (colorBy === "circuit") return n.circuit || null;
      if (colorBy === "jurisdiction") return n.jurisdiction || null;
      if (colorBy === "judge_author_party") {
        const row = resolve(caseIndex, {
          opinionId: n.opinion_id,
          caseId: n.case_id,
        });
        const value = row?.judgeAuthorParty ?? n.judge_author_party ?? null;
        return value ? String(value) : null;
      }
      return n.court_level || null;
    },
    [colorBy, caseIndex, colorAt, groups],
  );

  const groupColors = useMemo(() => {
    const map = new Map();
    if (groups?.length) {
      groups.forEach((group) => map.set(group.id, group.color));
      return map;
    }
    if (colorAt) {
      const keys = [...new Set(nodes.map(groupKey).filter((k) => k !== null))].sort();
      keys.forEach((k, i) => map.set(k, FORK_COLORS[i % FORK_COLORS.length]));
      return map;
    }
    if (colorBy === "leaf") {
      leafColor.forEach((c, id) => map.set(id, c));
      return map;
    }
    if (!colorAt && colorBy === "rank") {
      Object.entries(RANK_COLORS).forEach(([k, c]) => map.set(k, c));
      return map;
    }
    const keys = [...new Set(nodes.map(groupKey).filter((k) => k !== null))].sort();
    keys.forEach((k, i) => map.set(k, CATEGORICAL[i % CATEGORICAL.length]));
    return map;
  }, [colorAt, colorBy, groups, leafColor, nodes, groupKey]);

  const colorOf = useCallback(
    (n) => {
      const k = groupKey(n);
      return (k !== null && groupColors.get(k)) || noLeafColor;
    },
    [groupKey, groupColors, noLeafColor],
  );

  const groupLabel = useCallback(
    (key) => {
      if (groups?.length) {
        return groups.find((group) => group.id === key)?.label ?? key;
      }
      if (colorAt) return treeNodes.get(key)?.label ?? key;
      if (colorBy === "leaf") return treeNodes.get(key)?.label ?? key;
      if (colorBy === "rank") return RANK_LABELS[key] ?? key;
      if (colorBy === "circuit") return `circuit ${key}`;
      if (colorBy === "judge_author_party") {
        return facetValueLabel("judge_author_party", key, caseIndex);
      }
      return key;
    },
    [colorBy, colorAt, groups, treeNodes, caseIndex],
  );

  return { groupKey, groupColors, colorOf, groupLabel };
}

/** Every edge between two drawn cases is shown. Shared by both views. */
export function makeEdgeVisible({ nodeVisible, byId }) {
  return (e) => nodeVisible(byId.get(e.s)) && nodeVisible(byId.get(e.t));
}
