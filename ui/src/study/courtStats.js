import { COURT_FACETS, asKey, courtFacetsOf, resolve } from "./caseKeys";
import { INK } from "./theme";

/** `facet:value`, the key format a selection is expressed in. */
export function facetKey(facetId, value) {
  return `${facetId}:${value}`;
}

/**
 * Per-node court statistics.
 *
 * @param {Array} nodes    `corpus_level.json` nodes
 * @param {object} caseIndex from buildCaseIndex
 * @returns {Map<string, {
 *   total: number, resolved: number, unresolved: number,
 *   byFacet: Object<string, Map<string, number>>,
 *   caseKeys: Array<{caseId: string, keys: string[]}>,
 * }>}
 */
export function buildNodeCourtStats(nodes, caseIndex) {
  const out = new Map();
  if (!Array.isArray(nodes)) return out;

  for (const node of nodes) {
    if (!node?.id) continue;
    const cases = Array.isArray(node.cases) ? node.cases : [];
    const byFacet = {};
    for (const facet of COURT_FACETS) byFacet[facet.id] = new Map();

    const caseKeys = [];
    let unresolved = 0;

    for (const raw of cases) {
      const caseId = asKey(raw);
      if (!caseId) continue;
      const row = resolve(caseIndex, { caseId });
      const facets = courtFacetsOf(row);
      const keys = [];
      for (const [facetId, value] of Object.entries(facets)) {
        const key = facetKey(facetId, value);
        keys.push(key);
        const counter = byFacet[facetId];
        counter.set(value, (counter.get(value) ?? 0) + 1);
      }
      if (keys.length === 0) unresolved += 1;
      caseKeys.push({ caseId, keys });
    }

    out.set(node.id, {
      total: caseKeys.length,
      resolved: caseKeys.length - unresolved,
      unresolved,
      byFacet,
      caseKeys,
    });
  }
  return out;
}

/** Shared shape for the selector: per facet, every value with its case count. */
function facetOptionsFromCounts(merged) {
  return COURT_FACETS.map((facet) => ({
    ...facet,
    values: [...merged[facet.id].entries()]
      .sort((a, b) => (b[1] - a[1]) || String(a[0]).localeCompare(String(b[0])))
      .map(([value, count]) => ({ value, count, key: facetKey(facet.id, value) })),
  }));
}

function emptyCounts() {
  const out = {};
  for (const facet of COURT_FACETS) out[facet.id] = new Map();
  return out;
}

/**
 * Option lists for the selector: every value present, with its case count, per
 * facet.
 *
 * `scopeId` is the node to count within — the root for corpus-wide counts, or
 * any interior node to condition on that branch. A node's `cases` list is
 * cumulative (the root holds every case, and children partition their parent),
 * so scoping is a lookup rather than a walk: the subtree's cases are exactly
 * the scope node's own list.
 */
export function courtFacetOptions(stats, scopeId) {
  const rootStats = scopeId ? stats.get(scopeId) : null;
  const merged = emptyCounts();

  const sources = rootStats ? [rootStats] : [...stats.values()];
  for (const entry of sources) {
    for (const facet of COURT_FACETS) {
      for (const [value, count] of entry.byFacet[facet.id]) {
        const counter = merged[facet.id];
        // Without a root the same case can be counted in several nodes; take the
        // max rather than the sum so the number is never absurdly inflated.
        counter.set(value, rootStats
          ? count
          : Math.max(counter.get(value) ?? 0, count));
      }
    }
  }

  return facetOptionsFromCounts(merged);
}

/**
 * How many of a node's cases match the selection. A case matching two selected
 * keys (say `circuit:9` and `court_level:federal_district`) counts once, which is
 * why this walks per-case key sets instead of adding up per-facet counters.
 */
export function matchInNode(entry, selection) {
  if (!entry || !selection || selection.size === 0) return null;
  let matched = 0;
  for (const { keys } of entry.caseKeys) {
    for (const key of keys) {
      if (selection.has(key)) {
        matched += 1;
        break;
      }
    }
  }
  return {
    matched,
    total: entry.total,
    share: entry.total ? matched / entry.total : 0,
    unresolved: entry.unresolved,
  };
}

/** This node's matching cases as a fraction of every case matching the selection. */
export function categoryShare(matched, categoryTotal) {
  return categoryTotal ? matched / categoryTotal : 0;
}

/**
 * How many cases in the tree match the selection. Read off the root when there
 * is one; without a root, the max over nodes so overlapping case lists are not
 * summed.
 */
export function selectionCategoryTotal(stats, selection, rootId) {
  if (!stats) return 0;
  const root = rootId ? stats.get(rootId) : null;
  if (root) return matchInNode(root, selection)?.matched ?? 0;
  let max = 0;
  for (const entry of stats.values()) {
    max = Math.max(max, matchInNode(entry, selection)?.matched ?? 0);
  }
  return max;
}

/** The matching case ids in a node — what the drawer opens on. */
export function matchingCaseIds(entry, selection) {
  if (!entry || !selection || selection.size === 0) return [];
  return entry.caseKeys
    .filter(({ keys }) => keys.some((key) => selection.has(key)))
    .map(({ caseId }) => caseId);
}

/**
 * Restrict a list of case ids to those matching the court selection.
 *
 * Empty selection is a no-op, so the drawer still shows everyone until a facet
 * is picked. Union semantics, same as `matchingCaseIds` / `matchInNode`: a case
 * matching any selected key is kept. Used for both tree-node drawers and the
 * unassigned-cases drawer, which has no `courtStats` entry.
 */
export function filterCaseIdsByCourtSelection(caseIds, selection, caseIndex) {
  const ids = [];
  const seen = new Set();
  for (const raw of caseIds ?? []) {
    const caseId = asKey(raw);
    if (!caseId || seen.has(caseId)) continue;
    seen.add(caseId);
    ids.push(caseId);
  }
  if (!selection || selection.size === 0) return ids;
  return ids.filter((caseId) => {
    const row = resolve(caseIndex, { caseId });
    const facets = courtFacetsOf(row);
    return Object.entries(facets).some(
      ([facetId, value]) => selection.has(facetKey(facetId, value)),
    );
  });
}

/**
 * Restrict a list of case ids to an explicit allow-list.
 *
 * `null` / `undefined` is a no-op (nothing has been selected). An empty Set
 * means a selection that matched nobody, so the result is empty — that is the
 * "this case has no citers" state, and must not be confused with no selection.
 */
export function filterCaseIdsByAllowlist(caseIds, allowed) {
  const ids = [];
  const seen = new Set();
  for (const raw of caseIds ?? []) {
    const caseId = asKey(raw);
    if (!caseId || seen.has(caseId)) continue;
    seen.add(caseId);
    ids.push(caseId);
  }
  if (allowed == null) return ids;
  return ids.filter((caseId) => allowed.has(caseId));
}

/**
 * The same selection applied to the chronology's cases.
 *
 * Returns the payload node indices that match, so the scene can dim the rest. Two
 * differences from the tree side, both deliberate:
 *
 *   - the corpus here is *every* swept case, not just the ones the tree run
 *     covered, so the totals differ from the root node's and are reported apart;
 *   - court metadata is taken from the run's case index when it has the case, and
 *     from the payload node itself otherwise. The payload carries `court_level`,
 *     `circuit`, `jurisdiction` and `court_id` for every node, so a case outside
 *     the tree run is still filterable.
 */
export function matchingCitationIndices(citationNodes, selection, caseIndex) {
  if (!selection || selection.size === 0) return null;
  const matched = new Set();
  let unresolved = 0;

  for (const node of citationNodes ?? []) {
    const row = resolve(caseIndex, {
      opinionId: node.opinion_id,
      caseId: node.case_id,
    });
    // `courtFacetsOf` reads camelCase; the payload is snake_case, so fall back to
    // a shaped object rather than duplicating the facet list.
    const facets = courtFacetsOf(row ?? {
      courtLevel: node.court_level,
      circuit: node.circuit,
      jurisdiction: node.jurisdiction,
      courtId: node.court_id,
      judgeAuthorParty: node.judge_author_party,
    });
    const keys = Object.entries(facets).map(([facetId, value]) => facetKey(facetId, value));
    if (keys.length === 0) {
      unresolved += 1;
      continue;
    }
    if (keys.some((key) => selection.has(key))) matched.add(node.i);
  }

  return {
    indices: matched,
    matched: matched.size,
    total: (citationNodes ?? []).length,
    unresolved,
  };
}

/** White -> teal ramp. The exponent keeps a small nonzero share visible. */
export function shareFill(share) {
  if (!(share > 0)) return null;
  const t = Math.pow(Math.min(1, share), 0.7);
  const mix = (from, to) => Math.round(from + (to - from) * t);
  return `rgb(${mix(255, 89)}, ${mix(255, 147)}, ${mix(255, 160)})`;
}

/** Text stays readable once the fill goes dark. */
export function shareTextColor(share) {
  return share > 0.55 ? "#f8fafc" : INK.dark;
}

