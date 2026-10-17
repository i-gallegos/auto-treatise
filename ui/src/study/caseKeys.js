const LAYER_LABELS = {
  run_manifest: "run manifest",
  citation_graph: "citation graph",
  current_db: "current DB",
  none: "none",
};

/** Identifiers are strings everywhere in this app. Blank becomes null. */
export function asKey(value) {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  return text === "" ? null : text;
}

/** Numeric fields: blank / non-numeric become null, not 0. */
function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function record(raw, source) {
  const caseId = asKey(raw.case_id ?? raw.caseId ?? raw.id);
  const opinionId = asKey(raw.opinion_id ?? raw.opinionId);
  if (!caseId && !opinionId) return null;
  return {
    caseId,
    opinionId,
    clusterId: asKey(raw.cluster_id ?? raw.clusterId),
    caseName: raw.case_name ?? raw.caseName ?? raw.name ?? null,
    decisionDate: raw.decision_date ?? raw.decisionDate ?? raw.date ?? null,
    courtId: raw.court_id ?? raw.courtId ?? null,
    courtName: raw.court_name ?? raw.courtName ?? null,
    courtAbbreviation:
      raw.court_abbreviation ?? raw.courtAbbreviation ?? raw.court ?? null,
    courtLevel: raw.court_level ?? raw.courtLevel ?? null,
    courtSystem: raw.court_system ?? raw.courtSystem ?? null,
    jurisdiction: raw.jurisdiction ?? null,
    circuit: raw.circuit === undefined ? null : raw.circuit,
    rank: raw.rank ?? null,
    // Times the opinion cites the seed, from the raw fetch DB. The annotation
    // sample is ranked on this, and the case drawer sorts the same way.
    citationDepth: finiteNumber(raw.citation_depth ?? raw.citationDepth),
    precedentialStatus:
      raw.precedential_status ?? raw.precedentialStatus ?? raw.status ?? null,
    judgeAuthorParty: raw.judge_author_party ?? raw.judgeAuthorParty ?? null,
    source,
  };
}

/** Merge `next` into `prev`, keeping whichever fields the winner actually has. */
function merge(prev, next) {
  if (!prev) return next;
  const out = { ...prev };
  for (const [key, value] of Object.entries(next)) {
    if (key === "source") continue;
    if (value !== null && value !== undefined) out[key] = value;
  }
  // `source` names the highest-precedence layer that supplied this case, which
  // is the one whose mapping is in force.
  out.source = next.source;
  return out;
}

const EMPTY_INDEX = {
  source: "none",
  sourceLabel: LAYER_LABELS.none,
  verified: false,
  byCaseId: new Map(),
  byOpinionId: new Map(),
  missingOpinionId: new Set(),
  duplicateOpinionIds: [],
  orphanCaseIds: new Set(),
  size: 0,
  dbRowCount: null,
  dbPath: null,
  generatedAt: null,
};

/**
 * Build the index. Every argument is optional; with none of them the result is a
 * well-formed empty index, so callers never branch on null.
 *
 * @param {object|null} runManifest   parsed `case_index.json`
 * @param {object|null} citationGraph parsed `citation_graph.json`
 * @param {object|null} dbIndex       payload from `/api/db-browser/case-index`
 */
export function buildCaseIndex({ runManifest, citationGraph, dbIndex } = {}) {
  const layers = [
    ["current_db", dbIndex?.cases],
    ["citation_graph", citationGraph?.nodes],
    ["run_manifest", runManifest?.cases],
  ].filter(([, rows]) => Array.isArray(rows) && rows.length > 0);

  if (layers.length === 0) return EMPTY_INDEX;

  const byCaseId = new Map();
  const byOpinionId = new Map();
  let source = "none";

  for (const [layer, rows] of layers) {
    source = layer; // layers are in ascending precedence
    for (const raw of rows) {
      const parsed = record(raw, layer);
      if (!parsed) continue;
      const existing = parsed.caseId
        ? byCaseId.get(parsed.caseId)
        : byOpinionId.get(parsed.opinionId);
      const merged = merge(existing, parsed);
      if (merged.caseId) byCaseId.set(merged.caseId, merged);
      if (merged.opinionId) byOpinionId.set(merged.opinionId, merged);
    }
  }

  const missingOpinionId = new Set();
  const opinionCases = new Map();
  for (const [caseId, row] of byCaseId) {
    if (!row.opinionId) {
      missingOpinionId.add(caseId);
      continue;
    }
    if (!opinionCases.has(row.opinionId)) opinionCases.set(row.opinionId, []);
    opinionCases.get(row.opinionId).push(caseId);
  }

  const manifest = source === "run_manifest" ? runManifest : null;
  return {
    source,
    sourceLabel: LAYER_LABELS[source] ?? source,
    verified: source === "run_manifest",
    byCaseId,
    byOpinionId,
    missingOpinionId,
    duplicateOpinionIds: [...opinionCases.entries()]
      .filter(([, ids]) => ids.length > 1)
      .sort(),
    orphanCaseIds: new Set(
      (manifest?.orphan_case_ids ?? runManifest?.orphan_case_ids ?? []).map(asKey)
        .filter(Boolean),
    ),
    size: byCaseId.size,
    dbRowCount:
      manifest?.meta?.db_row_count ?? dbIndex?.meta?.db_row_count ?? null,
    dbPath: manifest?.meta?.db_path ?? dbIndex?.meta?.db_path ?? null,
    generatedAt: manifest?.meta?.generated_at ?? null,
  };
}

/** Look a case up by either id. Returns null rather than throwing. */
export function resolve(index, { caseId, opinionId } = {}) {
  if (!index) return null;
  const oid = asKey(opinionId);
  if (oid && index.byOpinionId.has(oid)) return index.byOpinionId.get(oid);
  const cid = asKey(caseId);
  if (cid && index.byCaseId.has(cid)) return index.byCaseId.get(cid);
  return null;
}

export function opinionIdFor(index, caseId) {
  return resolve(index, { caseId })?.opinionId ?? null;
}

export function caseIdFor(index, opinionId) {
  return resolve(index, { opinionId })?.caseId ?? null;
}

/**
 * Query parameters for `/api/db-browser/case`, preferring the permanent id and
 * falling back to the rowid — which is the only handle for a row that has no
 * opinion id at all.
 */
export function caseQueryParams(index, { caseId, opinionId } = {}) {
  const row = resolve(index, { caseId, opinionId });
  const params = {};
  const oid = row?.opinionId ?? asKey(opinionId);
  const cid = row?.caseId ?? asKey(caseId);
  if (oid) params.opinion_id = oid;
  if (cid) params.id = cid;
  return params;
}

/**
 * The facets the Tree tab's highlight panel groups by.
 *
 * The judge facets are written by `descriptive_findings.judge_party`. A `?`
 * suffix on a panel value means at least one panel member could not be matched
 * to the FJC directory, so the composition shown is of the members that were.
 * That distinction lives inside the value rather than in a facet of its own
 * because selection here is a pure OR union (see `matchInNode` in courtStats):
 * a separate "coverage" facet could never be intersected with this one.
 * Selecting the three unsuffixed values is how you ask for fully resolved
 * panels only.
 */
export const COURT_FACETS = [
  { id: "court_level", label: "Court level", field: "courtLevel" },
  { id: "circuit", label: "Circuit", field: "circuit" },
  { id: "jurisdiction", label: "Jurisdiction", field: "jurisdiction" },
  { id: "court_id", label: "Court", field: "courtId" },
  { id: "judge_author_party", label: "Author appointing party", field: "judgeAuthorParty" },
];

/** `{facet: value}` for one case; missing values are omitted, not stringified. */
export function courtFacetsOf(row) {
  if (!row) return {};
  const out = {};
  for (const facet of COURT_FACETS) {
    const value = row[facet.field];
    if (value !== null && value !== undefined && String(value).trim() !== "") {
      out[facet.id] = String(value);
    }
  }
  return out;
}

const AUTHOR_PARTY_LABELS = {
  R: "Republican appointee",
  D: "Democratic appointee",
  O: "other party",
  mixed: "mixed appointing parties",
  non_article_iii: "not an Article III court (no appointing president)",
  unknown: "unknown",
};

/** Human label for a facet value, e.g. `circuit:9` -> "9th Circuit". */
export function facetValueLabel(facetId, value, index) {
  if (facetId === "circuit") {
    if (value === "dc") return "D.C. Circuit";
    if (value === "fed") return "Federal Circuit";
    const n = Number(value);
    if (Number.isFinite(n)) {
      const suffix = n === 1 ? "st" : n === 2 ? "nd" : n === 3 ? "rd" : "th";
      return `${n}${suffix} Circuit`;
    }
    return `circuit ${value}`;
  }
  if (facetId === "court_level") return String(value).replace(/_/g, " ");
  if (facetId === "judge_author_party") return AUTHOR_PARTY_LABELS[value] ?? value;
  if (facetId === "court_id") {
    for (const row of index?.byCaseId?.values() ?? []) {
      if (row.courtId === value) {
        return row.courtAbbreviation || row.courtName || value;
      }
    }
    return value;
  }
  return value;
}

/** Fetch the fallback map for a doctrine. Resolves to null on any failure. */
export async function fetchDbCaseIndex(doctrine) {
  if (!doctrine) return null;
  try {
    const res = await fetch(
      `/api/db-browser/case-index?${new URLSearchParams({ doctrine })}`,
      { cache: "no-store" },
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
