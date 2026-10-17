import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { caseQueryParams, resolve as resolveCase } from "./caseKeys";
import { GREEN, INK, ORANGE, TEAL } from "./theme";
import {
  SymbolAnalysisCard,
  MajorityBoundaryChip,
  MajorityBoundaryDivider,
  MajorityParagraphBody,
} from "./CaseDrawerParts";
import {
  parseParagraphsExport,
  parseSegmentationExport,
  parseRelevanceExport,
  parseMajorityBoundaryExport,
  parseCaseSymbolProfileExport,
  primaryCitation,
} from "./caseParsers";
// Attribution colours and sentence collapsing, shared by every tab that renders
// segmented text. This file previously carried its own identical copy.
import { ATTRIBUTION_STYLES, collapseSegments, getAttributionStyle } from "./segmentation";
import { TreeLineageBox } from "./TreeLineageBox";
import { assignmentFromWalk, walkRunLineage } from "./treeLineage";
import { displayNodeId } from "./nodeIds";

/**
 * Each judge on the panel.
 *
 * The facet collapses a panel to one value, and a "?" on that value only says
 * *that* something went unmatched. This says *who*, and why — which is the
 * difference between a caveat and something you can check. `match_method` is
 * shown verbatim rather than prettified: `unmatched_not_in_fjc` on a magistrate
 * judge is the expected outcome, not a defect, and the raw string is what the
 * judge_party.md report tallies.
 */
function JudgePanelBox({ payload }) {
  const judges = useMemo(() => {
    if (!payload) return null;
    try {
      const parsed = JSON.parse(payload);
      const judges = Array.isArray(parsed) ? parsed : parsed?.judges;
      return Array.isArray(judges) && judges.length ? judges : null;
    } catch {
      return null;
    }
  }, [payload]);

  if (!judges) return null;

  const partyColor = (party) =>
    party === "R" ? ORANGE.shade1 : party === "D" ? TEAL.shade1 : INK.muted;

  return (
    <div
      style={{
        marginBottom: 8,
        padding: "8px 10px",
        borderRadius: 6,
        border: `1px solid ${INK.rule}`,
        background: INK.wash,
        fontSize: 10,
        lineHeight: 1.4,
        display: "grid",
        gap: 4,
      }}
    >
      <div style={{ fontWeight: 700, color: INK.dark }}>
        Panel · appointing party
      </div>
      {judges.map((judge, i) => (
        <div
          key={`${judge.last_name}-${i}`}
          style={{ display: "flex", gap: 6, alignItems: "baseline" }}
        >
          <span style={{ color: partyColor(judge.party), fontWeight: 700, minWidth: 14 }}>
            {judge.party ?? "—"}
          </span>
          <span style={{ color: INK.dark }}>
            {judge.fjc_name || `${judge.first_name} ${judge.last_name}`.trim()}
          </span>
          {judge.role && judge.role !== "unknown" && (
            <span style={{ color: INK.muted }}>· {judge.role}</span>
          )}
          <span
            style={{
              color: INK.muted,
              marginLeft: "auto",
              fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
            }}
            title={
              judge.appointing_president
                ? `appointed by ${judge.appointing_president}`
                : "no FJC match"
            }
          >
            {judge.match_method}
          </span>
        </div>
      ))}
    </div>
  );
}

function buildNodeIndex(nodes) {
  if (!Array.isArray(nodes)) return {};
  const idx = {};
  for (const n of nodes) {
    if (n.lhs_id && !idx[n.lhs_id]) idx[n.lhs_id] = n;
    if (n.id && !idx[n.id]) idx[n.id] = n;
  }
  return idx;
}

const CONFIDENCE_STYLES = {
  high:   { color: GREEN.shade2, background: "rgba(133, 165, 95, 0.12)",  border: "rgba(133, 165, 95, 0.35)"  },
  medium: { color: ORANGE.shade1, background: "rgba(209, 110, 76, 0.12)",  border: "rgba(209, 110, 76, 0.35)"  },
  low:    { color: ORANGE.shade2, background: "rgba(115, 49, 28, 0.12)",  border: "rgba(115, 49, 28, 0.35)"  },
};

function ConfidenceChip({ confidence, compact = false }) {
  if (!confidence) return null;
  const key = String(confidence).toLowerCase();
  const style = CONFIDENCE_STYLES[key] ?? {
    color: INK.muted,
    background: "rgba(71, 85, 105, 0.1)",
    border: "rgba(71, 85, 105, 0.3)",
  };
  return (
    <span
      className={`study-confidence-chip${compact ? " study-confidence-chip--compact" : ""}`}
      style={{
        color: style.color,
        background: style.background,
        border: `1px solid ${style.border}`,
      }}
      title={`Assignment confidence: ${key}`}
    >
      {key}
    </span>
  );
}

/**
 * Case-level split confidence lives on the child via inbound_assignments
 * (how the parent split placed the case here). Root has none.
 * Falls back to parent.assignments for older corpus_level.json files.
 */
function getCaseAssignmentConfidence(treeEntry, caseId, nodeIndex) {
  if (!treeEntry || caseId == null) return null;
  const id = String(caseId);
  const inbound = treeEntry.inbound_assignments?.[id]?.confidence;
  if (inbound) return String(inbound);
  const parentId = treeEntry.parent_id;
  if (!parentId || !nodeIndex) return null;
  const parent = nodeIndex[parentId];
  const fromParent = parent?.assignments?.[id]?.confidence;
  return fromParent ? String(fromParent) : null;
}

const CONFIDENCE_SORT_RANK = { high: 0, medium: 1, low: 2 };

/** Missing ranks last on a descending key, the way the annotation sample does. */
function descendingMetric(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : Number.NEGATIVE_INFINITY;
}

/**
 * Same ranking the annotation panel lists by: citation depth (times the opinion
 * cites the seed), then court authority, then assignment confidence.
 */
function sortCaseIds(caseIds, treeEntry, nodeIndex, caseIndex) {
  return [...caseIds].sort((a, b) => {
    const rowA = resolveCase(caseIndex, { caseId: a });
    const rowB = resolveCase(caseIndex, { caseId: b });
    const depthDelta = descendingMetric(rowB?.citationDepth) - descendingMetric(rowA?.citationDepth);
    if (depthDelta) return depthDelta;
    const rankDelta = descendingMetric(rowB?.rank) - descendingMetric(rowA?.rank);
    if (rankDelta) return rankDelta;
    const ca = String(getCaseAssignmentConfidence(treeEntry, a, nodeIndex) || "").toLowerCase();
    const cb = String(getCaseAssignmentConfidence(treeEntry, b, nodeIndex) || "").toLowerCase();
    const ra = ca in CONFIDENCE_SORT_RANK ? CONFIDENCE_SORT_RANK[ca] : 3;
    const rb = cb in CONFIDENCE_SORT_RANK ? CONFIDENCE_SORT_RANK[cb] : 3;
    if (ra !== rb) return ra - rb;
    return String(a).localeCompare(String(b), undefined, { numeric: true });
  });
}

function reasonLabel(reason) {
  const raw = String(reason || "unassigned");
  return raw.replace(/_/g, " ");
}

function groupIdsByReason(ids, caseAnnotations) {
  if (!caseAnnotations) return [{ reason: null, ids }];
  const groups = [];
  const indexByReason = {};
  for (const id of ids) {
    const reason = caseAnnotations[id]?.reason || "unassigned";
    if (indexByReason[reason] == null) {
      indexByReason[reason] = groups.length;
      groups.push({ reason, ids: [] });
    }
    groups[indexByReason[reason]].ids.push(id);
  }
  return groups;
}

// ── Component ──────────────────────────────────────────────────────────────────
const MIN_META_HEIGHT = 60;
const MIN_PARAGRAPHS_HEIGHT = 200;

export function DoctrineCaseDrawer({
  open,
  onClose,
  doctrine,
  nodeLabel,
  treeEntry,
  caseIds,
  doctrineGraph,
  cfgRecords = [],
  caseIndex = null,
  // Row clicks stay in this drawer. CfgView uses this to highlight the tree
  // node an unassigned case belonged to; it must not pin the chronology, or
  // the citer filter would replace the list being browsed.
  onCaseSelect = null,
  /** Optional per-id notes (reason, explanation, nodeId) — used for unassigned cases. */
  caseAnnotations = null,
}) {
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  const [dbCase, setDbCase] = useState(null);
  const [loadingCase, setLoadingCase] = useState(false);
  /** case id (string) -> case_name from DB */
  const [caseNamesById, setCaseNamesById] = useState({});
  /** case id (string) -> primary reporter cite from DB */
  const [citationsById, setCitationsById] = useState({});
  const [nodeDetailsOpen, setNodeDetailsOpen] = useState(true);
  const [metaHeight, setMetaHeight] = useState(160);
  const [resizingMeta, setResizingMeta] = useState(false);
  const metaDragRef = useRef(null);
  const detailSectionRef = useRef(null);

  useEffect(() => {
    if (!resizingMeta) return undefined;
    const onMove = (ev) => {
      const drag = metaDragRef.current;
      const room = detailSectionRef.current.clientHeight - MIN_PARAGRAPHS_HEIGHT;
      const next = drag.height + ev.clientY - drag.y;
      setMetaHeight(Math.min(Math.max(next, MIN_META_HEIGHT), Math.max(room, MIN_META_HEIGHT)));
    };
    const onUp = () => setResizingMeta(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizingMeta]);
  const drawerRef = useRef(null);
  const closeBtnRef = useRef(null);

  const rawIds = (Array.isArray(caseIds) ? caseIds : []).map((id) => String(id));
  const rawIdsKey = rawIds.join(",");
  const treeNodeIndex = useMemo(() => {
    if (!doctrineGraph || !Array.isArray(doctrineGraph.nodes)) return {};
    const idx = {};
    for (const n of doctrineGraph.nodes) {
      if (n?.id) idx[n.id] = n;
    }
    return idx;
  }, [doctrineGraph]);
  const ids = useMemo(
    () => sortCaseIds(rawIds, treeEntry, treeNodeIndex, caseIndex),
    [rawIdsKey, treeEntry, treeNodeIndex, caseIndex]
  );
  const idsKey = ids.join(",");
  const listGroups = useMemo(
    () => groupIdsByReason(ids, caseAnnotations),
    [idsKey, caseAnnotations]
  );

  // Select first case when opening (deepest citation after sort)
  useEffect(() => {
    if (!open || !ids.length) return;
    setSelectedCaseId((prev) => (prev != null && ids.includes(prev) ? prev : ids[0]));
  }, [open, idsKey]);

  useEffect(() => {
    if (!open) {
      setDbCase(null);
      setLoadingCase(false);
      setCaseNamesById({});
      setCitationsById({});
    }
  }, [open]);

  // Prefetch case_name for every case in the list (so rows show names before click)
  useEffect(() => {
    if (!open || !doctrine || !ids.length) return;
    let cancelled = false;
    Promise.all(
      ids.map((caseId) => {
        // Addressed by opinion_id when the index has one: the tree's case ids
        // belong to the database as it was at induction time.
        const query = new URLSearchParams({
          doctrine,
          ...caseQueryParams(caseIndex, { caseId }),
        });
        return fetch(`/api/db-browser/case?${query.toString()}`, { cache: "no-store" })
          .then((res) => (res.ok ? res.json() : null))
          .then((payload) => {
            return {
              id: caseId,
              caseName: payload?.case?.case_name ?? null,
              citation: primaryCitation(payload?.case?.citation),
            };
          });
      })
    ).then((rows) => {
      if (cancelled) return;
      const nextNames = {};
      const nextCites = {};
      for (let i = 0; i < rows.length; i += 1) {
        const row = rows[i];
        if (row.caseName) nextNames[row.id] = row.caseName;
        if (row.citation) nextCites[row.id] = row.citation;
      }
      setCaseNamesById(nextNames);
      setCitationsById(nextCites);
    });
    return () => { cancelled = true; };
  }, [open, doctrine, idsKey, caseIndex]);

  // Fetch full DB row
  useEffect(() => {
    if (!open || !doctrine || selectedCaseId == null) { setDbCase(null); return; }
    let cancelled = false;
    setLoadingCase(true);
    setDbCase(null);
    const query = new URLSearchParams({
      doctrine,
      ...caseQueryParams(caseIndex, { caseId: selectedCaseId }),
    });
    fetch(`/api/db-browser/case?${query.toString()}`, { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        if (cancelled) return;
        const row = payload?.case ?? null;
        setDbCase(row);
        if (row?.case_name && selectedCaseId != null) {
          setCaseNamesById((prev) => ({
            ...prev,
            [selectedCaseId]: row.case_name,
          }));
        }
      })
      .catch(() => { if (!cancelled) setDbCase(null); })
      .finally(() => { if (!cancelled) setLoadingCase(false); });
    return () => { cancelled = true; };
  }, [open, doctrine, selectedCaseId, caseIndex]);

  // Keyboard / scroll-lock
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const t = requestAnimationFrame(() => closeBtnRef.current?.focus());
    return () => cancelAnimationFrame(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (e.key === "Escape") { onClose(); e.preventDefault(); return; }
      if (e.key !== "Tab" || !drawerRef.current) return;
      const focusables = Array.from(drawerRef.current.querySelectorAll(
        'button:not([disabled]),[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'
      ));
      if (!focusables.length) return;
      const first = focusables[0], last = focusables[focusables.length - 1];
      if (e.shiftKey) { if (document.activeElement === first) { last.focus(); e.preventDefault(); } }
      else if (document.activeElement === last) { first.focus(); e.preventDefault(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // ── Derived data ─────────────────
  const paragraphList = useMemo(
    () => parseParagraphsExport(dbCase?.opinion_paragraphs_clean_json ?? null),
    [dbCase]
  );
  const relevancePayload = useMemo(
    () => parseRelevanceExport(dbCase?.paragraph_relevance_cache_json ?? null),
    [dbCase]
  );
  const segmentationPayload = useMemo(
    () => parseSegmentationExport(dbCase?.paragraph_sentence_segmentation_cache_json ?? null),
    [dbCase]
  );
  const relevantParagraphIndices = useMemo(() => {
    if (!relevancePayload?.relevant_paragraphs) return new Set();
    return new Set(
      relevancePayload.relevant_paragraphs
        .map((e) => Number(e.paragraph_index))
        .filter((i) => Number.isFinite(i))
    );
  }, [relevancePayload]);
  const segmentationByParagraph = useMemo(() => {
    const m = {};
    for (const row of segmentationPayload) {
      const idx = Number(row.paragraph_index);
      if (Number.isFinite(idx)) m[idx] = Array.isArray(row.sentences) ? row.sentences : [];
    }
    return m;
  }, [segmentationPayload]);
  const majorityBoundary = useMemo(
    () => parseMajorityBoundaryExport(dbCase?.majority_boundary_json ?? null),
    [dbCase]
  );
  const truncatedAtIndex =
    majorityBoundary && majorityBoundary.status === "boundary_found" && majorityBoundary.endIndex !== null
      ? majorityBoundary.endIndex
      : null;

  // CFG profile from the DB column (same as DB browser)
  const caseSymbolProfile = useMemo(() => {
    const dbProfile = parseCaseSymbolProfileExport(dbCase?.case_symbol_profile_json ?? null);
    if (dbProfile) return dbProfile;
    if (!Array.isArray(cfgRecords) || selectedCaseId == null) return null;
    for (let i = 0; i < cfgRecords.length; i += 1) {
      const row = cfgRecords[i];
      if (!row || typeof row !== "object") continue;
      if (String(row.id ?? row.case_id ?? "") !== selectedCaseId) continue;
      return parseCaseSymbolProfileExport(row.profile ?? null);
    }
    return null;
  }, [dbCase, cfgRecords, selectedCaseId]);
  const cfgNodeIndex = useMemo(() => buildNodeIndex(doctrineGraph?.nodes ?? []), [doctrineGraph]);

  const caseName = dbCase?.case_name ?? String(selectedCaseId ?? "");
  const selectedConfidence = useMemo(
    () => getCaseAssignmentConfidence(treeEntry, selectedCaseId, treeNodeIndex),
    [treeEntry, selectedCaseId, treeNodeIndex]
  );
  const treeEntryDescription = treeEntry?.description ?? treeEntry?.note ?? "";
  const treeEntrySplitCriterion = treeEntry?.split_criterion ?? "";
  const treeEntryReasoningChain = treeEntry?.reasoning_chain ?? "";
  const treeEntryLeftDescription = treeEntry?.left_description ?? "";
  const treeEntryRightDescription = treeEntry?.right_description ?? "";
  const selectedAnnotation = selectedCaseId != null && caseAnnotations
    ? caseAnnotations[selectedCaseId] ?? null
    : null;
  const selectedAssignment = useMemo(() => {
    if (selectedCaseId == null) return null;
    return assignmentFromWalk(walkRunLineage(doctrineGraph, selectedCaseId));
  }, [doctrineGraph, selectedCaseId]);

  if (!open) return null;

  const titleId = "study-case-drawer-title";

  const panel = (
    <>
      <div className="study-case-drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <div
        ref={drawerRef}
        className="study-theme study-case-drawer-panel study-case-drawer-panel--db"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        {/* ── Header ── */}
        <div className="study-case-drawer-head">
          <div>
            <div id={titleId} className="study-case-drawer-title">
              {nodeLabel ?? "Cases"}
            </div>
            {treeEntryDescription && (
              <div className="study-case-drawer-sub" style={{ opacity: 0.7 }}></div>
            )}
          </div>
          <button ref={closeBtnRef} type="button" className="study-case-drawer-close" onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>

        {/* ── Body ── */}
        <div className="study-case-drawer-body--db">

          {/* Left: case list */}
          <section className="dbb-case-panel dbb-case-panel--drawer">
            <div className="dbb-case-panel-top">
              <div style={{ fontSize: 15, fontWeight: 700 }}>
                Opinions
                <span className="dbb-count-chip" style={{ marginLeft: 6 }}>{ids.length}</span>
              </div>
            </div>
            <div className="dbb-case-list">
              {ids.length === 0 && <div className="dbb-empty-list">No cases.</div>}
              {listGroups.map((group) => (
                <div key={group.reason ?? "__all__"} className={group.reason ? "study-unassigned-group" : undefined}>
                  {group.reason && (
                    <div className="study-unassigned-group-head">
                      <span className="study-unassigned-reason-chip">
                        {reasonLabel(group.reason)}
                      </span>
                      <span className="study-unassigned-group-count">{group.ids.length}</span>
                    </div>
                  )}
                  {group.ids.map((caseId) => {
                    const isSelected = selectedCaseId === caseId;
                    const resolved = resolveCase(caseIndex, { caseId });
                    const rowName = caseNamesById[caseId] || resolved?.caseName || caseId;
                    const confidence = getCaseAssignmentConfidence(treeEntry, caseId, treeNodeIndex);
                    const citationDepth = resolved?.citationDepth;
                    const rowCite = citationsById[caseId];
                    const annotation = caseAnnotations ? caseAnnotations[caseId] : null;
                    return (
                      <button
                        key={caseId}
                        type="button"
                        className={`dbb-case-row${isSelected ? " selected" : ""}`}
                        onClick={() => {
                          setSelectedCaseId(caseId);
                          onCaseSelect?.(caseId);
                        }}
                      >
                        <div className="dbb-case-row-title-line">
                          <span className="dbb-case-row-title">{rowName}</span>
                          {isSelected && <span className="dbb-case-selected-marker">Selected</span>}
                        </div>
                        <div className="dbb-case-row-meta">
                          <span title="case_id (sqlite rowid) · opinion_id (CourtListener)">
                            id: {caseId}
                            {resolved?.opinionId ? ` · op ${resolved.opinionId}` : ""}
                          </span>
                          {resolved?.courtAbbreviation && (
                            <>
                              <span className="dbb-case-row-meta-sep"> · </span>
                              <span>{resolved.courtAbbreviation}</span>
                            </>
                          )}
                          {rowCite && (
                            <>
                              <span className="dbb-case-row-meta-sep"> · </span>
                              <span title="Reporter citation">{rowCite}</span>
                            </>
                          )}
                          {Number.isFinite(citationDepth) && (
                            <>
                              <span className="dbb-case-row-meta-sep"> · </span>
                              <span
                                className="ann-cites on-basis"
                                title={`Cites the seed ${citationDepth} time(s)`}
                              >
                                depth {citationDepth}
                              </span>
                            </>
                          )}
                          {confidence && (
                            <>
                              <span className="dbb-case-row-meta-sep"> · </span>
                              <ConfidenceChip confidence={confidence} compact />
                            </>
                          )}
                          {annotation?.nodeId && (
                            <>
                              <span className="dbb-case-row-meta-sep"> · </span>
                              <span>
                                node {displayNodeId(annotation.nodeId)}
                                {annotation.isLeaf ? " · leaf" : ""}
                              </span>
                            </>
                          )}
                        </div>
                        {annotation?.explanation && (
                          <div className="dbb-case-row-note">{annotation.explanation}</div>
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          </section>

          {/* Right: 3-panel detail */}
          <section className="study-case-drawer-detail-section" ref={detailSectionRef}>
            <div className="dbb-detail-headline">
              <div className="dbb-detail-casename">
                {selectedCaseId != null ? caseName || "Case detail" : "Select a case"}
              </div>
            </div>
            <div className="study-case-drawer-detail-meta" style={{ height: metaHeight }}>

            {selectedAnnotation && (
              <div
                style={{
                  marginBottom: 8,
                  padding: "8px 10px",
                  borderRadius: 6,
                  border: "1px solid rgba(194, 65, 12, 0.28)",
                  background: "rgba(194, 65, 12, 0.06)",
                  fontSize: 10,
                  lineHeight: 1.4,
                  display: "grid",
                  gap: 6,
                }}
              >
                <div style={{ fontSize: 10, fontWeight: 700, color: ORANGE.shade2 }}>
                  Unassigned during tree induction
                </div>
                {selectedAnnotation.reason && (
                  <div>
                    <span className="study-unassigned-reason-chip">
                      {reasonLabel(selectedAnnotation.reason)}
                    </span>
                  </div>
                )}
                {selectedAnnotation.explanation && (
                  <div>{selectedAnnotation.explanation}</div>
                )}
                {selectedAnnotation.nodeId && (
                  <div style={{ fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif' }}>
                    node {displayNodeId(selectedAnnotation.nodeId)}
                    {selectedAnnotation.isLeaf ? " · leaf" : ""}
                  </div>
                )}
              </div>
            )}

            <JudgePanelBox payload={dbCase?.judge_panel_json} />

            <TreeLineageBox
              assignment={selectedAssignment}
              className="study-case-drawer-lineage"
            />

            {(treeEntryDescription || treeEntrySplitCriterion || treeEntryReasoningChain) && (
              <div className="study-case-drawer-node-details">
                <button
                  type="button"
                  className="study-case-drawer-node-details-toggle"
                  aria-expanded={nodeDetailsOpen}
                  onClick={() => setNodeDetailsOpen((open) => !open)}
                >
                  <span>Corpus-level node details</span>
                  <span className="study-case-drawer-node-details-hint">
                    {nodeDetailsOpen ? "Hide" : "Show"}
                  </span>
                </button>
                {nodeDetailsOpen && (
                  <div className="study-case-drawer-node-details-body">
                    {treeEntryDescription && (
                      <div>
                        <span style={{ fontWeight: 700, marginRight: 4 }}>Description:</span>
                        {treeEntryDescription}
                      </div>
                    )}
                    {treeEntrySplitCriterion && (
                      <div>
                        <span style={{ fontWeight: 700, marginRight: 4 }}>Split criterion:</span>
                        {treeEntrySplitCriterion}
                      </div>
                    )}
                    {treeEntryReasoningChain && (
                      <div>
                        <span style={{ fontWeight: 700, marginRight: 4 }}>Reasoning:</span>
                        <span style={{ whiteSpace: "pre-wrap" }}>{treeEntryReasoningChain}</span>
                      </div>
                    )}
                    {(treeEntryLeftDescription || treeEntryRightDescription) && (
                      <div style={{ display: "grid", gap: 4 }}>
                        {treeEntryLeftDescription && (
                          <div>
                            <span style={{ fontWeight: 700, marginRight: 4 }}>Left cluster:</span>
                            {treeEntryLeftDescription}
                          </div>
                        )}
                        {treeEntryRightDescription && (
                          <div>
                            <span style={{ fontWeight: 700, marginRight: 4 }}>Right cluster:</span>
                            {treeEntryRightDescription}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            </div>
            <div
              className={`study-case-drawer-resize${resizingMeta ? " dragging" : ""}`}
              role="separator"
              aria-orientation="horizontal"
              title="drag to resize"
              onMouseDown={(ev) => {
                metaDragRef.current = { y: ev.clientY, height: metaHeight };
                setResizingMeta(true);
              }}
            />

            {selectedCaseId == null ? (
              <div className="study-case-drawer-empty">Select a case from the list.</div>
            ) : (
              <div className="study-case-drawer-3col">
                {/* ── Panel 1: Clean paragraphs with relevance + segmentation ── */}
                <div className="dbb-opinion-panel">
                  <div className="dbb-opinion-panel-head">
                    Clean paragraphs (`opinion_paragraphs_clean_json`)
                  </div>
                  {!loadingCase && <MajorityBoundaryChip boundary={majorityBoundary} />}
                  {!loadingCase && relevancePayload && (
                    <div style={{ fontSize: 10, marginBottom: 6, opacity: 0.85 }}>
                      Highlighting {relevantParagraphIndices.size} relevant paragraph(s)
                      {relevancePayload.query ? ` for query: "${relevancePayload.query}"` : ""}.
                    </div>
                  )}
                  {!loadingCase && segmentationPayload.length > 0 && (
                    <div style={{ fontSize: 10, marginBottom: 6, opacity: 0.85, display: "grid", gap: 4 }}>
                      <div>
                        Sentence attribution labels loaded for {segmentationPayload.length} relevant paragraph(s).
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {Object.values(ATTRIBUTION_STYLES).map((style) => (
                          <span
                            key={style.label}
                            style={{
                              fontSize: 10,
                              padding: "2px 6px",
                              borderRadius: 999,
                              color: style.color,
                              background: style.background,
                              border: `1px solid ${style.border}`,
                            }}
                          >
                            {style.label}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="dbb-opinion-scrollbox">
                    {loadingCase
                      ? "Loading case..."
                      : paragraphList.length === 0
                        ? (
                          <div style={{ fontSize: 10, opacity: 0.75 }}>
                            No clean paragraphs found.
                          </div>
                        )
                        : (
                          <div style={{ display: "grid", gap: 6 }}>
                            {paragraphList.map((paragraph, idx) => {
                              const paragraphNumber = idx + 1;
                              const isBoundaryParagraph = truncatedAtIndex !== null && idx === truncatedAtIndex;
                              const isPastBoundary = truncatedAtIndex !== null && idx > truncatedAtIndex;
                              const isCut = isPastBoundary || (isBoundaryParagraph && !majorityBoundary.splitBoundary);
                              const isRelevant = relevantParagraphIndices.has(idx);
                              const segmentedSentences = segmentationByParagraph[idx] || [];
                              return (
                                <div key={`${paragraphNumber}-${paragraph.slice(0, 16)}`}>
                                  {isBoundaryParagraph && !majorityBoundary.splitBoundary && (
                                    <MajorityBoundaryDivider marker={majorityBoundary.marker} />
                                  )}
                                  <div
                                    style={{
                                      fontSize: 10,
                                      lineHeight: 1.35,
                                      borderLeft: isRelevant
                                        ? `3px solid ${GREEN.base}`
                                        : isBoundaryParagraph
                                          ? `3px solid ${ORANGE.base}`
                                          : "1px solid transparent",
                                      paddingLeft: isRelevant || isBoundaryParagraph ? 8 : 10,
                                      background: isRelevant
                                        ? "rgba(133, 165, 95, 0.12)"
                                        : isBoundaryParagraph
                                          ? "rgba(209, 110, 76, 0.1)"
                                          : "transparent",
                                    }}
                                  >
                                    <div style={{ fontSize: 10, opacity: isCut ? 0.4 : 0.75, marginBottom: 2 }}>
                                      [{paragraphNumber}]
                                      {isRelevant ? "  ← Relevant paragraph" : ""}
                                      {isCut ? "  ← Separate opinion, not sent to models" : ""}
                                    </div>
                                    <MajorityParagraphBody
                                      paragraph={paragraph}
                                      boundary={majorityBoundary}
                                      isBoundaryParagraph={isBoundaryParagraph}
                                      isPastBoundary={isPastBoundary}
                                    />
                                    {isRelevant && segmentedSentences.length > 0 && (
                                      <div style={{ marginTop: 6, display: "grid", gap: 4 }}>
                                        {collapseSegments(segmentedSentences).map((sentence, si) => {
                                          const style = getAttributionStyle(sentence.attribution);
                                          return (
                                            <div
                                              key={si}
                                              style={{
                                                fontSize: 10,
                                                lineHeight: 1.35,
                                                padding: "4px 6px",
                                                borderRadius: 4,
                                                background: style.background,
                                                border: `1px solid ${style.border}`,
                                              }}
                                            >
                                              <span style={{ fontWeight: 700, marginRight: 6, color: style.color }}>
                                                {style.label}
                                              </span>
                                              <span>{sentence.text || ""}</span>
                                            </div>
                                          );
                                        })}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        )}
                  </div>
                </div>

                {/* ── Panel 2: CFG symbol profile ── */}
                <div className="dbb-opinion-panel">
                  <div className="dbb-opinion-panel-head" style={{ marginBottom: 4, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    CFG symbol profile
                    {caseSymbolProfile?.output_type && (
                      <span
                        style={{
                          fontSize: 10,
                          padding: "2px 7px",
                          borderRadius: 999,
                          background: "rgba(99, 102, 241, 0.12)",
                          border: "1px solid rgba(99, 102, 241, 0.35)",
                          color: TEAL.shade1,
                          fontWeight: 700,
                        }}
                      >
                        {caseSymbolProfile.output_type}
                      </span>
                    )}
                    {selectedConfidence && (
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 10, color: INK.muted }}>
                        confidence
                        <ConfidenceChip confidence={selectedConfidence} />
                      </span>
                    )}
                  </div>
                  <div className="dbb-opinion-scrollbox">
                    {loadingCase
                      ? "Loading case..."
                      : !caseSymbolProfile
                        ? (
                          <div style={{ fontSize: 10, opacity: 0.6, fontStyle: "italic" }}>
                            No CFG profile yet for this case.
                          </div>
                        )
                        : !Array.isArray(caseSymbolProfile.symbol_analyses) || caseSymbolProfile.symbol_analyses.length === 0
                          ? (
                            <div style={{ fontSize: 10, opacity: 0.6, fontStyle: "italic" }}>
                              Profile has no symbol analyses.
                            </div>
                          )
                          : (
                            <div style={{ display: "grid", gap: 8 }}>
                              {caseSymbolProfile.symbol_analyses.map((sa, idx) => (
                                <SymbolAnalysisCard
                                  key={`${sa.symbol_id}-${idx}`}
                                  sa={sa}
                                  nodeInfo={cfgNodeIndex[sa.symbol_id] ?? null}
                                />
                              ))}
                            </div>
                          )}
                  </div>
                </div>

              </div>
            )}
          </section>

        </div>
      </div>
    </>
  );

  return createPortal(panel, document.body);
}
