import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { DoctrineCaseDrawer } from "./DoctrineCaseDrawer";
import { CfgHtmlTree } from "./CfgHtmlTree";
import { resolve as resolveCase } from "./caseKeys";
import { filterCaseIdsByAllowlist, filterCaseIdsByCourtSelection } from "./courtStats";
import { INK, TEAL } from "./theme";
import { displayNodeId } from "./nodeIds";

function buildNodeIndex(nodes) {
  const out = {};
  for (const node of nodes) {
    if (node && node.id) out[node.id] = node;
  }
  return out;
}

function buildDerivationPath(nodeId, nodeIndex, rootId) {
  if (!nodeId || !nodeIndex[nodeId]) return "";

  const segments = [];
  let currentId = nodeId;
  const seen = {};

  while (currentId && nodeIndex[currentId] && !seen[currentId]) {
    seen[currentId] = true;
    const node = nodeIndex[currentId];
    const label = node.label ?? node.id;
    const skipGenericRoot = currentId === rootId && /^Corpus root$/i.test(String(label));
    if (!skipGenericRoot) segments.unshift(label);
    if (currentId === rootId) break;
    currentId = node.parent_id ?? null;
  }

  return segments.join(" > ");
}

function buildPathNodeIds(nodeId, nodeIndex, rootId) {
  const out = {};
  if (!nodeId || !nodeIndex[nodeId]) return out;

  let currentId = nodeId;
  const seen = {};
  while (currentId && nodeIndex[currentId] && !seen[currentId]) {
    seen[currentId] = true;
    out[currentId] = true;
    if (currentId === rootId) break;
    currentId = nodeIndex[currentId].parent_id ?? null;
  }
  return out;
}

/** Normalize new per-id records and legacy `{ ids, leaf_reason }` groups. */
function normalizeUnassignedRecords(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const row of raw) {
    if (!row || typeof row !== "object") continue;
    if (row.id != null && !Array.isArray(row.ids)) {
      const reason = row.reason || row.leaf_reason || "unassigned";
      out.push({
        id: String(row.id),
        reason,
        explanation:
          row.explanation
          || `Unassigned during tree induction (${reason}).`,
        node_id: row.node_id ?? null,
        is_leaf: !!row.is_leaf,
      });
      continue;
    }
    if (!Array.isArray(row.ids)) continue;
    const reason = row.reason || row.leaf_reason || "unassigned_at_node";
    for (const caseId of row.ids) {
      out.push({
        id: String(caseId),
        reason,
        explanation:
          row.explanation
          || `Unassigned during tree induction (${reason}).`,
        node_id: row.node_id ?? null,
        is_leaf: !!row.is_leaf,
      });
    }
  }
  return out;
}

export function CfgView({
  cfgGraph,
  cfgRecords,
  studyDoctrine,
  unassignedCases = [],
  caseIndex = null,
  // Court-share or citing-case shading computed by DoctrineTreeTab. Passed through
  // to CfgHtmlTree's existing annotation channels, so tree rendering itself is
  // unchanged and the two shadings never have to know about each other.
  nodeAnnotations = null,
  onCaseSelect = null,
  selectedCaseId = null,
  /** Court / jurisdiction facet keys currently selected in the Tree tab. */
  courtSelection = null,
  /** Case ids citing the chronology selection; null when nothing is selected. */
  citerCaseIds = null,
  /** DOM node in the court panel's facet row where the search controls render. */
  searchSlot = null,
}) {
  const nodes = useMemo(() => (Array.isArray(cfgGraph?.nodes) ? cfgGraph.nodes : []), [cfgGraph]);
  const rootId = cfgGraph?.root_id ?? null;

  const [selNode,        setSelNode]        = useState(null);
  const [drawerNode,     setDrawerNode]     = useState(null);
  /** null | "node" | "unassigned" — the case drawer starts closed. */
  const [drawerKind,     setDrawerKind]     = useState(null);
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [remoteSearchRows, setRemoteSearchRows] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [activeMatchCaseId, setActiveMatchCaseId] = useState("");
  const [activeMatchNodeId, setActiveMatchNodeId] = useState(null);

  const nodeDepthById = useMemo(() => {
    const out = {};
    for (const node of nodes) {
      if (!node || !node.id) continue;
      const depthNum = Number(node.depth);
      out[node.id] = Number.isFinite(depthNum) ? depthNum : 0;
    }
    return out;
  }, [nodes]);

  const nodeIndexById = useMemo(() => buildNodeIndex(nodes), [nodes]);

  const unassignedRecords = useMemo(
    () => normalizeUnassignedRecords(unassignedCases),
    [unassignedCases],
  );

  const unassignedIds = useMemo(() => {
    const ids = [];
    const seen = {};
    for (const row of unassignedRecords) {
      if (seen[row.id]) continue;
      seen[row.id] = true;
      ids.push(row.id);
    }
    return ids;
  }, [unassignedRecords]);

  const unassignedAnnotations = useMemo(() => {
    const out = {};
    for (const row of unassignedRecords) {
      if (out[row.id]) continue;
      out[row.id] = {
        reason: row.reason,
        explanation: row.explanation,
        nodeId: row.node_id,
        isLeaf: row.is_leaf,
      };
    }
    return out;
  }, [unassignedRecords]);

  const caseIdToNodeIds = useMemo(() => {
    const out = {};
    for (const node of nodes) {
      if (!node || !node.id || !Array.isArray(node.cases)) continue;
      for (const caseIdRaw of node.cases) {
        const caseId = String(caseIdRaw ?? "").trim();
        if (!caseId) continue;
        if (!out[caseId]) out[caseId] = [];
        out[caseId].push(node.id);
      }
    }
    return out;
  }, [nodes]);

  const selNodeObj = selNode ? nodes.find((n) => n.id === selNode) : null;
  const drawerNodeObj = drawerNode ? nodes.find((n) => n.id === drawerNode.id) : null;
  const citerFilterActive = citerCaseIds != null;
  const caseFilterActive = (!!courtSelection && courtSelection.size > 0) || citerFilterActive;
  const filteredSelCaseIds = useMemo(
    () => filterCaseIdsByAllowlist(
      filterCaseIdsByCourtSelection(selNodeObj?.cases, courtSelection, caseIndex),
      citerCaseIds,
    ),
    [selNodeObj?.cases, courtSelection, caseIndex, citerCaseIds],
  );
  const filteredDrawerCaseIds = useMemo(
    () => filterCaseIdsByAllowlist(
      filterCaseIdsByCourtSelection(drawerNodeObj?.cases, courtSelection, caseIndex),
      citerCaseIds,
    ),
    [drawerNodeObj?.cases, courtSelection, caseIndex, citerCaseIds],
  );
  const filteredUnassignedIds = useMemo(
    () => filterCaseIdsByAllowlist(
      filterCaseIdsByCourtSelection(unassignedIds, courtSelection, caseIndex),
      citerCaseIds,
    ),
    [unassignedIds, courtSelection, caseIndex, citerCaseIds],
  );
  const nodeDescription = selNodeObj?.description ?? selNodeObj?.note ?? "";
  const nodeSummary = selNodeObj?.summary || "";
  const leftDescription = selNodeObj?.left_description ?? selNodeObj?.left_note ?? "";
  const rightDescription = selNodeObj?.right_description ?? selNodeObj?.right_note ?? "";
  const leftChild = selNodeObj?.left_id ? nodeIndexById[selNodeObj.left_id] : null;
  const rightChild = selNodeObj?.right_id ? nodeIndexById[selNodeObj.right_id] : null;
  const leftLabel = selNodeObj?.left_label || leftChild?.label || "";
  const rightLabel = selNodeObj?.right_label || rightChild?.label || "";
  const axisType = selNodeObj?.axis_type || "";
  const splitCriterionShort = selNodeObj?.split_criterion_short || "";
  const decisionRule = selNodeObj?.decision_rule || "";
  const hasAxisProposal =
    !!(selNodeObj?.split_criterion || axisType || splitCriterionShort || decisionRule
      || leftDescription || rightDescription);

  /** Arrival confidence for the selected node (inbound); null for root. */
  const selectedInboundLowFraction = useMemo(() => {
    if (!selNodeObj || selNodeObj.parent_id == null) return null;
    if (selNodeObj.inbound_assignments && typeof selNodeObj.inbound_assignments === "object") {
      if (selNodeObj.low_confidence_fraction != null) {
        const direct = Number(selNodeObj.low_confidence_fraction);
        if (Number.isFinite(direct)) return direct;
      }
    }
    const parent = nodeIndexById[selNodeObj.parent_id];
    const assignments = parent?.assignments;
    if (!assignments || typeof assignments !== "object") return null;
    const caseIds = Array.isArray(selNodeObj.cases) ? selNodeObj.cases.map(String) : [];
    let low = 0;
    let total = 0;
    for (let i = 0; i < caseIds.length; i += 1) {
      const entry = assignments[caseIds[i]];
      if (!entry || !entry.confidence) continue;
      total += 1;
      if (entry.confidence === "low") low += 1;
    }
    if (total === 0) return null;
    return low / total;
  }, [selNodeObj, nodeIndexById]);

  const handleNodeClick = (node) => {
    setSelNode((prev) => (prev === node.id ? null : node.id));
  };

  useEffect(() => {
    let cancelled = false;
    const timeoutId = window.setTimeout(() => {
      const nextSearch = searchInput.trim();
      setSearch(nextSearch);
      if (!nextSearch || !studyDoctrine) {
        setSearchLoading(false);
        setSearchError("");
        return;
      }

      setSearchLoading(true);
      setSearchError("");

      const query = new URLSearchParams({
        doctrine: studyDoctrine,
        page: "1",
        pageSize: "250",
        search: nextSearch,
      });

      fetch(`/api/db-browser/cases?${query.toString()}`, { cache: "no-store" })
        .then((res) => (res.ok ? res.json() : { cases: [] }))
        .then((payload) => {
          if (cancelled) return;
          const rows = Array.isArray(payload.cases) ? payload.cases : [];
          setRemoteSearchRows(rows);
        })
        .catch(() => {
          if (cancelled) return;
          setRemoteSearchRows([]);
          setSearchError("Could not load case search results.");
        })
        .finally(() => {
          if (!cancelled) setSearchLoading(false);
        });
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [searchInput, studyDoctrine]);

  const searchResults = useMemo(() => {
    if (!search) return [];

    const byCaseId = {};
    const normalizedQuery = search.toLowerCase();

    for (const row of remoteSearchRows) {
      if (!row || typeof row !== "object") continue;
      const hasId = row.id !== null && row.id !== undefined;
      const caseId = hasId ? String(row.id) : "";
      if (!caseId) continue;
      if (!byCaseId[caseId]) {
        byCaseId[caseId] = {
          caseId,
          caseName: "",
        };
      }
      if (!byCaseId[caseId].caseName && row.case_name) {
        byCaseId[caseId].caseName = String(row.case_name);
      }
    }

    for (const caseId of Object.keys(caseIdToNodeIds)) {
      if (!caseId.toLowerCase().includes(normalizedQuery)) continue;
      if (!byCaseId[caseId]) {
        byCaseId[caseId] = {
          caseId,
          caseName: "",
        };
      }
    }

    const out = [];
    for (const caseId of Object.keys(byCaseId)) {
      const nodeIds = caseIdToNodeIds[caseId] ?? [];
      if (nodeIds.length === 0) continue;
      let preferredNodeId = nodeIds[0];
      for (const nodeId of nodeIds) {
        if ((nodeDepthById[nodeId] ?? 0) > (nodeDepthById[preferredNodeId] ?? 0)) {
          preferredNodeId = nodeId;
        }
      }
      out.push({
        caseId,
        caseName: byCaseId[caseId].caseName || `Case ${caseId}`,
        nodeIds,
        preferredNodeId,
        derivationPath: buildDerivationPath(preferredNodeId, nodeIndexById, rootId),
      });
    }

    out.sort((left, right) => {
      const byName = left.caseName.localeCompare(right.caseName);
      if (byName !== 0) return byName;
      return left.caseId.localeCompare(right.caseId);
    });

    return out;
  }, [caseIdToNodeIds, nodeDepthById, nodeIndexById, remoteSearchRows, rootId, search]);

  const activeMatchStillVisible = useMemo(() => {
    if (!activeMatchCaseId) return false;
    for (const result of searchResults) {
      if (result.caseId === activeMatchCaseId) return true;
    }
    return false;
  }, [activeMatchCaseId, searchResults]);
  const effectiveActiveMatchCaseId = activeMatchStillVisible ? activeMatchCaseId : "";
  const effectiveActiveMatchNodeId = activeMatchStillVisible ? activeMatchNodeId : null;

  const activeSearchResult = useMemo(() => {
    if (!effectiveActiveMatchCaseId) return null;
    for (const result of searchResults) {
      if (result.caseId === effectiveActiveMatchCaseId) return result;
    }
    return null;
  }, [effectiveActiveMatchCaseId, searchResults]);

  const highlightNodeIds = useMemo(() => {
    if (effectiveActiveMatchNodeId) {
      return buildPathNodeIds(effectiveActiveMatchNodeId, nodeIndexById, rootId);
    }
    if (!search) return {};
    const out = {};
    for (const result of searchResults) {
      for (const nodeId of result.nodeIds) {
        out[nodeId] = true;
      }
    }
    return out;
  }, [effectiveActiveMatchNodeId, nodeIndexById, rootId, search, searchResults]);


  const handleSearchResultClick = (result) => {
    setActiveMatchCaseId(result.caseId);
    setActiveMatchNodeId(result.preferredNodeId);
    setSelNode(result.preferredNodeId);
    // Also drive the chronology below: one click, both panes.
    onCaseSelect?.(result.caseId);
  };

  const closeCaseDrawer = () => {
    setDrawerKind(null);
    setDrawerNode(null);
  };

  const toggleUnassignedDrawer = () => {
    if (drawerKind === "unassigned") {
      closeCaseDrawer();
      return;
    }
    setDrawerNode(null);
    setDrawerKind("unassigned");
  };

  const handleDrawerCaseSelect = (caseId) => {
    // Drawer row clicks stay inside the drawer. Forwarding them to the
    // chronology would pin the case there, and the citer filter would then
    // replace this node's full list with only the cases that cite it.
    if (drawerKind !== "unassigned") return;
    const row = unassignedAnnotations[caseId];
    if (row?.nodeId && nodeIndexById[row.nodeId]) {
      setSelNode(row.nodeId);
    }
  };

  const unassignedDrawerOpen = drawerKind === "unassigned";
  const nodeDrawerOpen = drawerKind === "node" && !!drawerNodeObj?.cases?.length;

  const searchControls = (
    <div className="cfg-case-search-controls">
      <div className="cfg-case-search-header">
        <div className="cfg-case-search-header-main">
          <div className="cfg-case-search-title">Find case in tree</div>
          {search && (
            <span className="cfg-case-search-count">
              {searchLoading ? "Searching..." : `${searchResults.length} match${searchResults.length === 1 ? "" : "es"}`}
            </span>
          )}
        </div>
      </div>
      <div className="cfg-case-search-row">
        <input
          value={searchInput}
          onChange={(event) => setSearchInput(event.target.value)}
          className="cfg-case-search-input"
          placeholder="Search case name or ID..."
        />
        <button
          type="button"
          className="cfg-case-search-clear"
          onClick={() => {
            setSearchInput("");
            setActiveMatchCaseId("");
            setActiveMatchNodeId(null);
          }}
          disabled={searchInput.length === 0}
        >
          Clear
        </button>
      </div>
      <button
        type="button"
        className={`cfg-unassigned-toggle${unassignedDrawerOpen ? " is-open" : ""}`}
        onClick={toggleUnassignedDrawer}
        aria-expanded={unassignedDrawerOpen}
        aria-controls="study-case-drawer-title"
      >
        Unassigned cases
        <span className="study-unassigned-count">{unassignedRecords.length}</span>
      </button>
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0, overflow: "hidden" }}>
      {searchSlot ? createPortal(searchControls, searchSlot) : searchControls}
      {(searchError || activeSearchResult?.derivationPath || (search && !searchLoading)) && (
      <div className="cfg-case-search-box">
        {searchError && <div className="cfg-case-search-error">{searchError}</div>}
        {activeSearchResult && activeSearchResult.derivationPath && (
          <div className="cfg-case-search-derivation">
            <div className="cfg-case-search-derivation-label">Derivation</div>
            <div className="cfg-case-search-derivation-path">{activeSearchResult.derivationPath}</div>
          </div>
        )}
        {search && !searchLoading && !searchError && (
          <div className="cfg-case-search-results">
            {searchResults.length === 0 ? (
              <div className="cfg-case-search-empty">No tree nodes contain a matching case.</div>
            ) : (
              searchResults.map((result) => {
                const isActive = effectiveActiveMatchCaseId === result.caseId
                  || (selectedCaseId && selectedCaseId === result.caseId);
                const resolved = resolveCase(caseIndex, { caseId: result.caseId });
                return (
                  <button
                    key={result.caseId}
                    type="button"
                    className={`cfg-case-search-result ${isActive ? "active" : ""}`}
                    onClick={() => handleSearchResultClick(result)}
                  >
                    <span className="cfg-case-search-result-name">{result.caseName}</span>
                    <span className="cfg-case-search-result-meta">
                      id: {result.caseId}
                      {resolved?.opinionId ? ` · op ${resolved.opinionId}` : ""}
                      {resolved?.courtAbbreviation ? ` · ${resolved.courtAbbreviation}` : ""}
                      {" · nodes: "}{result.nodeIds.length}
                    </span>
                    {result.derivationPath && (
                      <span className="cfg-case-search-result-derivation">{result.derivationPath}</span>
                    )}
                  </button>
                );
              })
            )}
          </div>
        )}
      </div>
      )}

      <div className="study-tree-area" style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
        <div style={{ flex: 1, minHeight: 0, overflowY: "auto", overflowX: "auto" }}>
          <CfgHtmlTree
            depthSlot={searchSlot}
            nodes={nodes}
            rootId={rootId}
            onNodeClick={handleNodeClick}
            selectedNodeId={selNode}
            highlightNodeIds={highlightNodeIds}
            activeMatchNodeId={effectiveActiveMatchNodeId}
            nodeAnnotations={nodeAnnotations}
          />
        </div>
      </div>

      {/* ── Selected-node detail bottom panel ─────────────────────── */}
      {selNodeObj && (
        <div style={{
          borderTop: "1px solid var(--study-rule, #e2e8f0)",
          background: "var(--study-surface, #fff)",
          minHeight: 150,
          padding: "12px 14px 18px",
          flexShrink: 0,
          fontSize: 12,
        }}>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", gap: 14 }}>
            <div style={{ gridColumn: "1 / -1" }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, lineHeight: 1.3 }}>
                {selNodeObj.label ?? displayNodeId(selNodeObj.id)}
              </div>
              <div style={{ fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif', fontSize: 9, color: INK.muted }}>
                {displayNodeId(selNodeObj.id)} · {selNodeObj.is_leaf ? "leaf" : "split"}
                {selectedInboundLowFraction != null && (
                  <>
                    {" · "}
                    low_confidence_fraction: {selectedInboundLowFraction.toFixed(3)}
                  </>
                )}
              </div>
              {activeSearchResult && activeSearchResult.derivationPath && (
                <div className="study-cfg-deriv-breadcrumb" style={{ marginTop: 6 }}>
                  {activeSearchResult.derivationPath}
                </div>
              )}
            </div>

            <div>
              {nodeSummary && (
                <div style={{ fontSize: 12, lineHeight: 1.5, color: INK.dark, marginBottom: 10 }}>
                  {nodeSummary}
                </div>
              )}
              {nodeDescription && (
                <div style={{ fontSize: 11, lineHeight: 1.5, color: INK.mid, marginBottom: 10 }}>
                  {nodeDescription}
                </div>
              )}

              {hasAxisProposal && (
                <div style={{ marginBottom: 10, display: "grid", gap: 10 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: TEAL.shade1, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                    Axis proposal
                  </div>

                  {(axisType || splitCriterionShort) && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
                      {axisType && (
                        <span
                          style={{
                            fontSize: 10,
                            fontWeight: 700,
                            padding: "2px 7px",
                            borderRadius: 999,
                            background: "rgba(99, 102, 241, 0.12)",
                            border: "1px solid rgba(99, 102, 241, 0.35)",
                            color: TEAL.shade1,
                          }}
                        >
                          {axisType}
                        </span>
                      )}
                      {splitCriterionShort && (
                        <span style={{ fontSize: 11, fontWeight: 700, color: INK.dark, lineHeight: 1.35 }}>
                          {splitCriterionShort}
                        </span>
                      )}
                    </div>
                  )}

                  {selNodeObj.split_criterion && (
                    <div>
                      <div style={{ fontSize: 10, fontWeight: 700, color: INK.mid, marginBottom: 2 }}>
                        Split criterion
                      </div>
                      <div style={{ fontSize: 11, lineHeight: 1.45, color: INK.mid }}>
                        {selNodeObj.split_criterion}
                      </div>
                    </div>
                  )}

                </div>
              )}

              {selNodeObj.reasoning_chain && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: INK.mid, marginBottom: 2 }}>
                    Reasoning chain
                  </div>
                  <div style={{ fontSize: 11, lineHeight: 1.45, color: INK.mid, whiteSpace: "pre-wrap" }}>
                    {selNodeObj.reasoning_chain}
                  </div>
                </div>
              )}
            </div>

            <div>
              {decisionRule && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: INK.mid, marginBottom: 2 }}>
                    Decision rule
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      lineHeight: 1.45,
                      color: INK.mid,
                      padding: "6px 8px",
                      borderRadius: 5,
                      background: "rgba(15, 118, 110, 0.06)",
                      border: "1px solid rgba(15, 118, 110, 0.22)",
                    }}
                  >
                    {decisionRule}
                  </div>
                </div>
              )}
              {!selNodeObj.is_leaf && selNodeObj.left_id && selNodeObj.right_id && (
                <>
                  {(leftLabel || rightLabel || leftDescription || rightDescription) && (
                    <div style={{ marginBottom: 10, display: "grid", gap: 8 }}>
                      {(leftLabel || leftDescription) && (
                        <div>
                          <div style={{ fontSize: 10, fontWeight: 700, color: INK.mid, marginBottom: 2 }}>
                            Left{leftLabel ? `: ${leftLabel}` : " cluster"}
                          </div>
                          {leftDescription && (
                            <div style={{ fontSize: 11, lineHeight: 1.45, color: INK.mid }}>
                              {leftDescription}
                            </div>
                          )}
                        </div>
                      )}
                      {(rightLabel || rightDescription) && (
                        <div>
                          <div style={{ fontSize: 10, fontWeight: 700, color: INK.mid, marginBottom: 2 }}>
                            Right{rightLabel ? `: ${rightLabel}` : " cluster"}
                          </div>
                          {rightDescription && (
                            <div style={{ fontSize: 11, lineHeight: 1.45, color: INK.mid }}>
                              {rightDescription}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}

              {selNodeObj.is_leaf && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: INK.mid, marginBottom: 2 }}>
                    Leaf
                  </div>
                  <div style={{ fontSize: 11, lineHeight: 1.45, color: INK.muted }}>
                    {selNodeObj.leaf_reason
                      ? `Doctrinally homogeneous (reason: ${selNodeObj.leaf_reason}).`
                      : "Doctrinally homogeneous — no further split."}
                  </div>
                </div>
              )}

              {Array.isArray(selNodeObj.cases) && selNodeObj.cases.length > 0 && (
                <button
                  type="button"
                  className="study-tl-era-cases-btn study-nd-cases-open"
                  onClick={() => { setDrawerNode(selNodeObj); setDrawerKind("node"); }}
                >
                  View cases &amp; details
                  <span className="study-tl-era-cases-suffix">
                    {" "}({caseFilterActive
                      ? `${filteredSelCaseIds.length} of ${selNodeObj.cases.length}`
                      : selNodeObj.cases.length}
                    )
                  </span>
                </button>
              )}

              <button
                type="button"
                style={{
                  display: "block", marginTop: 12, fontSize: 10, color: INK.muted,
                  background: "none", border: "none", cursor: "pointer", padding: 0,
                }}
                onClick={() => setSelNode(null)}
              >
                ✕ Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Case drawer ───────────────────────────────────────────── */}
      <DoctrineCaseDrawer
        open={unassignedDrawerOpen || nodeDrawerOpen}
        onClose={closeCaseDrawer}
        doctrine={studyDoctrine}
        nodeLabel={unassignedDrawerOpen
          ? "Unassigned cases"
          : (drawerNodeObj ? (drawerNodeObj.label ?? drawerNodeObj.id) : "")}
        treeEntry={unassignedDrawerOpen ? null : drawerNodeObj}
        caseIds={unassignedDrawerOpen ? filteredUnassignedIds : filteredDrawerCaseIds}
        caseAnnotations={unassignedDrawerOpen ? unassignedAnnotations : null}
        caseIndex={caseIndex}
        onCaseSelect={handleDrawerCaseSelect}
        courtRecords={[]}
        cfgRecords={cfgRecords ?? []}
        doctrineGraph={cfgGraph}
        activeEra={null}
        activeThreadId={null}
      />
    </div>
  );
}
