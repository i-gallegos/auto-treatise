import { useCallback, useEffect, useMemo, useState } from "react";
import { analysisCaseLevelPath, analysisRunDir } from "./analysisPaths";
import { CfgView } from "./CfgView";
import { CourtHighlightPanel } from "./CourtHighlightPanel";
import { asKey, resolve as resolveCase } from "./caseKeys";
import {
  buildNodeCourtStats,
  courtFacetOptions,
  matchingCitationIndices,
} from "./courtStats";
import {
  buildTreeAnnotations,
} from "./treeHighlight";
import { CitationPane } from "./citations/CitationPane";
import { useCitationData } from "./citations/useCitationData";
import { usePaneState } from "./citations/usePaneState";
import { citersOf } from "./citations/reachability";
import { SELECTED_COLOR } from "./citations/useCitationData";
import { CaseDetail, SplitRows } from "./citations/CitationSidebar";

const SPLIT_STORAGE_KEY = "study.treeTab.splitPct";
const CHRONOLOGY_HEIGHT = 370;

/** A new Set with `key` flipped. */
function toggledSet(prev, key) {
  const next = new Set(prev);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  return next;
}

function readStoredSplit() {
  try {
    const raw = window.localStorage.getItem(SPLIT_STORAGE_KEY);
    const value = Number(raw);
    return Number.isFinite(value) && value >= 25 && value <= 85 ? value : 58;
  } catch {
    return 58;
  }
}

export function DoctrineTreeTab({
  studyDoctrine,
  studyRun,
  runKind = "full",
  cfgGraph,
  cfgRecords,
  unassignedCases = [],
  caseIndex = null,
  citationGraph = null,
  citationLoading = false,
  splitIndex = null,
}) {
  const hasCfg = !!cfgGraph && Array.isArray(cfgGraph.nodes) && cfgGraph.nodes.length > 0;
  const nodes = useMemo(() => (hasCfg ? cfgGraph.nodes : []), [hasCfg, cfgGraph]);
  const rootId = cfgGraph?.root_id ?? null;

  // -- court highlight ---------------------------------------------------- //

  const [courtSelection, setCourtSelection] = useState(() => new Set());

  const courtStats = useMemo(
    () => buildNodeCourtStats(nodes, caseIndex),
    [nodes, caseIndex],
  );

  const courtOptions = useMemo(
    () => courtFacetOptions(courtStats, rootId),
    [courtStats, rootId],
  );

  const toggleCourt = useCallback((key) => {
    setCourtSelection((prev) => toggledSet(prev, key));
  }, []);

  // -- citation graph ----------------------------------------------------- //

  const data = useCitationData(citationGraph);
  const { hasData, byId, treeNodes, nodes: citNodes, indexFor } = data;

  const [hoverIndex, setHoverIndex] = useState(null);
  const [selectedOpinionId, setSelectedOpinionId] = useState(null);
  const [splitPct, setSplitPct] = useState(readStoredSplit);
  const [dragging, setDragging] = useState(false);

  // Controls for the chronology pane. Held here rather than inside the pane so a
  // colour choice survives anything that remounts it.
  const [searchSlot, setSearchSlot] = useState(null);
  const paneState = usePaneState({ colorBy: "rank" });

  const selectedIndex = useMemo(
    () => (selectedOpinionId ? indexFor({ opinionId: selectedOpinionId }) : null),
    [selectedOpinionId, indexFor],
  );

  /** The cases citing the selection — one hop, never a chain. */
  const citers = useMemo(
    () => (selectedIndex === null || selectedIndex === undefined
      ? null
      : citersOf(data.adjacency, selectedIndex)),
    [data.adjacency, selectedIndex],
  );

  /**
   * The same court selection, applied to the chronology's cases. Its totals are
   * reported separately from the tree's because the corpora differ: the citation
   * graph holds every swept case, the tree only the ones its run covered.
   */
  const courtMatches = useMemo(
    () => matchingCitationIndices(citNodes, courtSelection, caseIndex),
    [citNodes, courtSelection, caseIndex],
  );

  const selectByIndex = useCallback(
    (index) => {
      const node = byId.get(index);
      const opinionId = asKey(node?.opinion_id);
      setSelectedOpinionId((prev) => (prev && prev === opinionId ? null : opinionId));
    },
    [byId],
  );

  /** Called by CfgView when a search result is clicked. Drawer rows do not. */
  const selectByCaseId = useCallback(
    (caseId) => {
      const opinionId = resolveCase(caseIndex, { caseId })?.opinionId
        ?? asKey(byId.get(indexFor({ caseId }))?.opinion_id);
      if (opinionId) setSelectedOpinionId((prev) => (prev === opinionId ? null : opinionId));
    },
    [caseIndex, byId, indexFor],
  );

  // -- tree annotations --------------------------------------------------- //

  const citerCaseIds = useMemo(() => {
    if (!citers) return null;
    const out = new Set();
    for (const index of citers) {
      const node = byId.get(index);
      const caseId = asKey(node?.case_id)
        ?? resolveCase(caseIndex, { opinionId: asKey(node?.opinion_id) })?.caseId;
      if (caseId) out.add(caseId);
    }
    return out;
  }, [citers, byId, caseIndex]);

  const selectedCaseId = useMemo(() => {
    if (selectedIndex === null || selectedIndex === undefined) return null;
    const node = byId.get(selectedIndex);
    return asKey(node?.case_id)
      ?? resolveCase(caseIndex, { opinionId: asKey(node?.opinion_id) })?.caseId
      ?? null;
  }, [selectedIndex, byId, caseIndex]);

  // Channel precedence lives in treeHighlight.js, pure and unit-tested.
  const nodeAnnotations = useMemo(
    () => buildTreeAnnotations({
      nodes,
      courtStats,
      courtSelection,
      selectedCaseId,
      citerCaseIds,
      rootId,
    }),
    [nodes, courtStats, courtSelection,
      selectedCaseId, citerCaseIds, rootId],
  );

  // -- draggable split ---------------------------------------------------- //

  useEffect(() => {
    if (!dragging) return undefined;
    const onMove = (ev) => {
      const host = document.querySelector(".study-tree-split");
      if (!host) return;
      const rect = host.getBoundingClientRect();
      const pct = ((ev.clientY - rect.top) / rect.height) * 100;
      setSplitPct(Math.min(85, Math.max(25, pct)));
    };
    const onUp = () => setDragging(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragging]);

  useEffect(() => {
    try {
      window.localStorage.setItem(SPLIT_STORAGE_KEY, String(splitPct));
    } catch {
      /* private mode: the split just does not persist */
    }
  }, [splitPct]);

  if (!hasCfg) {
    return (
      <div className="study-tree-shell">
        <p className="study-cfg-empty">
          No CFG grammar for this doctrine/run. Add{" "}
          <code className="study-code-inline">corpus_level.json</code> under{" "}
          <code className="study-code-inline">
            {analysisRunDir(studyDoctrine, studyRun, runKind)}
          </code>
          {" "}and keep shared case profiles in{" "}
          <code className="study-code-inline">
            {analysisCaseLevelPath(studyDoctrine, runKind)}
          </code>
          .
        </p>
      </div>
    );
  }

  const focusNode = hoverIndex !== null && hoverIndex !== undefined
    ? byId.get(hoverIndex)
    : (selectedIndex !== null && selectedIndex !== undefined ? byId.get(selectedIndex) : null);


  return (
    <div className="study-tree-shell study-tree-split">
      <CourtHighlightPanel
        options={courtOptions}
        selection={courtSelection}
        onToggle={toggleCourt}
        onClear={() => setCourtSelection(new Set())}
        extraRef={setSearchSlot}
        caseIndex={caseIndex}
        disabled={!caseIndex || caseIndex.size === 0}
        disabledReason={
          "no case index for this run — build one with `python -m "
          + "auto_treatise.case_index --doctrine <d>`"
        }
      />
      {(selectedCaseId || citerCaseIds?.size) && courtSelection.size > 0 ? (
        <p className="study-cit-note">
          A case is selected, so the node fill is showing the cases that cite it; the
          court selection stays in the badges. Clear the case to shade by court again.
        </p>
      ) : null}

      <div className="study-tree-pane" style={{ flex: `0 0 ${splitPct}%` }}>
        <CfgView
          searchSlot={searchSlot}
          cfgGraph={cfgGraph}
          cfgRecords={cfgRecords ?? []}
          studyDoctrine={studyDoctrine}
          unassignedCases={unassignedCases}
          caseIndex={caseIndex}
          nodeAnnotations={nodeAnnotations}
          onCaseSelect={selectByCaseId}
          selectedCaseId={selectedCaseId}
          courtSelection={courtSelection}
          citerCaseIds={citerCaseIds}
        />
      </div>

      <div
        className={`study-tree-handle${dragging ? " dragging" : ""}`}
        onMouseDown={() => setDragging(true)}
        role="separator"
        aria-orientation="horizontal"
        title="drag to resize"
      />

      <div className="study-tree-pane study-tree-pane--chron">
        {citationLoading ? (
          <p className="study-cfg-empty">Loading citation graph…</p>
        ) : !hasData ? (
          <p className="study-cfg-empty">
            No citation graph for this run. Build one with{" "}
            <code className="study-code-inline">
              python -m citations --doctrine {studyDoctrine}
              {studyRun ? ` --run-id ${studyRun}` : ""}
            </code>
            .
          </p>
        ) : (
          // The same pane the Citations tab renders — one toolbar, one legend, one
          // drawing surface, three highlight modes — with the tree tab's own
          // selection state and sidebar content.
          <CitationPane
            data={data}
            layout="time"
            state={paneState}
            height={CHRONOLOGY_HEIGHT}
            clipId="study-tree-chron-clip"
            sidebarClassName="study-tree-chron-side"
            toolbarLeft={
              <span className="study-court-title">Chronology × hierarchy</span>
            }
            courtMatches={courtMatches?.indices ?? null}
            citers={citers}
            selectedIndex={selectedIndex}
            onHoverNode={setHoverIndex}
            onClickNode={selectByIndex}
            onBackgroundClick={() => setSelectedOpinionId(null)}
            caseIndex={caseIndex}
            sidebarTop={
              <>
                {courtMatches ? (
                  <>
                    <h4>Court filter</h4>
                    <p className="study-muted">
                      <b>{courtMatches.matched}</b> of {courtMatches.total} cases match;
                      the rest are dimmed. Citations are kept lit only where both ends
                      match, which is the “do these courts cite each other” question.
                      {courtMatches.unresolved
                        ? ` ${courtMatches.unresolved} case(s) carry no court metadata and can never match.`
                        : ""}
                      {selectedIndex !== null && selectedIndex !== undefined
                        ? " A case is selected, so the cases citing it are showing"
                          + " instead — clear it to see the filter again."
                        : ""}
                    </p>
                  </>
                ) : null}

                <h4>Selected case</h4>
                {selectedIndex === null || selectedIndex === undefined ? (
                  <p className="study-muted">
                    Click a case to mark it in the tree and shade the nodes holding the
                    cases that cite it directly.
                  </p>
                ) : (
                  <CiterReport
                    node={byId.get(selectedIndex)}
                    citers={citers}
                    splitIndex={splitIndex}
                    caseIndex={caseIndex}
                    onClear={() => setSelectedOpinionId(null)}
                  />
                )}

                {focusNode && focusNode.i !== selectedIndex ? (
                  <>
                    <h4>Hovered</h4>
                    <CaseDetail
                      node={focusNode}
                      leafLabel={treeNodes.get(focusNode.leaf)?.label}
                      splitIndex={splitIndex}
                      caseIndex={caseIndex}
                    />
                  </>
                ) : null}
              </>
            }
          />
        )}
      </div>
    </div>
  );
}

/**
 * What the selection actually found. The count matters as much as the picture:
 * many cases in these corpora are cited by nothing in the corpus, and that is a
 * finding rather than a rendering failure.
 */
function CiterReport({ node, citers, onClear, splitIndex = null, caseIndex = null }) {
  if (!node) return null;
  const total = citers?.size ?? 0;
  return (
    <div className="study-cit-detail">
      <div className="study-cit-detail-name" style={{ color: SELECTED_COLOR }}>
        {node.name}
      </div>
      <dl>
        <dt>Court</dt><dd>{node.court_name || node.court || "?"}</dd>
        <dt>Decided</dt><dd>{node.date ?? "?"}</dd>
        <SplitRows node={node} splitIndex={splitIndex} caseIndex={caseIndex} />
        <dt>Cited by</dt><dd>{node.in_deg}</dd>
        <dt>Citing cases shown</dt><dd>{total}</dd>
        <dt>Opinion</dt><dd>{node.opinion_id}</dd>
      </dl>
      {total === 0 ? (
        <p className="study-muted">
          Nothing in this corpus cites it. That is common: most cases in these corpora
          cite the seed case and each other rarely.
        </p>
      ) : null}
      <button type="button" className="study-court-clear" onClick={onClear}>
        Clear selection
      </button>
    </div>
  );
}
