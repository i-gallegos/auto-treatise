import {
  COLOR_OPTIONS,
  makeEdgeVisible,
  useCitationColors,
} from "./useCitationData";
import { CitationScene, H, W } from "./CitationScene";
import { Legend } from "./CitationSidebar";
import { degreeSelection, describeDegreeSelection, inducedSubgraph, maxInDegree } from "./degreeFilter";
import { layoutGroups } from "./groupLayout";
import { displayRank, restrictChronologyRanks } from "./rankBands";
import { buildAdjacency } from "./reachability";
import { useCallback, useMemo } from "react";

const LAYOUT_TITLES = {
  force: "citation graph — u → v means u cites v",
  time: null,
};

export function CitationPane({
  data,
  layout = "force",
  state,
  height = H,
  width = W,
  title,
  note,
  clipId = "study-cit-clip",
  toolbarLeft = null,
  toolbarExtra = null,
  legendTitle,
  sidebarTop = null,
  sidebarBottom = null,
  sidebarClassName = "",
  caseIndex = null,
  // -- highlight inputs, all optional ---------------------------------------- //
  courtMatches = null,
  citers = null,
  selectedIndex = null,
  focusIndex = null,
  dimOutside = null,
  onHoverNode,
  onClickNode,
  onBackgroundClick,
  colorAt = null,
  onlyUnder = null,
  onlyRanks = null,
  onlyCircuit = null,
  groups = null,
  groupLayout = false,
  hubLabel = null,
  fadeOutsideRings = false,
  marker = null,
  medians = null,
  ringIndices = null,
  ringCaption = null,
  inboundOnly = false,
}) {
  const { nodes, byId, treeNodes, leafColor, noLeafColor, hasTree } = data;
  const {
    colorBy, setColorBy,
    treeOnly, setTreeOnly,
    minCitedBy, setMinCitedBy, hidden, toggleHidden,
  } = state;


  const { groupKey, groupColors, colorOf, groupLabel } = useCitationColors({
    nodes, leafColor, treeNodes, colorBy, noLeafColor, caseIndex, colorAt, groups,
  });

  // Training-case view: degrees and the threshold are counted on the induced
  // subgraph, so a case cited only from outside the tree is not a hub here.
  const training = useMemo(
    () => (treeOnly && hasTree
      ? inducedSubgraph(nodes, data.links, (n) => !!n.leaf)
      : null),
    [treeOnly, hasTree, nodes, data.links],
  );
  const degreeNodes = training?.nodes ?? nodes;
  const degreeAdj = useMemo(
    () => (training ? buildAdjacency(training.links) : data.adjacency),
    [training, data.adjacency],
  );
  const degreeOf = useCallback(
    (n) => (training ? (training.inDeg.get(n.i)?.length ?? 0) : (n.in_deg ?? 0)),
    [training],
  );
  const outDegreeOf = useCallback(
    (n) => (training ? (training.outDeg.get(n.i)?.length ?? 0) : (n.out_deg ?? 0)),
    [training],
  );

  const maxDegree = useMemo(() => maxInDegree(degreeNodes), [degreeNodes]);
  const degrees = useMemo(
    () => degreeSelection(degreeNodes, degreeAdj, minCitedBy),
    [degreeNodes, degreeAdj, minCitedBy],
  );

  const nodeVisible = useCallback(
    (n) => {
      if (!n) return false;
      const k = groupKey(n);
      if (hidden.has(k === null ? "__none__" : k)) return false;
      // `leaf` is the citation payload's tree membership: set iff the case sits
      // on a leaf of the induced tree. Unassigned / never-profiled cases stay
      // in the sweep with `leaf: null`.
      if (treeOnly && hasTree && !n.leaf) return false;
      // A figure can keep one branch of the tree. The Rahimi chronology drops
      // the tradition-inquiry-bypassed child so the date shift is readable.
      if (onlyUnder && !(n.path ?? []).includes(onlyUnder)) return false;
      if (onlyCircuit && String(n.circuit) !== String(onlyCircuit)) return false;
      if (onlyRanks?.length && !onlyRanks.includes(displayRank(n))) return false;
      if (degrees && !degrees.visible.has(n.i)) return false;
      return true;
    },
    [groupKey, hidden, treeOnly, hasTree, degrees, onlyUnder, onlyCircuit, onlyRanks],
  );

  // Same chronology as the unfiltered view, limited to the figure's court bands.
  // A case in those bands that is not on the colored branch is grey, and only
  // when a citation joins it to a teal or orange case. Courts outside the
  // bands are omitted, and so are the citations to them.
  const ghostIndices = useMemo(() => {
    if (!onlyUnder && !onlyRanks?.length) return null;
    const core = new Set();
    nodes.forEach((n) => {
      if (nodeVisible(n)) core.add(n.i);
    });
    const linked = new Set();
    (data.links ?? []).forEach((e) => {
      const sourceIn = core.has(e.s);
      const targetIn = core.has(e.t);
      if (sourceIn === targetIn) return;
      linked.add(sourceIn ? e.t : e.s);
    });
    const ghosts = new Set();
    nodes.forEach((n) => {
      if (!n || !linked.has(n.i) || nodeVisible(n)) return;
      if (onlyRanks?.length && !onlyRanks.includes(displayRank(n))) return;
      if (treeOnly && hasTree && !n.leaf) return;
      if (degrees && !degrees.visible.has(n.i)) return;
      ghosts.add(n.i);
    });
    return ghosts;
  }, [onlyUnder, onlyRanks, nodes, data.links, nodeVisible, treeOnly, hasTree, degrees]);

  const edgeVisible = useMemo(() => {
    const drawn = (n) => !!n && (nodeVisible(n) || !!ghostIndices?.has(n.i));
    return makeEdgeVisible({ nodeVisible: drawn, byId });
  }, [nodeVisible, byId, ghostIndices]);

  // On a figure, radius follows the edges actually drawn, which are every
  // in-edge of a shown case. A grey citer outside the figure is one of those.
  const shownInDegree = useMemo(() => {
    if (!ghostIndices && !groupLayout) return null;
    const counts = new Map();
    (data.links ?? []).forEach((e) => {
      if (!edgeVisible(e)) return;
      counts.set(e.t, (counts.get(e.t) || 0) + 1);
    });
    return counts;
  }, [ghostIndices, groupLayout, data.links, edgeVisible]);
  const figureDegreeOf = useCallback(
    (n) => shownInDegree?.get(n.i) ?? 0,
    [shownInDegree],
  );

  const legendGroups = useMemo(() => {
    const counts = new Map();
    const counted = (training?.nodes ?? nodes).filter((n) => {
      if (onlyUnder && !(n.path ?? []).includes(onlyUnder)) return false;
      if (onlyCircuit && String(n.circuit) !== String(onlyCircuit)) return false;
      if (onlyRanks?.length && !onlyRanks.includes(displayRank(n))) return false;
      return true;
    });
    counted.forEach((n) => {
      const k = groupKey(n);
      const key = k === null ? "__none__" : k;
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [nodes, training, groupKey, onlyUnder, onlyCircuit, onlyRanks]);

  const baseAxis = training && data.timeAxisTree ? data.timeAxisTree : data.timeAxis;
  const compressed = useMemo(
    () => (layout === "time" && onlyRanks?.length
      ? restrictChronologyRanks(nodes, baseAxis, onlyRanks)
      : null),
    [layout, onlyRanks, nodes, baseAxis],
  );
  const sceneData = compressed ? { ...data, nodes: compressed.nodes } : data;
  const sceneAxis = compressed?.timeAxis ?? baseAxis;

  const colorLabel = groups?.length
    ? "reading"
    : colorAt
      ? "history fork"
      : (COLOR_OPTIONS.find((o) => o.id === colorBy)?.label ?? colorBy);
  const layoutPositions = useMemo(() => {
    if (!groupLayout) return null;
    return layoutGroups(
      nodes.filter(nodeVisible),
      data.links,
      groupKey,
      fadeOutsideRings ? ringIndices : null,
    );
  }, [groupLayout, nodes, data.links, nodeVisible, groupKey, fadeOutsideRings, ringIndices]);
  const treeCaseCount = useMemo(
    () => nodes.reduce((n, node) => n + (node.leaf ? 1 : 0), 0),
    [nodes],
  );

  return (
    <>
      <div className="study-cit-toolbar">
        {toolbarLeft}
        <label className="study-cit-ctl">
          color
          <select
            value={colorBy}
            disabled={!!colorAt || !!groups?.length}
            title={colorAt || groups?.length ? "This figure sets the color." : undefined}
            onChange={(e) => setColorBy(e.target.value)}
          >
            {COLOR_OPTIONS.map((option) => (
              <option
                key={option.id}
                value={option.id}
                disabled={option.needsTree && !hasTree}
              >
                {option.label}
              </option>
            ))}
          </select>
        </label>
        {hasTree ? (
          <label
            className="study-cit-ctl"
            title={
              "Hide cases that are in the citation sweep but never landed on a "
              + "leaf of the induced tree (unassigned, or skipped before profiling)."
            }
          >
            <input
              type="checkbox"
              checked={treeOnly}
              onChange={(e) => setTreeOnly(e.target.checked)}
            />
            Only cases in the tree
          </label>
        ) : null}
        {maxDegree > 1 ? (
          <label
            className="study-cit-ctl"
            title={
              "Keep only cases cited at least this many times within the corpus, "
              + "plus the cases that cite them (faded). 0 shows everything."
            }
          >
            Minimum cited by
            <input
              type="range"
              min={0}
              max={maxDegree}
              step={1}
              value={Math.min(minCitedBy, maxDegree)}
              onChange={(e) => setMinCitedBy(Number(e.target.value))}
            />
            <span className="study-cit-thresh-value">{minCitedBy || "off"}</span>
          </label>
        ) : null}
        {toolbarExtra}
      </div>

      {note ? <p className="study-cit-note">{note}</p> : null}

      <div className="study-cit-body">
        <div className="study-cit-stage">
          <CitationScene
            layout={layout}
            data={sceneData}
            colorOf={colorOf}
            nodeVisible={nodeVisible}
            edgeVisible={edgeVisible}
            height={height}
            width={width}
            clipId={clipId}
            title={title ?? LAYOUT_TITLES[layout]}
            courtMatches={courtMatches}
            fadedIndices={degrees?.citers ?? null}
            citers={citers}
            selectedIndex={selectedIndex}
            focusIndex={focusIndex}
            dimOutside={dimOutside}
            treeCoords={!!training}
            timeAxis={sceneAxis}
            marker={marker}
            medians={medians}
            ringIndices={ringIndices}
            ringCaption={ringCaption}
            degreeOf={shownInDegree ? figureDegreeOf : (training ? degreeOf : null)}
            outDegreeOf={training ? outDegreeOf : null}
            inboundOnly={inboundOnly}
            ghostIndices={ghostIndices}
            positions={layoutPositions}
            hubLabel={hubLabel}
            plainEdges={groupLayout}
            fadeOutsideRings={fadeOutsideRings}
            onHoverNode={onHoverNode}
            onClickNode={onClickNode}
            onBackgroundClick={onBackgroundClick}
          />
        </div>

        <aside className={`study-cit-side${sidebarClassName ? ` ${sidebarClassName}` : ""}`}>
          {sidebarTop}
          {treeOnly && hasTree ? (
            <>
              <h4>Tree-only filter</h4>
              <p className="study-muted">
                {treeCaseCount} of {nodes.length} cases sit on a tree leaf; the rest
                are hidden. Citations that leave the tree are hidden with them.
                {nodes.some((node) => Number.isFinite(node.fx_tree))
                  ? " Positions are the citation layout of those cases alone."
                  : ""}
              </p>
            </>
          ) : null}
          {degrees ? (
            <>
              <h4>Most-cited filter</h4>
              <p className="study-muted">
                {describeDegreeSelection(degrees)}
                {degrees.hubCount > 0
                  ? " Faded cases cite one of them directly; the relation is one hop, "
                    + "never a chain."
                  : ""}
              </p>
              <button
                type="button"
                className="study-court-clear"
                onClick={() => setMinCitedBy(0)}
              >
                show all cases
              </button>
            </>
          ) : null}
          <h4>{legendTitle ?? `Legend — ${colorLabel}`}</h4>
          <Legend
            groups={legendGroups}
            hidden={hidden}
            onToggle={toggleHidden}
            colorFor={(key) => groupColors.get(key)}
            labelFor={groupLabel}
            noLeafColor={noLeafColor}
          />
          {sidebarBottom}
        </aside>
      </div>
    </>
  );
}

export default CitationPane;
