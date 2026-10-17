import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { INK, LEAF, MATCH, SEARCH, SPLIT, TEAL } from "./theme";
import { displayNodeId } from "./nodeIds";

function buildNodeIndex(nodes) {
  const index = {};
  for (const node of nodes ?? []) {
    if (node && node.id) index[node.id] = node;
  }
  return index;
}

function detectRootId(nodes, rootId) {
  if (rootId) return rootId;
  for (const node of nodes ?? []) {
    if (node && (node.parent_id === null || node.parent_id === undefined)) return node.id;
  }
  return null;
}

function maxDepth(nodeId, idx) {
  if (!nodeId || !idx[nodeId]) return 0;
  const n = idx[nodeId];
  const left = n.left_id ? maxDepth(n.left_id, idx) : n.depth ?? 0;
  const right = n.right_id ? maxDepth(n.right_id, idx) : n.depth ?? 0;
  return Math.max(n.depth ?? 0, left, right);
}

function CaseBadge({ cases }) {
  const count = Array.isArray(cases) ? cases.length : 0;
  if (count === 0) return null;
  return (
    <span style={{ fontSize: 8, color: INK.muted }}>
      {count} case{count === 1 ? "" : "s"}
    </span>
  );
}

function extraOutlineStyle(annotation) {
  if (annotation?.ring?.color) {
    const width = annotation.ring.width ?? 2;
    const style = annotation.ring.style ?? "dashed";
    const offset = annotation.ring.offset ?? 2;
    return {
      outline: `${width}px ${style} ${annotation.ring.color}`,
      outlineOffset: `${offset}px`,
    };
  }
  if (annotation?.dotted) {
    return { outline: `2px dotted ${annotation.dotted}`, outlineOffset: "2px" };
  }
  return null;
}
const MUTED_BORDER = INK.rule;
const MUTED_BACKGROUND = "#ffffff";
const MUTED_TEXT = INK.muted;

/**
 * Optional overlay for the incremental case addition tab and the Tree tab's court
 * and citation highlights. Each entry may carry:
 *   fill        - background colour (annotator path in the incremental tab;
 *                 court-share or citer count on the Tree tab)
 *   border      - border colour for a node the overlay *selects*, so matches read
 *                 as outlined-and-filled against muted neighbours
 *   muted       - drop the default leaf-green / split-orange styling to neutral
 *                 grey. Without this the base palette competes with the overlay
 *                 fill and every node still looks coloured, which is exactly what
 *                 makes a filter hard to read
 *   outline     - border colour, an independent channel from `fill` so a node
 *                 can show both "on the annotated path" and "on a predicted path"
 *   outlineGlow - ring colour paired with `outline`
 *   outlineWidth - CSS border-width in px when `outline` is set; defaults to 2.
 *                 The incremental tab scales this by predicted mass.
 *   dotted      - CSS `outline` colour, drawn outside the border as a third
 *                 independent channel (selected-case path on the Tree tab)
 *   ring        - { color, style, width, offset } for a dashed/dotted CSS
 *                 outline; used by the incremental tab for Embedding NN so it
 *                 can share a node with Auto-Treatise's solid `outline`
 *   badge       - short string replacing the case count
 *   textColor   - label colour, for when a dark fill would swallow the default
 * Absent (as in the Tree tab with no overlay active), rendering is unchanged.
 */
function TreeNode({
  nodeId,
  idx,
  visibleDepth,
  onNodeClick,
  selectedNodeId,
  highlightNodeIds,
  activeMatchNodeId,
  nodeAnnotations,
}) {
  if (!nodeId || !idx[nodeId]) return null;
  const node = idx[nodeId];
  const hasChildren = !node.is_leaf && node.left_id && node.right_id;
  const collapsed = hasChildren && (node.depth ?? 0) >= visibleDepth;
  const isSelected = selectedNodeId === node.id;
  const isHighlighted = !!highlightNodeIds?.[node.id];
  const isActiveMatch = activeMatchNodeId === node.id;
  const nodeDescription = node.description ?? node.note ?? "";

  let borderColor = node.is_leaf ? LEAF.border : SPLIT.border;
  let background = node.is_leaf ? LEAF.fill : SPLIT.fill;
  let boxShadow = "none";
  let borderWidth = 1;

  const annotation = nodeAnnotations?.[node.id] ?? null;
  // Muting comes first so `fill` and `border` below can paint over it: a node the
  // overlay selects keeps its own colours, everything else falls back to neutral
  // rather than staying leaf-green or split-orange.
  if (annotation?.muted) {
    borderColor = MUTED_BORDER;
    background = MUTED_BACKGROUND;
  }
  if (annotation?.fill) {
    background = annotation.fill;
  }
  if (annotation?.border) {
    borderColor = annotation.border;
    borderWidth = 2;
  }

  if (isHighlighted) {
    borderColor = SEARCH.border;
    if (!annotation?.fill) background = SEARCH.fill;
  }
  if (isActiveMatch) {
    borderColor = MATCH.border;
    if (!annotation?.fill) background = MATCH.fill;
    boxShadow = `0 0 0 1px ${TEAL.shade1}33`;
  }
  // The outline is applied after the fill so the two channels stay independent.
  if (annotation?.outline) {
    borderColor = annotation.outline;
    borderWidth = annotation.outlineWidth ?? 2;
    if (annotation.outlineGlow) {
      boxShadow = `0 0 0 2px ${annotation.outlineGlow}`;
    }
  }
  if (isSelected) {
    // A click adds an outer ring rather than recolouring the border, so it does
    // not erase the predicted-path outline.
    boxShadow = annotation?.outlineGlow
      ? `0 0 0 2px ${annotation.outlineGlow}, 0 0 0 4px ${TEAL.shade1}4d`
      : `0 0 0 2px ${TEAL.shade1}4d`;
    if (!annotation?.outline) borderColor = MATCH.border;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <button
        type="button"
        onClick={() => onNodeClick?.(node)}
        style={{
          borderStyle: "solid",
          borderWidth,
          borderColor,
          background,
          // A CSS outline sits outside the border, giving a third channel that
          // does not collide with the predicted-path border.
          ...extraOutlineStyle(annotation),
          borderRadius: 7,
          padding: "4px 7px",
          minWidth: 96,
          maxWidth: 160,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 2,
          cursor: "pointer",
          boxShadow,
        }}
        title={annotation?.title}
      >
        <span style={{ fontSize: 8, fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif', color: INK.muted }}>{displayNodeId(node.id)}</span>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            // A dark overlay fill would otherwise swallow the label; a muted node
            // recedes so the selected ones carry the eye.
            color: annotation?.textColor
              ?? (annotation?.muted ? MUTED_TEXT : INK.dark),
            textAlign: "center",
            lineHeight: 1.25,
          }}
        >
          {node.label ?? displayNodeId(node.id)}
        </span>
        {annotation?.badge ? (
          <span
            style={{
              fontSize: 9,
              fontWeight: 700,
              color: annotation.textColor ?? INK.mid,
            }}
          >
            {annotation.badge}
          </span>
        ) : (
          <CaseBadge cases={node.cases} />
        )}
      </button>
      {isSelected && nodeDescription && (
        <div
          style={{
            marginTop: 5,
            maxWidth: 600,
            fontSize: 9,
            lineHeight: 1.35,
            color: INK.mid,
            background: "#ffffff",
            border: `1px solid ${INK.rule}`,
            borderRadius: 6,
            padding: "5px 7px",
            textAlign: "left",
          }}
        >
          {nodeDescription}
        </div>
      )}

      {hasChildren && (
        <>
          <svg
            width="1"
            height="14"
            style={{ display: "block", overflow: "visible", pointerEvents: "none" }}
            aria-hidden="true"
          >
            <line x1="0.5" y1="0" x2="0.5" y2="14" stroke={INK.rule} strokeWidth="1" shapeRendering="crispEdges" />
          </svg>
          {collapsed ? (
            <div style={{ fontSize: 9, color: INK.muted, border: `1px dashed ${INK.rule}`, borderRadius: 6, padding: "1px 7px" }}>
              hidden split
            </div>
          ) : (
            <div style={{ position: "relative", display: "flex", alignItems: "flex-start", gap: 8, width: "100%", paddingTop: 14 }}>
              <svg
                width="100%"
                height="14"
                style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none", overflow: "visible" }}
                aria-hidden="true"
              >
                <line x1="25%" y1="0.5" x2="75%" y2="0.5" stroke={INK.rule} strokeWidth="1" shapeRendering="crispEdges" />
                <line x1="25%" y1="0.5" x2="25%" y2="14" stroke={INK.rule} strokeWidth="1" shapeRendering="crispEdges" />
                <line x1="75%" y1="0.5" x2="75%" y2="14" stroke={INK.rule} strokeWidth="1" shapeRendering="crispEdges" />
              </svg>
              <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
                <TreeNode
                  nodeId={node.left_id}
                  idx={idx}
                  visibleDepth={visibleDepth}
                  onNodeClick={onNodeClick}
                  selectedNodeId={selectedNodeId}
                  highlightNodeIds={highlightNodeIds}
                  activeMatchNodeId={activeMatchNodeId}
                  nodeAnnotations={nodeAnnotations}
                />
              </div>
              <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
                <TreeNode
                  nodeId={node.right_id}
                  idx={idx}
                  visibleDepth={visibleDepth}
                  onNodeClick={onNodeClick}
                  selectedNodeId={selectedNodeId}
                  highlightNodeIds={highlightNodeIds}
                  activeMatchNodeId={activeMatchNodeId}
                  nodeAnnotations={nodeAnnotations}
                />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function CfgHtmlTree({
  nodes,
  rootId,
  onNodeClick,
  selectedNodeId,
  highlightNodeIds,
  activeMatchNodeId,
  nodeAnnotations,
  depthSlot = null,
}) {
  const idx = useMemo(() => buildNodeIndex(nodes), [nodes]);
  const actualRootId = useMemo(() => detectRootId(nodes, rootId), [nodes, rootId]);
  const treeDepth = useMemo(() => maxDepth(actualRootId, idx), [actualRootId, idx]);
  const [visibleDepth, setVisibleDepth] = useState(0);

  if (visibleDepth !== treeDepth && visibleDepth > treeDepth) {
    setVisibleDepth(treeDepth);
  }

  if (!actualRootId) {
    return <div style={{ padding: 20, color: INK.muted, fontSize: 12 }}>No dendrogram nodes to display.</div>;
  }

  const depthControls = (
    <div className="cfg-depth-controls">
      <span>Depth</span>
      <input
        type="range"
        min={0}
        max={treeDepth}
        value={visibleDepth}
        onChange={(e) => setVisibleDepth(Number(e.target.value))}
      />
      <span style={{ fontSize: 9, color: TEAL.shade1 }}>
        {visibleDepth} / {treeDepth}
      </span>
      <button
        type="button"
        onClick={() => setVisibleDepth(treeDepth)}
        style={{
          fontSize: 9,
          color: TEAL.shade1,
          background: "none",
          border: `1px solid ${TEAL.base}4d`,
          borderRadius: 5,
          padding: "1px 6px",
          cursor: "pointer",
        }}
      >
        All
      </button>
    </div>
  );

  return (
    <div style={{ position: "relative", padding: "4px 10px 14px", overflowX: "auto" }}>
      {depthSlot ? createPortal(depthControls, depthSlot) : depthControls}
      <div style={{ width: "max-content", minWidth: "100%", margin: "0 auto", padding: "0 28px", boxSizing: "border-box" }}>
        <div style={{ display: "flex", justifyContent: "center" }}>
          <TreeNode
            nodeId={actualRootId}
            idx={idx}
            visibleDepth={visibleDepth}
            onNodeClick={onNodeClick}
            selectedNodeId={selectedNodeId}
            highlightNodeIds={highlightNodeIds}
            activeMatchNodeId={activeMatchNodeId}
            nodeAnnotations={nodeAnnotations}
          />
        </div>
      </div>
    </div>
  );
}

export default CfgHtmlTree;
