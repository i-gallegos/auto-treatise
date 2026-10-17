import { useLayoutEffect, useRef, useState } from "react";
import { CITER_COLOR, COURT_MATCH_COLOR, SELECTED_COLOR } from "./useCitationData";
import { axisLinesForRank } from "./rankBands";
import { displayNodeId } from "../nodeIds";

export const W = 1000;
export const H = 640;

const radiusOf = (inDeg) => 2.6 + Math.sqrt(inDeg || 0) * 1.7;
const GHOST_FILL = "#e6e6e6";

function coord(n, treeCoords, treeKey, fullKey) {
  if (treeCoords && Number.isFinite(n[treeKey])) return n[treeKey];
  return n[fullKey];
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

function medianDay(value) {
  const text = String(value).slice(0, 10);
  return text.length === 7 ? `${text}-15` : text;
}

function medianLabel(value) {
  const [year, month] = String(value).split("-");
  return `${MONTHS[Number(month) - 1]} ${year}`;
}

function utcDate(value) {
  const [year, month, day] = String(value).slice(0, 10).split("-").map(Number);
  return new Date(Date.UTC(year, (month || 1) - 1, day || 1));
}

const LEGEND_ITEMS = [
  { key: "vll", fill: "#5993a0", label: "Strict History" },
  { key: "vlr", fill: "#d16e4c", label: "Abstract Principles" },
  { key: "other", fill: "#e6e6e6", label: "Other" },
  { key: "cites", ring: true, label: "Cites ", italic: "Rahimi" },
];
const LEGEND_GAP = 36;

function legendFallbackWidth(item) {
  const text = `${item.label}${item.italic || ""}${item.nodeId ? ` (${displayNodeId(item.nodeId)})` : ""}`;
  return text.length * 11.2;
}

function canMeasureText() {
  const method = globalThis.SVGTextContentElement?.prototype?.getComputedTextLength;
  return typeof method === "function" && !String(method).includes("notImplemented");
}

function PlotLegend({ x0 }) {
  const refs = useRef([]);
  const [widths, setWidths] = useState(null);
  useLayoutEffect(() => {
    if (!canMeasureText()) return undefined;
    let cancelled = false;
    const measure = () => {
      if (cancelled) return;
      const next = LEGEND_ITEMS.map((item, i) => {
        const el = refs.current[i];
        const measured = el ? el.getComputedTextLength() : 0;
        return measured > 0 ? measured : legendFallbackWidth(item);
      });
      setWidths((prev) => (prev && prev.every((w, i) => w === next[i]) ? prev : next));
    };
    measure();
    document.fonts.ready.then(measure);
    return () => { cancelled = true; };
  }, []);
  let x = x0;
  const y = 20;
  return (
    <g className="study-cit-legend-plot">
      {LEGEND_ITEMS.map((item, i) => {
        const textW = widths ? widths[i] : legendFallbackWidth(item);
        const rowX = x;
        x += 22 + textW + LEGEND_GAP;
        return (
          <g key={item.key}>
            {item.ring ? (
              <circle className="study-cit-legend-ring" cx={rowX + 8} cy={y} r={8} />
            ) : (
              <circle cx={rowX + 8} cy={y} r={8} fill={item.fill} />
            )}
            <text
              ref={(el) => { refs.current[i] = el; }}
              className="study-cit-legend-plot-label"
             
              x={rowX + 22}
              y={y + 6}
            >
              {item.label}
              {item.italic ? <tspan className="study-cit-legend-em">{item.italic}</tspan> : null}
              {item.nodeId ? ` (${displayNodeId(item.nodeId)})` : null}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function TimeAxis({ axis, rect, marker = null, medians = null }) {
  const bands = axis.bands || [];
  const lo = utcDate(axis.date_min);
  const hi = utcDate(axis.date_max);
  const span = Math.max(hi - lo, 1);
  const xOf = (date) => rect.x + (0.02 + 0.975 * ((utcDate(date) - lo) / span)) * rect.w;
  const years = [];
  for (let year = lo.getUTCFullYear(); year <= hi.getUTCFullYear(); year += 1) {
    const t = new Date(Date.UTC(year, 0, 1));
    if (t < lo || t > hi) continue;
    years.push([year, xOf(`${year}-01-01`)]);
  }
  const markerX = marker?.date ? xOf(marker.date) : null;
  const lineTop = medians?.length ? 16 : rect.y;
  return (
    <g>
      {bands.map((band, i) => {
        const y0 = rect.y + (0.04 + 0.92 * band.y0) * rect.h;
        const y1 = rect.y + (0.04 + 0.92 * band.y1) * rect.h;
        const labelX = medians?.length ? rect.x - 10 : rect.x + 4;
        const lines = axisLinesForRank(band.rank, band.label);
        return (
          <g key={band.rank}>
            {i % 2 === 0 ? (
              <rect className="study-cit-band" x={rect.x} y={y0} width={rect.w} height={y1 - y0} />
            ) : null}
            <text className="study-cit-axis-text" x={labelX} y={y0 + 18} textAnchor={medians?.length ? "end" : "start"}>
              {lines.map((line, li) => (
                <tspan key={li} x={labelX} dy={li === 0 ? 0 : "1.15em"}>
                  {line}
                </tspan>
              ))}
            </text>
          </g>
        );
      })}
      {years.map(([year, x]) => (
        <g key={year}>
          <line className="study-cit-axis" x1={x} y1={rect.y} x2={x} y2={rect.y + rect.h} />
          <text className="study-cit-axis-text" x={x + 3} y={rect.y + rect.h - 3}>
            {year}
          </text>
        </g>
      ))}
      {markerX !== null ? (
        <g>
          <line
            className="study-cit-marker"
            x1={markerX}
            y1={lineTop}
            x2={markerX}
            y2={marker.bottom ?? rect.y + rect.h}
          />
          <text className="study-cit-marker-label" x={markerX + 10} y={medians?.length ? rect.y - 28 : rect.y + 22}>
            <tspan className="study-cit-legend-em">{marker.label || "Rahimi"}</tspan>
            {medians?.length ? (
              <tspan x={markerX + 10} dy="1em">{medianLabel(marker.date)}</tspan>
            ) : null}
          </text>
        </g>
      ) : null}
      {medians?.length ? <PlotLegend x0={rect.x + 8} /> : null}
      {(medians ?? []).map((median) => {
        const day = medianDay(median.date);
        const x = xOf(day);
        const placeLeft = marker?.date && day < marker.date;
        return (
          <g key={median.date}>
            <line
              className="study-cit-median"
              x1={x}
              y1={lineTop}
              x2={x}
              y2={marker?.bottom ?? rect.y + rect.h}
              stroke={median.color}
            />
            <text
              className="study-cit-median-label"
             
              x={placeLeft ? x - 12 : x + 12}
              y={medians?.length ? rect.y - 28 : rect.y + 22}
              textAnchor={placeLeft ? "end" : "start"}
              fill={median.color}
            >
              {medianLabel(median.date)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function monthsBetween(lo, hi) {
  const out = [];
  let y = lo.getUTCFullYear();
  let m = lo.getUTCMonth();
  const end = hi.getUTCFullYear() * 12 + hi.getUTCMonth();
  while (y * 12 + m <= end) {
    out.push(`${y}-${String(m + 1).padStart(2, "0")}`);
    m += 1;
    if (m === 12) { m = 0; y += 1; }
  }
  return out;
}

function tally(nodes, colorOf, markerDate) {
  const before = new Map();
  const after = new Map();
  const byMonth = new Map();
  for (const node of nodes) {
    if (!node.date) continue;
    const side = node.date < markerDate ? before : after;
    const color = colorOf(node);
    side.set(color, (side.get(color) || 0) + 1);
    const month = node.date.slice(0, 7);
    if (!byMonth.has(month)) byMonth.set(month, new Map());
    const bin = byMonth.get(month);
    bin.set(color, (bin.get(color) || 0) + 1);
  }
  const total = (map) => [...map.values()].reduce((sum, n) => sum + n, 0);
  return { before, after, byMonth, beforeN: total(before), afterN: total(after) };
}

function DateHistogram({ nodes, colorOf, axis, marker, medians = null, rect }) {
  const lo = utcDate(axis.date_min);
  const hi = utcDate(axis.date_max);
  const span = Math.max(hi - lo, 1);
  const xOf = (date) => rect.x + (0.02 + 0.975 * ((utcDate(date) - lo) / span)) * rect.w;
  const months = monthsBetween(lo, hi);
  const barSpan = (month) => {
    const x0 = xOf(`${month}-01`);
    const [year, mon] = month.split("-").map(Number);
    const next = mon === 12 ? `${year + 1}-01-01` : `${year}-${String(mon + 1).padStart(2, "0")}-01`;
    return [x0, Math.max((xOf(next) - x0) * 0.78, 1)];
  };
  const counts = tally(nodes, colorOf, marker.date);
  const base = rect.y + rect.h - 2;
  const room = rect.h - 8;
  const labelX = rect.x - 10;
  return (
    <g className="study-cit-hist">
      <text className="study-cit-axis-text" x={labelX} y={rect.y + 22} textAnchor="end">
        <tspan x={labelX}>Proportion of</tspan>
        <tspan x={labelX} dy="1.15em">Monthly cases</tspan>
      </text>
      {months.map((month) => {
        const [x0, width] = barSpan(month);
        const bin = counts.byMonth.get(month) ?? new Map();
        const total = [...bin.values()].reduce((sum, n) => sum + n, 0);
        let y = base;
        const parts = [...bin.entries()].sort((a, b) => a[0].localeCompare(b[0]));
        return (
          <g key={month}>
            {parts.map(([color, n]) => {
              const h = total ? (n / total) * room : 0;
              y -= h;
              return (
                <rect key={color} className="study-cit-hist-bar" x={x0} y={y} width={width} height={h} fill={color} />
              );
            })}
          </g>
        );
      })}
    </g>
  );
}

export function CitationScene({
  layout = "force",
  data,
  colorOf,
  nodeVisible,
  edgeVisible,
  height = H,
  width = W,
  title,
  focusIndex = null,
  citers = null,
  selectedIndex = null,
  courtMatches = null,
  dimOutside = null,
  fadedIndices = null,
  treeCoords = false,
  timeAxis = null,
  marker = null,
  medians = null,
  ringIndices = null,
  ringCaption = null,
  degreeOf = null,
  outDegreeOf = null,
  inboundOnly = false,
  ghostIndices = null,
  positions = null,
  hubLabel = null,
  plainEdges = false,
  fadeOutsideRings = false,
  onHoverNode,
  onClickNode,
  onBackgroundClick,
  children,
  clipId = "study-cit-clip",
}) {
  const { nodes, links, outAdj, inAdj, treeNodes } = data;
  // Edges must use these same objects. A figure recompresses `nodes` without
  // rebuilding the payload's index, and the degree slider then leaves lines
  // meeting where the circle used to be.
  const axis = timeAxis ?? data.timeAxis;
  // Destructured rather than read as `zoom.svgRef` at each use: the refs are
  // handed straight to `ref=` and to event handlers, never read during render.
  const showHist = layout === "time" && !!marker?.date;
  const histH = showHist ? 108 : 0;
  const courtGutter = medians?.length ? 236 : 0;
  // The dated figure draws labels and rules in the margin. Keep them off the
  // viewBox edge, or the export clips the caption, the rules, and the bars.
  const plotTop = medians?.length ? 72 : 26;
  const plotRight = medians?.length ? 28 : 40;
  const bottomPad = medians?.length ? 28 : 40;
  const rect = {
    x: (medians?.length ? 36 : 40) + courtGutter,
    y: plotTop,
    w: width - ((medians?.length ? 36 : 40) + courtGutter) - plotRight,
    h: height - histH - plotTop - bottomPad,
  };
  const histRect = {
    x: rect.x, y: rect.y + rect.h + 6, w: rect.w, h: Math.max(histH - 10, 0),
  };

  const posOf = (n) => {
    const placedHere = positions?.get(n.i);
    const nx = placedHere
      ? placedHere[0]
      : coord(n, treeCoords, layout === "time" ? "tx_tree" : "fx_tree", layout === "time" ? "tx" : "fx");
    const ny = placedHere
      ? placedHere[1]
      : coord(n, treeCoords, layout === "time" ? "ty_tree" : "fy_tree", layout === "time" ? "ty" : "fy");
    const x = layout === "time"
      ? rect.x + (0.02 + 0.975 * ((nx - 0.02) / 0.96)) * rect.w
      : rect.x + nx * rect.w;
    return [x, rect.y + ny * rect.h];
  };

  const hasFocus = focusIndex !== null && focusIndex !== undefined;
  const hasCiters = !!citers;

  const inCiterSet = (index) => index === selectedIndex || (citers?.has(index) ?? false);

  const placed = new Map((nodes ?? []).map((n) => [n.i, n]));
  const edgeElements = [];
  (links ?? []).forEach((e, k) => {
    if (!edgeVisible(e)) return;
    const a = placed.get(e.s);
    const b = placed.get(e.t);
    if (!a || !b) return;
    const [x1, y1] = posOf(a);
    const [x2, y2] = posOf(b);
    const mx = (x1 + x2) / 2 + (y2 - y1) * 0.08;
    // A long citation bows upward. Keep the bend inside the plot so it does
    // not leave the top and read as a court band that is not drawn.
    const my = Math.min(
      rect.y + rect.h,
      Math.max(rect.y, (y1 + y2) / 2 - (x2 - x1) * 0.08),
    );
    let cls = plainEdges ? "study-cit-edge plain" : "study-cit-edge";
    if (hasFocus) {
      if (e.s === focusIndex) cls += " hl-out";
      else if (e.t === focusIndex) cls += " hl-in";
      else cls += " dim";
    } else if (hasCiters) {
      // The citations into the selected case are the thing being traced.
      // `inboundOnly` keeps citer-to-citer edges dim: the figure is the star
      // around one opinion, not the subgraph of its citers.
      const inbound = e.t === selectedIndex && citers.has(e.s);
      const among = inCiterSet(e.s) && inCiterSet(e.t);
      cls += (inboundOnly ? inbound : among) ? " hl-in" : " dim";
    } else if (courtMatches) {
      // A citation between two selected courts stays lit: that is the "do these
      // courts cite each other" question the filter is usually asked for.
      cls += courtMatches.has(e.s) && courtMatches.has(e.t) ? "" : " dim";
    } else if (dimOutside && !(dimOutside.has(e.s) && dimOutside.has(e.t))) {
      cls += " dim";
    } else if (fadeOutsideRings && ringIndices && !ringIndices.has(e.s) && !ringIndices.has(e.t)) {
      cls += " circuit-bg";
    }
    // A citation into or out of a faded case is faded with it.
    if (fadedIndices && !cls.includes("dim")
      && (fadedIndices.has(e.s) || fadedIndices.has(e.t))) {
      cls += " faded";
    }
    edgeElements.push(
      <path
        key={k}
        className={cls}
        strokeWidth={Math.min(1 + Math.log(e.d || 1), 3)}
        d={`M${x1.toFixed(1)},${y1.toFixed(1)}Q${mx.toFixed(1)},${my.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`}
      />,
    );
  });

  const visibleNodes = (nodes ?? [])
    .filter(nodeVisible)
    .sort((a, b) => a.in_deg - b.in_deg);
  const ghostNodes = ghostIndices
    ? (nodes ?? []).filter((n) => ghostIndices.has(n.i) && !nodeVisible(n))
    : [];

  // Rings drawn around the clicked case, outside its circle. A separate element
  // rather than a fatter stroke or a bigger radius: node radius encodes in-degree,
  // so growing the selected node would say something false about it.
  const haloElements = [];
  const glowElements = [];

  const nodeElements = visibleNodes.map((n) => {
    const [x, y] = posOf(n);
    let cls = "study-cit-node";
    let fill = colorOf(n);
    let stroke;
    let strokeWidth;
    let picked = false;

    if (hasFocus) {
      const near = n.i === focusIndex
        || (outAdj.get(focusIndex) || []).some((k) => links[k].t === n.i)
        || (inAdj.get(focusIndex) || []).some((k) => links[k].s === n.i);
      if (!near) cls += " dim";
      if (n.i === focusIndex) {
        cls += " sel";
        picked = true;
        fill = SELECTED_COLOR;
      }
    } else if (hasCiters) {
      if (n.i === selectedIndex) {
        cls += " sel";
        picked = true;
        fill = SELECTED_COLOR;
      } else if (citers.has(n.i)) {
        fill = CITER_COLOR;
      } else {
        cls += " dim";
      }
    } else if (courtMatches) {
      if (courtMatches.has(n.i)) {
        // Keeps whatever `colorOf` says — the colour-by dropdown still means what
        // it means — and adds a ring, so "selected" is a separate channel from
        // "coloured by court level".
        stroke = COURT_MATCH_COLOR;
        strokeWidth = 1.4;
      } else {
        cls += " dim";
      }
    } else if (dimOutside && !dimOutside.has(n.i)) {
      cls += " dim";
    }

    if (fadedIndices?.has(n.i) && !cls.includes("dim") && !cls.includes("sel")) {
      cls += " faded";
    }
    if (fadeOutsideRings && ringIndices && !ringIndices.has(n.i) && !cls.includes("circuit-bg")) {
      cls += " circuit-bg";
    }

    const matchedByCourt = !!courtMatches?.has(n.i);
    const ringed = !!ringIndices?.has(n.i);
    // Size stays the in-degree encoding. Selection is carried by colour and by the
    // halo below, never by radius.
    const citedBy = degreeOf?.(n) ?? n.in_deg ?? 0;
    const cites = outDegreeOf?.(n) ?? n.out_deg ?? 0;
    const r0 = radiusOf(citedBy) * (matchedByCourt ? 1.35 : 1);

    if (picked) {
      const ring = r0 + 3.5;
      haloElements.push(
        <circle
          key={`halo-${n.i}`}
          className="study-cit-halo"
          cx={x.toFixed(1)}
          cy={y.toFixed(1)}
          r={ring}
          stroke={SELECTED_COLOR}
        />,
      );
    } else if (ringed) {
      const glow = r0 + 3.4;
      glowElements.push(
        <circle
          key={`glow-${n.i}`}
          className="study-cit-cite-glow"
          cx={x.toFixed(1)}
          cy={y.toFixed(1)}
          r={glow}
        />,
      );
      const ring = r0 + 4.4;
      haloElements.push(
        <circle
          key={`cite-${n.i}`}
          className="study-cit-cite-ring"
          cx={x.toFixed(1)}
          cy={y.toFixed(1)}
          r={ring}
        />,
      );
    }

    return (
      <circle
        key={n.i}
        className={cls}
        cx={x.toFixed(1)}
        cy={y.toFixed(1)}
        r={r0}
        fill={fill}
        stroke={stroke}
        strokeWidth={strokeWidth}
        onMouseEnter={() => onHoverNode?.(n.i)}
        onMouseLeave={() => onHoverNode?.(null)}
        onClick={(ev) => {
          ev.stopPropagation();
          onClickNode?.(n.i);
        }}
      >
        <title>
          {`${n.name}\n${n.court ?? "?"} · ${n.date ?? "?"}`}
          {n.leaf ? `\n${treeNodes.get(n.leaf)?.label ?? n.leaf}` : ""}
          {`\ncites ${cites} in corpus · cited by ${citedBy}`}
          {ringed ? `\n${ringCaption || "cites the marked authority"}` : ""}
          {citers?.has(n.i) ? "\ncites the selected case" : ""}
          {matchedByCourt ? "\nmatches the court filter" : ""}
          {fadedIndices?.has(n.i) ? "\nbelow the threshold — cites a case above it" : ""}
        </title>
      </circle>
    );
  });

  const hubLabels = hubLabel && selectedIndex !== null && selectedIndex !== undefined
    ? visibleNodes.filter((n) => n.i === selectedIndex).map((n) => {
      const [x, y] = posOf(n);
      const citedBy = degreeOf?.(n) ?? n.in_deg ?? 0;
      const r0 = radiusOf(citedBy);
      return (
        <text
          key={`hub-label-${n.i}`}
          className="study-cit-hub-label"
         
          x={x}
          y={y + r0 + 16}
          textAnchor="middle"
        >
          {hubLabel}
        </text>
      );
    })
    : [];

  const ghostElements = ghostNodes.map((n) => {
    const [x, y] = posOf(n);
    const citedBy = degreeOf?.(n) ?? n.in_deg ?? 0;
    const r0 = radiusOf(citedBy);
    return (
      <circle
        key={`ghost-${n.i}`}
        className="study-cit-node ghost"
        cx={x.toFixed(1)}
        cy={y.toFixed(1)}
        r={r0}
        fill={GHOST_FILL}
        onMouseEnter={() => onHoverNode?.(n.i)}
        onMouseLeave={() => onHoverNode?.(null)}
        onClick={(ev) => {
          ev.stopPropagation();
          onClickNode?.(n.i);
        }}
      >
        <title>
          {`${n.name}\n${n.court ?? "?"} · ${n.date ?? "?"}`}
          {n.leaf ? `\n${treeNodes.get(n.leaf)?.label ?? n.leaf}` : ""}
          {"\noutside the colored branch"}
        </title>
      </circle>
    );
  });

  return (
    <svg
      className="study-cit-svg"
      style={{ height }}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      onClick={(ev) => {
        if (ev.target === ev.currentTarget) onBackgroundClick?.();
      }}
    >
      <defs>
        <clipPath id={clipId}>
          <rect
            x={0}
            y={0}
            width={width}
            height={height}
          />
        </clipPath>
      </defs>

      {children}

      {title && !medians?.length ? (
        <text className="study-cit-panel-title" x={rect.x} y={18}>{title}</text>
      ) : null}

      <g clipPath={`url(#${clipId})`}>
        <g>
          {layout === "time" && axis ? (
            <TimeAxis
              axis={axis}
              rect={rect}
              marker={showHist ? { ...marker, bottom: histRect.y + histRect.h } : marker}
              medians={medians}
            />
          ) : null}
          <g>{edgeElements}</g>
          <g>{ghostElements}</g>
          <g>{glowElements}</g>
          <g>{nodeElements}</g>
          <g>{hubLabels}</g>
          {/* After the nodes, so a neighbouring circle cannot paint over the ring. */}
          <g>{haloElements}</g>
          {showHist && axis ? (
            <DateHistogram
              nodes={visibleNodes}
              colorOf={colorOf}
              axis={axis}
              marker={marker}
              medians={medians}
              rect={histRect}
            />
          ) : null}
        </g>
      </g>
    </svg>
  );
}

export default CitationScene;
