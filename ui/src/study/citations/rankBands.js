const Y0 = 0.04;
const YH = 0.92;

export const RANK_AXIS_LINES = {
  4: ["Supreme Court"],
  3: ["Federal appellate court", "State supreme court"],
  2: ["Federal district court", "State appellate court"],
  1: ["State trial court"],
  0: ["Other"],
};

export const RANK_LABELS = {
  4: "Supreme Court",
  3: "Federal appellate court / State supreme court",
  2: "Federal district court / State appellate court",
  1: "State trial court",
  0: "Other",
};

/** Lines to draw on the chronology y-axis for one rank band. */
export function axisLinesForRank(rank, fallback) {
  const lines = RANK_AXIS_LINES[rank];
  if (lines) return lines;
  if (fallback) return String(fallback).split("\n").filter(Boolean);
  return [`rank ${rank}`];
}

/** Rank used for the chronology and for colour-by-rank. */
export function displayRank(node) {
  if (!node) return 0;
  if (node.court_level === "federal_bankruptcy") return 0;
  const rank = Number(node.rank);
  return Number.isFinite(rank) ? rank : 0;
}

function innerY(ty) {
  return (Number(ty) - Y0) / YH;
}

function paddedY(inner) {
  return Y0 + YH * inner;
}

function bandForRank(bands, rank) {
  return (bands ?? []).find((band) => band.rank === rank) ?? null;
}

function fractionInBand(ty, band) {
  if (!band) return 0.5;
  const span = band.y1 - band.y0;
  if (!(span > 0)) return 0.5;
  const frac = (innerY(ty) - band.y0) / span;
  return Math.min(1, Math.max(0, frac));
}

/**
 * Remap bankruptcy onto Other and rebuild the time-axis bands so a payload
 * laid out under the old shared rank-1 band still draws correctly.
 *
 * @returns {{ nodes: object[], timeAxis: object | null }}
 */
export function applyChronologyRanks(payload) {
  const source = payload?.nodes;
  if (!Array.isArray(source) || source.length === 0) {
    return { nodes: source ?? [], timeAxis: payload?.time_axis ?? null };
  }

  const oldBands = payload.time_axis?.bands ?? [];
  const ranked = source.map((node) => {
    const rank = displayRank(node);
    return rank === node.rank ? node : { ...node, rank };
  });

  const ranks = [...new Set(ranked.map((node) => node.rank))].sort((a, b) => b - a);
  const nBands = Math.max(ranks.length, 1);
  const bands = ranks.map((rank, i) => ({
    rank,
    label: RANK_LABELS[rank] ?? `rank ${rank}`,
    y0: i / nBands,
    y1: (i + 1) / nBands,
  }));
  const byRank = new Map(bands.map((band) => [band.rank, band]));

  const nodes = ranked.map((node, i) => {
    const old = source[i];
    const oldBand = bandForRank(oldBands, old.rank);
    const newBand = byRank.get(node.rank);
    const frac = fractionInBand(old.ty, oldBand);
    const ty = paddedY(newBand.y0 + frac * (newBand.y1 - newBand.y0));
    if (ty === node.ty) return node;
    return { ...node, ty };
  });

  const timeAxis = payload.time_axis
    ? { ...payload.time_axis, bands }
    : { date_min: null, date_max: null, bands };

  return { nodes, timeAxis };
}

/**
 * Keep a few authority ranks and stretch them to fill the chronology.
 * Nodes outside `keepRanks` are left where they are; the caller hides them.
 */
export function restrictChronologyRanks(nodes, timeAxis, keepRanks) {
  const keep = [...new Set((keepRanks ?? []).map(Number))].sort((a, b) => b - a);
  if (keep.length === 0) return { nodes, timeAxis };
  const nBands = keep.length;
  const bands = keep.map((rank, i) => ({
    rank,
    label: RANK_LABELS[rank] ?? `rank ${rank}`,
    y0: i / nBands,
    y1: (i + 1) / nBands,
  }));
  const oldBands = timeAxis?.bands ?? [];
  const byRank = new Map(bands.map((band) => [band.rank, band]));
  const nextNodes = nodes.map((node) => {
    const newBand = byRank.get(displayRank(node));
    if (!newBand) return node;
    const oldBand = bandForRank(oldBands, displayRank(node));
    const remap = (ty) => {
      const value = Number(ty);
      if (!Number.isFinite(value)) return ty;
      const frac = fractionInBand(value, oldBand);
      return paddedY(newBand.y0 + frac * (newBand.y1 - newBand.y0));
    };
    const ty = remap(node.ty);
    const tyTree = remap(node.ty_tree);
    if (ty === node.ty && tyTree === node.ty_tree) return node;
    return { ...node, ty, ty_tree: tyTree };
  });
  const nextAxis = timeAxis ? { ...timeAxis, bands } : timeAxis;
  return { nodes: nextNodes, timeAxis: nextAxis };
}
