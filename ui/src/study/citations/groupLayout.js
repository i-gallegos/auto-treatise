function placeRings(positions, list, cx, cy, startRadius) {
  const perRing = 14;
  for (let offset = 0; offset < list.length; offset += perRing) {
    const slice = list.slice(offset, offset + perRing);
    const radius = startRadius + (offset / perRing) * 0.055;
    slice.forEach((node, i) => {
      const angle = -Math.PI / 2 + ((i + 0.5) * 2 * Math.PI) / slice.length;
      positions.set(node.i, [
        cx + radius * Math.cos(angle),
        cy + radius * Math.sin(angle),
      ]);
    });
  }
}

function placeCluster(positions, list, indeg, cx, cy, startRadius) {
  if (list.length === 0) return;
  const hub = list.reduce((best, node) => (
    (indeg.get(node.i) || 0) > (indeg.get(best.i) || 0) ? node : best
  ), list[0]);
  positions.set(hub.i, [cx, cy]);
  placeRings(
    positions,
    list.filter((node) => node.i !== hub.i),
    cx,
    cy,
    startRadius,
  );
}

export function layoutGroups(nodes, links, groupOf, focusIds = null) {
  const focused = (node) => !!focusIds?.has(node.i);
  const members = new Map();
  const focus = [];
  (nodes ?? []).forEach((node) => {
    if (focused(node)) {
      focus.push(node);
      return;
    }
    const key = groupOf(node) ?? "__none__";
    if (!members.has(key)) members.set(key, []);
    members.get(key).push(node);
  });
  const groups = [...members.entries()].sort(
    (a, b) => b[1].length - a[1].length || String(a[0]).localeCompare(String(b[0])),
  );
  const ids = new Set((nodes ?? []).map((node) => node.i));
  const indeg = new Map();
  (links ?? []).forEach((edge) => {
    if (!ids.has(edge.s) || !ids.has(edge.t)) return;
    indeg.set(edge.t, (indeg.get(edge.t) || 0) + 1);
  });

  const positions = new Map();
  // The marked circuit sits in one cluster. The other readings stay outside it,
  // so the circuit is a place on the canvas and not a ring on scattered nodes.
  placeCluster(positions, focus, indeg, 0, 0, 0.16);
  const count = Math.max(groups.length, 1);
  const orbit = focus.length ? 2.4 : 0.28;
  groups.forEach(([, list], index) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / count;
    const cx = focus.length ? orbit * Math.cos(angle) : 0.5 + (count === 1 ? 0 : orbit) * Math.cos(angle);
    const cy = focus.length ? orbit * Math.sin(angle) : 0.5 + (count === 1 ? 0 : orbit) * Math.sin(angle);
    placeCluster(positions, list, indeg, cx, cy, 0.1);
  });
  return fitPositions(positions);
}

/** Scale the whole layout into the canvas. A per-node clamp stacked overflow on the frame. */
function fitPositions(positions) {
  const points = [...positions.values()];
  if (points.length === 0) return positions;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  points.forEach(([x, y]) => {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  });
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const fitted = new Map();
  positions.forEach(([x, y], id) => {
    fitted.set(id, [
      0.08 + 0.84 * ((x - minX) / spanX),
      0.08 + 0.84 * ((y - minY) / spanY),
    ]);
  });
  return fitted;
}

/** Deepest listed group whose id sits on the case's root-to-leaf path. */
export function groupIdOf(path, groups) {
  let best = null;
  (groups ?? []).forEach((group) => {
    if (!(path ?? []).includes(group.id)) return;
    if (!best || group.id.length > best.id.length) best = group;
  });
  return best ? best.id : null;
}
