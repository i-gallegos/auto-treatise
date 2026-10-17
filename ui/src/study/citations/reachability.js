export function buildAdjacency(links) {
  const out = new Map();   // s -> [t]  (cases it cites)
  const inn = new Map();   // t -> [s]  (cases citing it)
  for (const edge of links ?? []) {
    if (!out.has(edge.s)) out.set(edge.s, []);
    if (!inn.has(edge.t)) inn.set(edge.t, []);
    out.get(edge.s).push(edge.t);
    inn.get(edge.t).push(edge.s);
  }
  return { out, inn };
}

function neighbours(adjacency, indices) {
  const found = new Set();
  const roots = new Set(
    (Array.isArray(indices) ? indices : [indices])
      .filter((i) => i !== null && i !== undefined),
  );
  for (const root of roots) {
    for (const neighbour of adjacency?.get(root) ?? []) {
      // A case in the starting set is not its own relation, even when two of the
      // starting cases cite each other.
      if (!roots.has(neighbour)) found.add(neighbour);
    }
  }
  return found;
}

/** The cases that cite `index` directly. */
export function citersOf(adjacency, index) {
  return neighbours(adjacency?.inn, index);
}

/** The cases that directly cite any of `indices` — one traversal, de-duplicated. */
export function citersOfAny(adjacency, indices) {
  return neighbours(adjacency?.inn, [...(indices ?? [])]);
}

