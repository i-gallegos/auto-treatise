export const ROOT_ID = "n0";

const FILE_RE = /^labels_(.+)\.json$/;

/**
 * @param {string} file  basename, e.g. `labels_gpt-5.6-sol.json`
 * @returns {{ tag: string, model: string } | null}
 */
export function parseLabelOverlayFile(file) {
  const m = String(file).match(FILE_RE);
  if (!m) return null;
  return { tag: m[1], model: m[1] };
}

export function originalLabelSuffix(node) {
  const label = node?.label;
  const original = node?.original_label;
  if (!original || original === label) return "";
  return original;
}

/** Shows model ids with the product spelling: "gpt-5.6-sol" becomes "GPT-5.6-sol". */
export function capitalizeGpt(text) {
  return String(text).replace(/(^|[^A-Za-z])gpt(?![A-Za-z])/gi, "$1GPT");
}

export function overlayDisplayName(option) {
  if (!option) return "Original";
  return capitalizeGpt(option.model || option.tag || "overlay");
}

/**
 * @param {object} corpus  corpus_level.json payload
 * @param {object|null} overlay  labels_*.json payload
 * @returns {object} shallow-copied corpus; `corpus` is not mutated
 */
export function applyLabelOverlay(corpus, overlay) {
  if (!overlay || !corpus?.nodes) return corpus;
  const nodes = corpus.nodes.map((node) => ({ ...node }));
  const byId = {};
  for (const node of nodes) byId[node.id] = node;
  const patches = overlay.nodes && typeof overlay.nodes === "object" ? overlay.nodes : {};
  for (const nid of Object.keys(patches)) {
    if (!(nid in byId)) continue;
    const patch = patches[nid];
    const node = byId[nid];
    if (nid !== ROOT_ID) {
      if ("label" in patch) {
        node.original_label = node.label;
        node.label = patch.label;
      }
      if ("summary" in patch) node.summary = patch.summary;
      if ("description" in patch) {
        node.description = patch.description;
        const parentId = node.parent_id;
        if (parentId && parentId in byId) {
          const parent = byId[parentId];
          if (node.id === parent.left_id) parent.left_description = patch.description;
          else if (node.id === parent.right_id) parent.right_description = patch.description;
        }
      }
    }
    if ("split_criterion_short" in patch) {
      node.split_criterion_short = patch.split_criterion_short;
    }
  }
  return { ...corpus, nodes };
}
