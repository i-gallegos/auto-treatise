import { GREEN, INK, ORANGE, TEAL } from "./theme";

export const ATTRIBUTION_STYLES = {
  CurrentCourt: {
    label: "CurrentCourt",
    color: TEAL.shade1,
    background: "rgba(89, 147, 160, 0.12)",
    border: "rgba(89, 147, 160, 0.35)",
  },
  Appellant: {
    label: "Appellant",
    color: ORANGE.shade2,
    background: "rgba(209, 110, 76, 0.12)",
    border: "rgba(209, 110, 76, 0.35)",
  },
  Appellee: {
    label: "Appellee",
    color: GREEN.shade2,
    background: "rgba(133, 165, 95, 0.12)",
    border: "rgba(133, 165, 95, 0.35)",
  },
  Amicus: {
    label: "Amicus",
    color: "#78445f",
    background: "rgba(159, 92, 127, 0.12)",
    border: "rgba(159, 92, 127, 0.35)",
  },
  LowerCourt: {
    label: "LowerCourt",
    color: "#967c28",
    background: "rgba(179, 148, 49, 0.12)",
    border: "rgba(179, 148, 49, 0.35)",
  },
};

export function getAttributionStyle(attribution) {
  if (attribution && ATTRIBUTION_STYLES[attribution]) {
    return ATTRIBUTION_STYLES[attribution];
  }
  return {
    label: attribution || "Unknown",
    color: INK.mid,
    background: "rgba(63, 63, 63, 0.1)",
    border: "rgba(63, 63, 63, 0.3)",
  };
}

/**
 * Collapse consecutive sentences that share the same (attribution, role, stance)
 * triple into a single entry — mirrors _format_segmented_paragraphs in utils.py.
 */
export function collapseSegments(sentences) {
  if (!sentences || sentences.length === 0) return [];
  const out = [];
  let cur = { ...sentences[0], text: sentences[0].text || "" };
  for (let i = 1; i < sentences.length; i += 1) {
    const s = sentences[i];
    if (
      s.attribution === cur.attribution &&
      s.role === cur.role &&
      s.stance === cur.stance
    ) {
      cur = { ...cur, text: cur.text + " " + (s.text || "") };
    } else {
      out.push(cur);
      cur = { ...s, text: s.text || "" };
    }
  }
  out.push(cur);
  return out;
}
