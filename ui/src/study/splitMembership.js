import { asKey, resolve } from "./caseKeys";

export const SPLIT_SIDES = ["train", "dev", "test"];

export function parseSplitPath(path) {
  const m = String(path).match(/doctrines\/([^/]+)\/data\/(split[^/]+\.json)$/);
  if (!m) return null;
  return { doctrine: m[1], file: m[2] };
}

/**
 * Pick one split file per doctrine. `split_q70.json` is what every doctrine
 * config currently names; if several files exist, that one wins.
 */
export function pickSplitFile(files) {
  if (!Array.isArray(files) || files.length === 0) return null;
  return files.find((f) => f.file === "split_q70.json") ?? files[0];
}

export function buildSplitIndex(payload) {
  if (!payload || typeof payload !== "object") return null;
  const byCaseId = new Map();
  const byOpinionId = new Map();
  const opinionByCaseId = new Map();
  for (const row of payload.case_index?.cases ?? []) {
    const caseId = asKey(row.case_id);
    const opinionId = asKey(row.opinion_id);
    if (caseId && opinionId) opinionByCaseId.set(caseId, opinionId);
  }
  const assign = (ids, side) => {
    for (const raw of ids ?? []) {
      const caseId = asKey(raw);
      if (!caseId) continue;
      byCaseId.set(caseId, side);
      const opinionId = opinionByCaseId.get(caseId);
      if (opinionId) byOpinionId.set(opinionId, side);
    }
  };
  for (const side of SPLIT_SIDES) assign(payload[side], side);
  assign(payload.undated, "undated");
  return {
    doctrine: payload.doctrine ?? null,
    tau: payload.split?.tau ?? null,
    byCaseId,
    byOpinionId,
  };
}

export function splitSideOf(index, { caseId, opinionId } = {}, caseIndex = null) {
  if (!index) return null;
  const cid = asKey(caseId);
  if (cid && index.byCaseId.has(cid)) return index.byCaseId.get(cid);
  const oid = asKey(opinionId);
  if (oid && index.byOpinionId.has(oid)) return index.byOpinionId.get(oid);
  const row = resolve(caseIndex, { caseId, opinionId });
  if (!row) return null;
  if (row.caseId && index.byCaseId.has(row.caseId)) return index.byCaseId.get(row.caseId);
  if (row.opinionId && index.byOpinionId.has(row.opinionId)) {
    return index.byOpinionId.get(row.opinionId);
  }
  return null;
}

export function splitSideLabel(side) {
  if (side === "train") return "train";
  if (side === "dev") return "dev";
  if (side === "test") return "test";
  if (side === "undated") return "undated (excluded)";
  return null;
}
