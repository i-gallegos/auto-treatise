export const RUN_KIND_FULL = "full";
export const RUN_KIND_TEST = "test";

const RUN_ARTIFACT_RE =
  /doctrines\/([^/]+)\/analysis\/(?:(test)\/)?runs\/([^/]+)\/(.+)$/;

const CASE_LEVEL_RE =
  /doctrines\/([^/]+)\/analysis\/(?:(test)\/)?case_level\.json$/;

export function parseRunArtifactPath(path) {
  const m = String(path).match(RUN_ARTIFACT_RE);
  if (!m) return null;
  return {
    doctrine: m[1],
    kind: m[2] === "test" ? RUN_KIND_TEST : RUN_KIND_FULL,
    runId: m[3],
    file: m[4],
  };
}

export function parseCaseLevelPath(path) {
  const m = String(path).match(CASE_LEVEL_RE);
  if (!m) return null;
  return {
    doctrine: m[1],
    kind: m[2] === "test" ? RUN_KIND_TEST : RUN_KIND_FULL,
  };
}

export function analysisRunDir(doctrine, runId, kind) {
  const analysis = kind === RUN_KIND_TEST ? "analysis/test" : "analysis";
  return `doctrines/${doctrine}/${analysis}/runs/${runId || "YYYY-MM-DD_full"}`;
}

export function analysisCaseLevelPath(doctrine, kind) {
  const analysis = kind === RUN_KIND_TEST ? "analysis/test" : "analysis";
  return `doctrines/${doctrine}/${analysis}/case_level.json`;
}

export function compareRunIdsDesc(a, b) {
  const am = String(a).match(/^(\d{4}-\d{2}-\d{2})_(.+)_(full|test)$/);
  const bm = String(b).match(/^(\d{4}-\d{2}-\d{2})_(.+)_(full|test)$/);
  if (am && bm) {
    const byDate = bm[1].localeCompare(am[1]);
    if (byDate !== 0) return byDate;
    if (am[3] === bm[3]) return 0;
    return am[3] === "full" ? 1 : -1;
  }
  return String(b).localeCompare(String(a));
}
