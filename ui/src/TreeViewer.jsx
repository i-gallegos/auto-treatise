import { useEffect, useMemo, useState } from "react";
import { DoctrineTreeTab } from "./study/DoctrineTreeTab";
import {
  RUN_KIND_TEST,
  compareRunIdsDesc,
  parseCaseLevelPath,
  parseRunArtifactPath,
} from "./study/analysisPaths";
import { applyLabelOverlay, capitalizeGpt, parseLabelOverlayFile } from "./study/labelOverlay";
import { LabelOverlaySelect } from "./study/LabelOverlaySelect";
import { buildCaseIndex, fetchDbCaseIndex } from "./study/caseKeys";
import { buildSplitIndex, parseSplitPath, pickSplitFile } from "./study/splitMembership";

// Every directory under doctrines/ that has a config.toml. Keys only: the glob
// is lazy and never loads the files.
const doctrineConfigGlob = import.meta.glob("../../doctrines/*/config.toml", {
  query: "?url",
  import: "default",
});
const STUDY_DOCTRINES = Object.keys(doctrineConfigGlob)
  .map((path) => /doctrines\/([^/]+)\/config\.toml$/.exec(path)[1])
  .sort();
const PAPER_TITLE = "Auto-Treatise: Mapping Interpretive Ambiguity in Case Law";

/** `loper_bright` -> `Loper Bright`. */
function doctrineDisplayName(slug) {
  return slug
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

const PREFERRED_OVERLAY_PREFIX = "gpt-5.6-sol";

// Each glob lists the full-run path and the `run.py --test` path
// (`analysis/test/runs/`). Vite requires these as static string literals.
const cfgCorpusRunGlob = import.meta.glob(
  [
    "../../doctrines/*/analysis/runs/*/corpus_level.json",
    "../../doctrines/*/analysis/test/runs/*/corpus_level.json",
  ],
  { import: "default", eager: true }
);
const cfgCaseSharedGlob = import.meta.glob(
  [
    "../../doctrines/*/analysis/case_level.json",
    "../../doctrines/*/analysis/test/case_level.json",
  ],
  { import: "default", eager: true }
);
const cfgRunMetadataGlob = import.meta.glob(
  [
    "../../doctrines/*/analysis/runs/*/tree_run_metadata.json",
    "../../doctrines/*/analysis/test/runs/*/tree_run_metadata.json",
  ],
  { import: "default", eager: true }
);
const cfgUnassignedRunGlob = import.meta.glob(
  [
    "../../doctrines/*/analysis/runs/*/unassigned_case_ids.json",
    "../../doctrines/*/analysis/test/runs/*/unassigned_case_ids.json",
  ],
  { import: "default", eager: true }
);
const cfgLabelOverlayGlob = import.meta.glob(
  [
    "../../doctrines/*/analysis/runs/*/labels_*.json",
    "../../doctrines/*/analysis/test/runs/*/labels_*.json",
  ],
  { import: "default", eager: true }
);

// Citation graph payloads written by `citations.viz --artifact`. Lazy: one
// payload carries every node, edge and precomputed coordinate, which is
// megabytes on the larger corpora.
const citationGraphGlob = import.meta.glob(
  [
    "../../doctrines/*/analysis/runs/*/citation_graph.json",
    "../../doctrines/*/analysis/test/runs/*/citation_graph.json",
  ],
  { import: "default" }
);

// The run's own case_id -> opinion_id map, written by auto_treatise.tree beside
// corpus_level.json. Lazy: it carries a row per case with its court metadata,
// which is small per run but pointless to bundle for every run at once.
const caseIndexGlob = import.meta.glob(
  [
    "../../doctrines/*/analysis/runs/*/case_index.json",
    "../../doctrines/*/analysis/test/runs/*/case_index.json",
  ],
  { import: "default" }
);

// Temporal train/dev/test splits written by datasets.splits.apply. Eager: each
// file is a list of ids plus an embedded case_index, small enough to keep on
// hand so a clicked case can report which side it sits on.
const splitGlob = import.meta.glob(
  "../../doctrines/*/data/split*.json",
  { import: "default", eager: true }
);

/** doctrine -> run_id -> loader for citation_graph.json */
const CITATION_GRAPH_LOADERS = (() => {
  const out = {};
  for (const [path, loader] of Object.entries(citationGraphGlob)) {
    const parsed = parseRunArtifactPath(path);
    if (!parsed || parsed.file !== "citation_graph.json") continue;
    const { doctrine, runId } = parsed;
    if (!out[doctrine]) out[doctrine] = {};
    out[doctrine][runId] = loader;
  }
  return out;
})();

async function loadCitationGraph(doctrine, runId) {
  const loader = CITATION_GRAPH_LOADERS[doctrine]?.[runId];
  if (!loader) return null;
  try {
    return await loader();
  } catch {
    return null;
  }
}

/** doctrine -> run_id -> loader for case_index.json */
const CASE_INDEX_LOADERS = (() => {
  const out = {};
  for (const [path, loader] of Object.entries(caseIndexGlob)) {
    const parsed = parseRunArtifactPath(path);
    if (!parsed || parsed.file !== "case_index.json") continue;
    const { doctrine, runId } = parsed;
    if (!out[doctrine]) out[doctrine] = {};
    out[doctrine][runId] = loader;
  }
  return out;
})();

async function loadCaseIndexManifest(doctrine, runId) {
  const loader = CASE_INDEX_LOADERS[doctrine]?.[runId];
  if (!loader) return null;
  try {
    return await loader();
  } catch {
    return null;
  }
}

/** doctrine -> built split index (train/dev/test membership). */
const SPLITS_BY_DOCTRINE = (() => {
  const grouped = {};
  for (const [path, payload] of Object.entries(splitGlob)) {
    const parsed = parseSplitPath(path);
    if (!parsed) continue;
    if (!grouped[parsed.doctrine]) grouped[parsed.doctrine] = [];
    grouped[parsed.doctrine].push({ file: parsed.file, payload });
  }
  const out = {};
  for (const [doctrine, files] of Object.entries(grouped)) {
    const picked = pickSplitFile(files);
    if (!picked) continue;
    out[doctrine] = buildSplitIndex(picked.payload);
  }
  return out;
})();

/** doctrine slug -> run_id -> { corpus, metadata, unassigned, kind } */
const CFG_RUNS_BY_DOCTRINE = (() => {
  const out = {};
  const ensureRun = (doctrine, runId, kind) => {
    if (!out[doctrine]) out[doctrine] = {};
    if (!out[doctrine][runId]) {
      out[doctrine][runId] = {
        corpus: null,
        metadata: null,
        unassigned: [],
        overlays: {},
        kind,
      };
    } else if (kind === RUN_KIND_TEST) {
      // A smoke tree under analysis/test/ wins over a same-named leftover in
      // analysis/runs/, which is the older layout for subset runs.
      out[doctrine][runId].kind = kind;
    }
    return out[doctrine][runId];
  };

  for (const path of Object.keys(cfgCorpusRunGlob)) {
    const parsed = parseRunArtifactPath(path);
    if (!parsed || parsed.file !== "corpus_level.json") continue;
    ensureRun(parsed.doctrine, parsed.runId, parsed.kind).corpus =
      cfgCorpusRunGlob[path];
  }

  for (const path of Object.keys(cfgRunMetadataGlob)) {
    const parsed = parseRunArtifactPath(path);
    if (!parsed || parsed.file !== "tree_run_metadata.json") continue;
    ensureRun(parsed.doctrine, parsed.runId, parsed.kind).metadata =
      cfgRunMetadataGlob[path];
  }

  for (const path of Object.keys(cfgUnassignedRunGlob)) {
    const parsed = parseRunArtifactPath(path);
    if (!parsed || parsed.file !== "unassigned_case_ids.json") continue;
    const payload = cfgUnassignedRunGlob[path];
    ensureRun(parsed.doctrine, parsed.runId, parsed.kind).unassigned =
      Array.isArray(payload) ? payload : [];
  }

  for (const path of Object.keys(cfgLabelOverlayGlob)) {
    const parsed = parseRunArtifactPath(path);
    if (!parsed) continue;
    const spec = parseLabelOverlayFile(parsed.file);
    if (!spec) continue;
    const payload = cfgLabelOverlayGlob[path];
    if (!payload || typeof payload.nodes !== "object") continue;
    const run = ensureRun(parsed.doctrine, parsed.runId, parsed.kind);
    const model = "model" in payload && payload.model ? payload.model : spec.model;
    const tag = "tag" in payload && payload.tag ? payload.tag : spec.tag;
    run.overlays[tag] = { tag, model, payload, file: parsed.file };
  }

  return out;
})();

/** doctrine slug -> kind -> array of case-level CFG records */
const CFG_CASES_BY_DOCTRINE = (() => {
  const map = {};
  for (const path of Object.keys(cfgCaseSharedGlob)) {
    const parsed = parseCaseLevelPath(path);
    if (!parsed) continue;
    const payload = cfgCaseSharedGlob[path];
    if (!map[parsed.doctrine]) map[parsed.doctrine] = {};
    map[parsed.doctrine][parsed.kind] = Array.isArray(payload)
      ? payload
      : (payload ? [payload] : []);
  }
  return map;
})();

function isValidCfgGraph(raw) {
  if (!raw || typeof raw !== "object") return false;
  if (!Array.isArray(raw.nodes)) return false;
  if (raw.schema === "binary_dendrogram_v1") return true;
  return raw.nodes.length > 0;
}

export default function TreeViewer() {
  const [studyDoctrine, setStudyDoctrine] = useState(STUDY_DOCTRINES[0]);
  const [studyRun, setStudyRun] = useState("");
  // null until the user picks one; then the default (GPT-5.6) applies.
  const [labelOverlayChoice, setLabelOverlayChoice] = useState(null);
  const [cfgGraph, setCfgGraph] = useState(null);
  const [unassignedCases, setUnassignedCases] = useState([]);
  // Keyed by `${doctrine}/${run}` so "still loading" is derived from a key
  // mismatch rather than a separate flag set inside the effect body.
  const [citationCache, setCitationCache] = useState({ key: null, payload: null });
  const [caseIndexCache, setCaseIndexCache] = useState({ key: null, manifest: null });
  // Fallback map, per doctrine rather than per run: it describes the current DB.
  const [dbCaseIndexCache, setDbCaseIndexCache] = useState({ key: null, payload: null });

  const runsForDoctrine = CFG_RUNS_BY_DOCTRINE[studyDoctrine];
  const runOptions = useMemo(() => {
    return Object.keys(runsForDoctrine ?? {}).sort(compareRunIdsDesc);
  }, [studyDoctrine, runsForDoctrine]);

  const groupedRunOptions = useMemo(() => {
    const test = [];
    const full = [];
    for (const runId of runOptions) {
      if (runsForDoctrine?.[runId]?.kind === RUN_KIND_TEST) test.push(runId);
      else full.push(runId);
    }
    return { test, full };
  }, [runOptions, runsForDoctrine]);

  const selectedRun = studyRun ? (runsForDoctrine?.[studyRun] ?? null) : null;
  const runKind = selectedRun?.kind ?? "full";
  const overlayOptions = useMemo(() => {
    const overlays = selectedRun?.overlays ?? {};
    return Object.values(overlays).sort((a, b) => a.tag.localeCompare(b.tag));
  }, [selectedRun]);

  const labelOverlayTag = useMemo(() => {
    const tags = overlayOptions.map((opt) => opt.tag);
    if (labelOverlayChoice !== null && (labelOverlayChoice === "" || tags.includes(labelOverlayChoice))) {
      return labelOverlayChoice;
    }
    const preferred = tags.filter((tag) => tag.startsWith(PREFERRED_OVERLAY_PREFIX));
    return preferred.find((tag) => tag.endsWith("_tools")) ?? preferred[0] ?? "";
  }, [overlayOptions, labelOverlayChoice]);

  const displayGraph = useMemo(() => {
    if (!cfgGraph) return null;
    if (!labelOverlayTag) return cfgGraph;
    const spec = selectedRun?.overlays?.[labelOverlayTag];
    if (!spec) return cfgGraph;
    return applyLabelOverlay(cfgGraph, spec.payload);
  }, [cfgGraph, labelOverlayTag, selectedRun]);

  useEffect(() => {
    if (runOptions.length === 0) {
      setStudyRun("");
      return;
    }
    if (!runOptions.includes(studyRun)) {
      setStudyRun(runOptions[0]);
    }
  }, [runOptions, studyRun]);

  useEffect(() => {
    if (!studyRun) {
      setCfgGraph(null);
      setUnassignedCases([]);
      return;
    }
    const runPayload = CFG_RUNS_BY_DOCTRINE[studyDoctrine]?.[studyRun] ?? null;
    const raw = runPayload?.corpus ?? null;
    setCfgGraph(isValidCfgGraph(raw) ? raw : null);
    setUnassignedCases(Array.isArray(runPayload?.unassigned) ? runPayload.unassigned : []);
  }, [studyDoctrine, studyRun]);

  // Lazy artifacts are keyed on the doctrine/run pair, so "still loading" is a
  // key mismatch rather than a separate flag.
  const runKey = studyRun ? `${studyDoctrine}/${studyRun}` : null;

  useEffect(() => {
    let cancelled = false;
    const key = runKey;
    (key ? loadCitationGraph(...key.split("/")) : Promise.resolve(null)).then((payload) => {
      if (!cancelled) setCitationCache({ key, payload });
    });
    return () => {
      cancelled = true;
    };
  }, [runKey]);

  const citationLoading = citationCache.key !== runKey;
  const citationGraph = citationLoading ? null : citationCache.payload;

  useEffect(() => {
    let cancelled = false;
    const key = runKey;
    (key ? loadCaseIndexManifest(...key.split("/")) : Promise.resolve(null)).then(
      (manifest) => {
        if (!cancelled) setCaseIndexCache({ key, manifest });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [runKey]);

  useEffect(() => {
    let cancelled = false;
    const key = studyDoctrine;
    fetchDbCaseIndex(key).then((payload) => {
      if (!cancelled) setDbCaseIndexCache({ key, payload });
    });
    return () => {
      cancelled = true;
    };
  }, [studyDoctrine]);

  /**
   * The case_id <-> opinion_id map for the selected run, merged from the run's own
   * manifest, the citation graph and the live DB (in that order of trust). Every
   * tab resolves case identity through this rather than assuming a rowid means
   * the same thing in two artifacts.
   */
  const caseIndex = useMemo(
    () => buildCaseIndex({
      runManifest: caseIndexCache.key === runKey ? caseIndexCache.manifest : null,
      citationGraph,
      dbIndex: dbCaseIndexCache.key === studyDoctrine ? dbCaseIndexCache.payload : null,
    }),
    [caseIndexCache, runKey, citationGraph, dbCaseIndexCache, studyDoctrine],
  );
  const splitIndex = SPLITS_BY_DOCTRINE[studyDoctrine] ?? null;

  const cfgRecords = useMemo(() => {
    const byKind = CFG_CASES_BY_DOCTRINE[studyDoctrine];
    if (!byKind) return [];
    return byKind[runKind] ?? byKind.full ?? [];
  }, [studyDoctrine, runKind]);

  return (
    <div className="eval-root study-theme">
      <header className="study-app-header">
        <h1 className="study-title">{PAPER_TITLE}</h1>
        <div className="study-header-row1">
          <label className="study-field">
            <span className="study-field-label">Doctrine</span>
            <select
              className="study-field-select study-field-select--bold"
              value={studyDoctrine}
              onChange={(e) => setStudyDoctrine(e.target.value)}
            >
              {STUDY_DOCTRINES.map((d) => (
                <option key={d} value={d}>{doctrineDisplayName(d)}</option>
              ))}
            </select>
          </label>
          <LabelOverlaySelect
            value={labelOverlayTag}
            options={overlayOptions}
            onChange={setLabelOverlayChoice}
          />
          <label className="study-field">
            <span className="study-field-label">Run</span>
            <select
              className="study-field-select"
              value={studyRun}
              onChange={(e) => setStudyRun(e.target.value)}
              disabled={runOptions.length === 0}
            >
              {runOptions.length === 0 ? (
                <option value="">No runs found</option>
              ) : groupedRunOptions.test.length > 0
                && groupedRunOptions.full.length > 0 ? (
                <>
                  <optgroup label="Test (run.py --test)">
                    {groupedRunOptions.test.map((runId) => (
                      <option key={runId} value={runId}>{capitalizeGpt(runId)}</option>
                    ))}
                  </optgroup>
                  <optgroup label="Full">
                    {groupedRunOptions.full.map((runId) => (
                      <option key={runId} value={runId}>{capitalizeGpt(runId)}</option>
                    ))}
                  </optgroup>
                </>
              ) : (
                runOptions.map((runId) => (
                  <option key={runId} value={runId}>{capitalizeGpt(runId)}</option>
                ))
              )}
            </select>
          </label>
        </div>
      </header>

      <div className="study-main-pad">
        <DoctrineTreeTab
          studyDoctrine={studyDoctrine}
          studyRun={studyRun}
          runKind={runKind}
          cfgGraph={displayGraph}
          cfgRecords={cfgRecords}
          unassignedCases={unassignedCases}
          caseIndex={caseIndex}
          citationGraph={citationGraph}
          citationLoading={citationLoading}
          splitIndex={splitIndex}
        />
      </div>
    </div>
  );
}
