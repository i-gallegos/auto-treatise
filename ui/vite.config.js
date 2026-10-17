import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import Database from "better-sqlite3";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Allow import.meta.glob to read doctrines/*/analysis/**/*.json
// including analysis/test/runs/ written by `run.py --test`.
const repoRoot = path.resolve(__dirname, "..");
const doctrinesRoot = path.resolve(repoRoot, "doctrines");

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function safeDoctrineSlug(input) {
  const value = String(input || "").trim();
  if (!/^[a-z0-9_]+$/i.test(value)) {
    return null;
  }
  return value;
}

function resolveDoctrineDbPath(doctrine) {
  return path.resolve(doctrinesRoot, doctrine, "data", "citing_cases_clean.db");
}

// Raw fetch DBs are not named consistently across doctrines: some sit at
// data/citing_cases.db, others keep the seed opinion id in the filename
// (e.g. jaffee/opinion_118042_citing_cases.db at the doctrine root).
const RAW_DB_SEARCH_DIRS = ["data", "."];

function resolveDoctrineRawDbPath(doctrine) {
  const doctrineDir = path.resolve(doctrinesRoot, doctrine);
  for (const relativeDir of RAW_DB_SEARCH_DIRS) {
    const dir = path.resolve(doctrineDir, relativeDir);
    if (!fs.existsSync(dir)) {
      continue;
    }
    const names = fs
      .readdirSync(dir)
      .filter((name) => name.endsWith("citing_cases.db"))
      .sort();
    const preferred = names.includes("citing_cases.db") ? "citing_cases.db" : names[0];
    if (!preferred) {
      continue;
    }
    const fullPath = path.join(dir, preferred);
    // Some doctrines keep a zero-byte placeholder DB.
    if (fs.statSync(fullPath).size > 0) {
      return fullPath;
    }
  }
  return null;
}

function parseIntParam(value, defaultValue, minValue, maxValue) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) {
    return defaultValue;
  }
  return Math.max(minValue, Math.min(maxValue, parsed));
}

function getCaseColumns(db) {
  const rows = db.prepare("PRAGMA table_info(cases)").all();
  return new Set(rows.map((row) => row.name));
}

/**
 * `citation_depth` lives in the raw fetch DB, not the clean one: how many times
 * each opinion cites the seed. Duplicate rows for one opinion keep the deepest
 * reading, matching `datasets.splits.core.read_citation_depths`.
 */
function readCitationDepths(rawDbPath) {
  const depths = {};
  if (!rawDbPath) return depths;
  const rawDb = new Database(rawDbPath, { readonly: true });
  const cols = getCaseColumns(rawDb);
  if (!cols.has("courtlistener_opinion_id") || !cols.has("citation_depth")) {
    rawDb.close();
    return depths;
  }
  const rows = rawDb.prepare(
    "SELECT courtlistener_opinion_id, citation_depth FROM cases "
    + "WHERE courtlistener_opinion_id IS NOT NULL AND citation_depth IS NOT NULL",
  ).all();
  rawDb.close();
  for (const row of rows) {
    const opinionId = row.courtlistener_opinion_id === null
      || row.courtlistener_opinion_id === undefined
      || String(row.courtlistener_opinion_id).trim() === ""
      ? null
      : String(row.courtlistener_opinion_id).trim();
    if (!opinionId) continue;
    const depth = Number.parseInt(row.citation_depth, 10);
    if (!Number.isFinite(depth)) continue;
    depths[opinionId] = opinionId in depths ? Math.max(depths[opinionId], depth) : depth;
  }
  return depths;
}

function findRawOpinionText(rawDb, cleanCase) {
  const rawCols = getCaseColumns(rawDb);
  if (!rawCols.has("opinion_text")) {
    return null;
  }

  const attempts = [];
  if (rawCols.has("id") && cleanCase.id !== null && cleanCase.id !== undefined) {
    attempts.push({
      sql: "SELECT opinion_text FROM cases WHERE id = ? LIMIT 1",
      params: [cleanCase.id],
    });
  }
  if (rawCols.has("courtlistener_opinion_id") && cleanCase.courtlistener_opinion_id) {
    // Clean and raw DBs disagree on whether this column is TEXT or INTEGER.
    attempts.push({
      sql:
        "SELECT opinion_text FROM cases WHERE CAST(courtlistener_opinion_id AS TEXT) = CAST(? AS TEXT) LIMIT 1",
      params: [cleanCase.courtlistener_opinion_id],
    });
  }
  if (rawCols.has("case_name") && cleanCase.case_name) {
    attempts.push({
      sql: "SELECT opinion_text FROM cases WHERE case_name = ? LIMIT 1",
      params: [cleanCase.case_name],
    });
  }

  for (const attempt of attempts) {
    const row = rawDb.prepare(attempt.sql).get(...attempt.params);
    if (row && row.opinion_text) {
      return row.opinion_text;
    }
  }
  return null;
}

function dbBrowserPlugin() {
  return {
    name: "db-browser-api",
    configureServer(server) {
      server.middlewares.use("/api/db-browser/cases", (req, res) => {
        const url = new URL(req.url || "", "http://localhost");
        const doctrine = safeDoctrineSlug(url.searchParams.get("doctrine"));
        if (!doctrine) {
          sendJson(res, 400, { error: "Missing or invalid doctrine slug." });
          return;
        }

        const dbPath = resolveDoctrineDbPath(doctrine);
        if (!fs.existsSync(dbPath)) {
          sendJson(res, 404, { error: "Clean DB not found for doctrine." });
          return;
        }

        const page = parseIntParam(url.searchParams.get("page"), 1, 1, 1000000);
        const pageSize = parseIntParam(url.searchParams.get("pageSize"), 25, 1, 200);
        const search = String(url.searchParams.get("search") || "").trim();
        const segmentedOnly = ["1", "true", "yes"].includes(
          String(url.searchParams.get("segmented") || "").trim().toLowerCase(),
        );
        const offset = (page - 1) * pageSize;

        const db = new Database(dbPath, { readonly: true });
        const cols = getCaseColumns(db);

        const caseNameColumn = cols.has("case_name")
          ? "case_name"
          : cols.has("name")
            ? "name"
            : null;
        const orderColumn = cols.has("id") ? "id" : "rowid";

        const selectColumns = ["rowid"];
        for (const col of ["id", "courtlistener_opinion_id", "case_name", "decision_date"]) {
          if (cols.has(col)) {
            selectColumns.push(col);
          }
        }
        if (!selectColumns.includes("case_name") && caseNameColumn) {
          selectColumns.push(`${caseNameColumn} AS case_name`);
        }

        // Search matches the case name *or* either identifier, so the tree's
        // "find case" box accepts a permanent opinion id as readily as a name.
        // CAST keeps it working whether the column is TEXT or INTEGER, which the
        // clean and raw DBs disagree about.
        const where = [];
        const params = [];
        if (search) {
          const clauses = [];
          if (caseNameColumn) {
            clauses.push(`${caseNameColumn} LIKE ?`);
            params.push(`%${search}%`);
          }
          for (const col of ["id", "courtlistener_opinion_id"]) {
            if (cols.has(col)) {
              clauses.push(`CAST(${col} AS TEXT) LIKE ?`);
              params.push(`%${search}%`);
            }
          }
          if (clauses.length) {
            where.push(`(${clauses.join(" OR ")})`);
          }
        }
        if (segmentedOnly && cols.has("paragraph_sentence_segmentation_cache_json")) {
          where.push("paragraph_sentence_segmentation_cache_json IS NOT NULL");
        }
        const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";

        const total = db
          .prepare(`SELECT COUNT(*) AS count FROM cases ${whereSql}`)
          .get(...params).count;

        const rows = db
          .prepare(
            `SELECT ${selectColumns.join(", ")} FROM cases ${whereSql} ORDER BY ${orderColumn} LIMIT ? OFFSET ?`,
          )
          .all(...params, pageSize, offset);
        db.close();

        sendJson(res, 200, {
          doctrine,
          page,
          pageSize,
          total,
          cases: rows,
        });
      });

      // The whole case_id <-> opinion_id map plus court metadata, in one
      // request. The fallback for runs that predate `case_index.json`: it
      // describes the database *as it is now*, so the client labels anything
      // resolved through it as unverified.
      server.middlewares.use("/api/db-browser/case-index", (req, res) => {
        const url = new URL(req.url || "", "http://localhost");
        const doctrine = safeDoctrineSlug(url.searchParams.get("doctrine"));
        if (!doctrine) {
          sendJson(res, 400, { error: "Missing or invalid doctrine slug." });
          return;
        }

        const dbPath = resolveDoctrineDbPath(doctrine);
        if (!fs.existsSync(dbPath)) {
          sendJson(res, 404, { error: "Clean DB not found for doctrine." });
          return;
        }

        const db = new Database(dbPath, { readonly: true });
        const cols = getCaseColumns(db);
        // Mirrors auto_treatise/case_index.py::_COLUMNS, including the renames,
        // so both producers hand the client the same record shape.
        const wanted = [
          ["id", "case_id"],
          ["courtlistener_opinion_id", "opinion_id"],
          ["courtlistener_cluster_id", "cluster_id"],
          ["case_name", "case_name"],
          ["decision_date", "decision_date"],
          ["court_id", "court_id"],
          ["court_name", "court_name"],
          ["court_abbreviation", "court_abbreviation"],
          ["court_level", "court_level"],
          ["jurisdiction", "jurisdiction"],
          ["circuit", "circuit"],
          ["precedential_status", "precedential_status"],
          ["judge_author_party", "judge_author_party"],
        ].filter(([col]) => cols.has(col));

        const selectSql = wanted
          .map(([col, alias]) => (col === alias ? col : `${col} AS ${alias}`))
          .join(", ");
        const orderColumn = cols.has("id") ? "id" : "rowid";
        const rows = db
          .prepare(`SELECT ${selectSql} FROM cases ORDER BY ${orderColumn}`)
          .all();
        db.close();

        // Joined from the raw fetch DB so the tree's case drawer can rank
        // every case the way the annotation sample is ranked, not only the
        // rows that made it into selected_cases.csv.
        const citationDepths = readCitationDepths(resolveDoctrineRawDbPath(doctrine));
        const cases = rows.map((row) => {
          const opinionId = row.opinion_id === null || row.opinion_id === undefined
            || String(row.opinion_id).trim() === ""
            ? null
            : String(row.opinion_id).trim();
          const citationDepth = opinionId && opinionId in citationDepths
            ? citationDepths[opinionId]
            : null;
          return {
            ...row,
            case_id: row.case_id === null || row.case_id === undefined
              ? null
              : String(row.case_id),
            opinion_id: opinionId,
            citation_depth: citationDepth,
          };
        });

        sendJson(res, 200, {
          doctrine,
          dbPath,
          meta: {
            db_path: dbPath,
            db_row_count: cases.length,
            n_cases: cases.length,
            derived: "current_db",
          },
          cases,
          missing_opinion_id: cases
            .filter((row) => !row.opinion_id)
            .map((row) => row.case_id),
        });
      });

      server.middlewares.use("/api/db-browser/case", (req, res) => {
        const url = new URL(req.url || "", "http://localhost");
        const doctrine = safeDoctrineSlug(url.searchParams.get("doctrine"));
        if (!doctrine) {
          sendJson(res, 400, { error: "Missing or invalid doctrine slug." });
          return;
        }

        const dbPath = resolveDoctrineDbPath(doctrine);
        if (!fs.existsSync(dbPath)) {
          sendJson(res, 404, { error: "Clean DB not found for doctrine." });
          return;
        }

        // A case is addressable either way. `opinion_id` is preferred by the
        // client because it survives dataset regeneration; `id` (the sqlite
        // rowid) stays supported for rows that have no opinion id and for
        // artifacts written before the index existed.
        const idParam = url.searchParams.get("id");
        const opinionIdParam = url.searchParams.get("opinion_id");
        if (!idParam && !opinionIdParam) {
          sendJson(res, 400, { error: "Pass id or opinion_id." });
          return;
        }

        const cleanDb = new Database(dbPath, { readonly: true });
        const cleanCols = getCaseColumns(cleanDb);
        const cleanSelect = ["rowid"];
        for (const col of [
          "id",
          "courtlistener_opinion_id",
          "case_name",
          "decision_date",
          "opinion_text",
          "opinion_text_clean",
          "opinion_paragraphs_clean_json",
          "majority_boundary_json",
          "paragraph_relevance_cache_json",
          "paragraph_sentence_segmentation_cache_json",
          "case_symbol_profile_json",
          "citation",
          "docket_number",
          // Per-judge detail of each case's panel. Served here
          // rather than in case-index because it is a list per case and only
          // the drawer, which loads one case at a time, needs it.
          "judge_panel_json",
        ]) {
          if (cleanCols.has(col)) {
            cleanSelect.push(col);
          }
        }

        let cleanCase = null;
        if (opinionIdParam && cleanCols.has("courtlistener_opinion_id")) {
          cleanCase = cleanDb
            .prepare(
              `SELECT ${cleanSelect.join(", ")} FROM cases `
              + "WHERE CAST(courtlistener_opinion_id AS TEXT) = CAST(? AS TEXT) LIMIT 1",
            )
            .get(String(opinionIdParam));
        }
        if (!cleanCase && idParam) {
          const caseId = Number.parseInt(idParam, 10);
          cleanCase = cleanDb
            .prepare(`SELECT ${cleanSelect.join(", ")} FROM cases WHERE id = ? LIMIT 1`)
            .get(Number.isFinite(caseId) ? caseId : idParam);
        }
        if (!cleanCase) {
          cleanDb.close();
          sendJson(res, 404, { error: "Case not found in clean DB." });
          return;
        }

        // Some clean DBs carry the untouched opinion_text themselves; others
        // only keep the cleaned version and need the raw fetch DB.
        let rawOpinionText = cleanCase.opinion_text || null;
        const rawDbPath = resolveDoctrineRawDbPath(doctrine);
        if (!rawOpinionText && rawDbPath) {
          const rawDb = new Database(rawDbPath, { readonly: true });
          rawOpinionText = findRawOpinionText(rawDb, cleanCase);
          rawDb.close();
        }

        cleanDb.close();
        sendJson(res, 200, {
          doctrine,
          case: cleanCase,
          rawOpinionText,
        });
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), dbBrowserPlugin()],
  server: {
    fs: {
      allow: [repoRoot],
    },
  },
});
