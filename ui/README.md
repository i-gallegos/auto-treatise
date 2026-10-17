# Auto-treatise UI

React + Vite app for viewing the doctrine tree built by the pipeline: the induced
tree, the citation chronology beneath it, and a drawer for each case.

## Run locally

```bash
cd ui
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`). The doctrine list is every
`doctrines/<d>/` directory that has a `config.toml`. Run artifacts under
`doctrines/<d>/analysis/` and splits under `doctrines/<d>/data/` are loaded with
`import.meta.glob`; case text and the case index come from the dev-server middleware in
`vite.config.js`, which reads `doctrines/<d>/data/citing_cases_clean.db`. That middleware
only runs under `npm run dev`, not in a `vite build` bundle.

Other scripts: `npm test`, `npm run lint`, `npm run build`.

## Case identity: `case_id` vs `opinion_id`

Two ids name the same opinion and they are **not** interchangeable.

| id | what it is | stability |
|---|---|---|
| `case_id` | the sqlite rowid of `citing_cases_clean.db`; what the tree artifacts contain | **reassigned on every dataset regeneration** |
| `opinion_id` | `courtlistener_opinion_id`, the permanent CourtListener id; what `citations/` keys on | permanent |

So a `corpus_level.json` written last month and a database rebuilt today disagree about
which case `"12"` is — silently, with no error anywhere. The fix is that each run records
its own map: `auto_treatise.tree` writes `case_index.json` next to `corpus_level.json`,
from the same database, in the same process.

The UI resolves case identity through `src/study/caseKeys.js`, which merges, in descending
order of trust:

1. `doctrines/<d>/analysis/runs/<run>/case_index.json` — recorded by the run. Authoritative.
2. `citation_graph.json` — carries both ids per node, plus authority rank.
3. `GET /api/db-browser/case-index?doctrine=` — the database as it is right now.

A chip beside the run selector says which layer is in force, how many cases are indexed,
and how many case ids the run references that the database no longer has. Amber means the
join was re-derived rather than recorded, so treat it as unverified. Backfill older runs:

```bash
python -m auto_treatise.case_index --all        # add --force to overwrite
```

Case text is fetched by `opinion_id` when one exists, falling back to `id` — rows with no
opinion id stay addressable by rowid, which is their only handle.

## Tree view

The induced tree on top, the citation chronology below it, split by a draggable handle
(position remembered in `localStorage`). Two organizations of the same cases, stacked so
one selection reads across both.

**Grammar tree** from per-run `corpus_level.json` (and shared case-level symbol profiles
from `case_level.json`). Click nodes to inspect descriptions and open the case drawer.
