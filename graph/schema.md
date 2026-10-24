# Research graph schema, version 2

`graph/graph.json` describes the Phase 2 research record. The release contains 3,138 nodes and 158 recovered submissions dated from 2026-08-22 through 2026-10-22. All 158 submitted estimator sources were located; 93 original submission archives were recovered. The graph retains failed, rejected and inconclusive work as well as adopted changes.

Dates determine order. Components identify mathematical objects, implementation methods or evaluation constraints; their definitions are in [components.json](components.json). An original record that provides only a calendar date retains that precision. A graph date is not a claim of a more precise timestamp than its evidence supports.

## Top level

| Field | Meaning |
|---|---|
| `meta` | Schema version, assembly time, frozen source path and Git commit, research objective |
| `components` | Component objects with `id`, `title` and `what` |
| `nodes` | Research and submission records |
| `edges` | Directed relationships between node IDs |
| `submissions` | Official submission records, source hashes and local artifact paths |
| `best_chain` | Submission node IDs that improved the recorded public score without failed networks |

## Nodes

| Field | Meaning |
|---|---|
| `id` | Unique date plus descriptive slug; submissions use `submission-<id>` |
| `directory` | Repository directory holding the generated `README.md`; `document` identifies an explicit alternate name |
| `kind` | `submission`, `experiment`, `decision`, `finding`, `lesson`, `incident` or `tool` |
| `title` | At most 120 characters, stating the mathematical or engineering object and result |
| `date` | ISO calendar date or UTC timestamp, at the precision available in the source |
| `component` | An ID from `components.json` |
| `outcome` | `adopted`, `rejected`, `inconclusive`, `failed` or `informational` |
| `hedge` | `derived`, `measured` or `conjectured`; the basis for the claim |
| `claim` | Description of the result, decision, lesson or incident |
| `measurement` | Recorded numerical statements with their reference, or `null` |
| `actor` | Attribution retained from the research record |
| `submission_ids` | Related submission IDs |
| `files` | Copied artifacts, each with `path`, `role`, `origin` and `sha256` |
| `evidence` | Repository-relative files and optional source-line anchors supporting the claim |
| `terms`, `terms_unknown` | Internal terminology and explicitly unresolved terms |
| `tags` | Original candidate names and experiment identifiers for search |
| `links` | External references with `kind` and `target` |
| `slice` | Mining pass that supplied the node |
| `record_limitations` | Known evidence limitations, when applicable |

An experiment must have a recorded measurement; progress chatter does not qualify. Other nodes capture findings, consequential decisions, written lessons, diagnosed incidents and tools relied on by the work. `adopted` denotes a change recorded as entering a submitted implementation or the documented work process. Numeric measurements are retained from the evidence rather than recomputed during assembly.

Node titles do not use agent names, story labels or unexplained internal code names. Original names remain in tags and byte-preserved evidence. The [glossary](GLOSSARY.md) documents definitions and separates explicit source definitions from inferred interpretations.

## Edges

Each edge has `from`, `to` and `kind`. Endpoints are node IDs. Kinds are `enables`, `supersedes`, `rejects`, `measured_by` and `shipped_in`. Duplicate edges and self-links are omitted. Relationships are only included when recovered from the record; the graph is not a complete dependency model of every program.

## Submissions

A submission record includes `submission_id`, its official `date` and `status`, `score_public`, network and failure counts, recorded utilization and timing values, and the corresponding `node`. Missing measurements remain null or absent. Public scores retain the archived precision; they are not private re-evaluation results.

`source_path` is always `submissions/<id>/estimator.py`. `source_sha256` identifies the exact located submitted file. Where recovered, `archive_path` and `archive_sha256` identify its original archive. Validation checks that the archive contains those estimator bytes. Research candidates that differ from the fired source are labeled `code`, not substituted for the canonical source.

`best_so_far` and `best_at_time` track the best clean recorded public evaluation. A submission with any failed networks does not enter this comparison even when its overall status is `GRADED`.

## Files and provenance

[manifest.json](manifest.json) is the copy plan. Each copy has an original path, destination, SHA-256 digest, byte count, roles and referencing nodes. [PROVENANCE.json](../PROVENANCE.json) records the actual copies. Submitted estimator sources and experiment programs retain their original bytes. Publication-edited memory notes use `RELEASE:` origins; `sha256` and `bytes` identify the released artifact, while `source_sha256` and `source_bytes` identify its original source. `publication_edit` distinguishes these edited notes from byte-preserved copies.

File roles are `source`, `code`, `result`, `log`, `record`, `config`, `archive` and `explainer`. Shared evidence can be referenced by many nodes. Every submission retains its own canonical source path even when two submissions used identical source bytes. The original designated release package retains its relative layout under `release/`.

Five August measurements were found only inside the original graph, which also contains unrelated Phase 1 work. Their exact evidence objects were selected into `graph/source_excerpts/`. [derived_sources.json](derived_sources.json) records the excerpt hashes and the original graph's hash and source anchors. These generated excerpts are explicitly distinct from byte-for-byte file copies. The affected nodes state the missing separate-result-artifact limitation.

Generated documentation, graph data, viewer and build tools are separate from copied evidence. Tensor files, virtual environments and oversized data artifacts are excluded; see the [data index](../data/DATA_INDEX.md).

## Layout and links

- `submissions/<id>/`: submitted source, available archive, associated records and generated README.
- `estimator/`: the two designated sources and their source-linked anatomy.
- `experiments/<component>/<slug>/`: actual programs under `code/`, recovered relative dependency layout, and a concise generated `README.md`. Result-only entries and other research records live under `records/research/`.
- `instruments/<slug>/`: tool programs under `code/` and a generated README.
- `records/`: original logs, ledger, rules and result records.
- `release/`: the original designated-submission release package.
- `graph/`: graph data, glossary, viewer, source excerpts and validator.
- `tools/`: release assembly and documentation scripts.

Paths are repository-relative. Evidence anchors use `#L<line>` or `#L<start>-L<end>` in byte-preserved source files. The HTML viewer defaults to GitHub links with line anchors; its file-link selector switches to local paths. `meta.github` can pin another repository or commit. Local text viewers may not implement GitHub-style line fragments. The local page needs no network to display or search the embedded records.

## Validation

Run `python3 -I graph/validate.py --strict --checksums`. It checks schema vocabulary, title rules, IDs and edge endpoints, file and archive hashes, copy provenance, evidence line bounds, generated excerpt provenance, unreferenced artifacts and the full release checksum set. It does not rerun estimators or re-evaluate scientific claims.

Assembly required the frozen source tree and the private `_build/` mining inputs. Those inputs are not distributed in this Git release. The checked-in graph, provenance, source artifacts and validator support inspection without that build environment.

## Interactive dependency map

The viewer includes the original Phase 2 dependency tree: 642 `P2R0` nodes and 791 recorded edges, last updated on 2026-08-31. `dependency_source.json` preserves the exact parsed source objects, while `dependency_records/` provides direct links to individual records. `dependency_map.json` records display parents separately from scientific edges. Display parents organize the tree; they do not create research claims. The later estimator development is joined to the original map through source-backed connections between specific implementations.

`code_root` identifies the original source directory whose relative layout is preserved in a program bundle. `code_status` states whether programs were recovered. `code_limitations`, when present, distinguishes supporting code, shared drivers and unrecovered exact run scripts. The [code inventory](code_inventory.json) lists availability for every measured experiment. Historical runs may also require training data or runtime dependencies documented in the reproduction guide and data index.

The dark, top-to-bottom view opens with all 676 nodes visible: 642 research questions and 34 development and final-build records. It retains the original 210 × 80 node boxes and closed-subtree controls, but packs subtrees into a narrower arrangement. Original node coordinates are not preserved. Parents remain above their children; layout distances are neither quantitative nor chronological. Scroll or pinch to zoom and drag to pan. Explicitly collapsing or expanding a subtree keeps its control anchored.

Selecting a node keeps the surrounding map visible. From the overview, the camera zooms into its neighborhood; when already zoomed in, a click leaves the camera in place. A pulsing outline identifies the selection, with a static outline when reduced motion is enabled. The minimap shows the current viewport and supports navigation. **Back**, browser Back/Forward, **Parent**, breadcrumbs and **Recenter** preserve orientation. Back restores the previous camera and selection. **Whole graph** restores all branches and clears filters.

Search covers the original map and all 3,144 dated research records, including experiments and submissions. It supports Arrow Up/Down and Enter. Selecting a dated record opens its evidence alongside the map. Narrow screens use a dismissible details panel without replacing the graph with an isolated branch.

Borders and written statuses distinguish resolved questions (green), closed questions (red), open questions (blue), capacity-only results (ochre), and blocked questions (gray). The recorded result explains what closure means: a measured failure, an operational stop, or a completed check can have different implications. Clicking a color in the key highlights matching questions and their connecting routes; clicking it again clears that highlight. The exact source status definitions, origin and hash are preserved in [guide.json](guide.json). A resolved question is not automatically a deployed method. Some measured closures name resource failures in their individual notes, so the viewer uses the general label “Closed by measurement.”

**Start here** opens the problem statement and a clear entry into the analytic baseline. A six-approach index provides another entry point. Each approach shows its idea, recorded result and individual attempts; only three attempts appear initially, with a control to reveal all of them. Reading order is navigation, not an added scientific dependency. The six groups cover all 638 descendants, alongside the root and three setup questions.

Every one of the 642 research nodes has an edited title and explanation; all 641 nonroot nodes have a concise recorded result. Raw notes, supporting evidence and terminology remain under **Technical details**. [guide.json](guide.json) contains this reading guide and source-linked definitions. Definitions retain the glossary’s explicit uncertainty for inferred terminology. These reading aids do not change source objects, statuses or scientific relationships.

The full 791-edge scientific source record remains unchanged. The viewer draws recorded research relationships together with separately documented development connections. Seven display parents are adjusted to follow concrete estimator revisions; their evidence is recorded in the development data. This is a display hierarchy, not a rewrite of the original source objects. Primary node links and **Open work folder** lead to existing repository directories using GitHub `tree` URLs, or relative directories in local mode. [work_index.json](work_index.json) inventories the directly linked folders and their available programs; **All folders** exposes every associated folder. The 21 research nodes without directly linked work files explicitly say so, and their node links return to the interactive explanation. Exact source files remain linked under technical details. Ctrl/⌘-click opens the node’s work folder in a new tab. D3 7.9.0 is [bundled with its license](vendor/README.md) and embedded in the HTML for offline use.

**Estimator development** highlights a continuous route through the same graph, starting with the analytic baseline. A reader follows 35 shared milestones before the final two variants split into parallel branches. The route includes the third- and fourth-moment estimator, the learned output correction, its native refit, eigensolve removal and faster cubic products; it then continues through correction inside the recursion, the rebuilt signed-pair state, learned birth directions, shared-basis compression, covariance corrections and later readout changes. Unsuccessful versions remain visible beside the continuing work. Selection retains all surrounding nodes and their positions.

[release_path.json](release_path.json) supplies the development records, the seven display-parent overrides, and evidence for every added connection. The original research objects and all their scientific edges are preserved. Each development node links to its existing experiment or submission directory. The September core replacement is explicitly labeled **Core rebuilt**: it was an independent implementation, not a source-file child of the preceding estimator. Other milestone connections can summarize intermediate fitted or runtime versions; the evidence explains what was carried forward. The path is a documented development sequence, not a claim to be the globally shortest possible research route.

At the end, the #337506 first-hidden readout feeds the shared 961-column fit, followed by alternative strength-1.5 and strength-1.35 exports, runtime assembly, and the designated #337619 and #337641 sources. The two variants are drawn side by side. Their exact parent and export hashes are checked against archived manifests. The graph has no detached release cluster or dotted shared-method shortcuts. Solid tree routes are checked against every unrelated node box.

Run `python3 -I tools/build_work_index.py --check` to verify the folder inventory, `node tools/check_viewer.cjs` for source topology, compact-layout geometry, source targets and build-parent checks, and `python3 -I tools/check_documents.py` for generated documentation links. The optional DOM interaction check is:

```sh
npm install --prefix /tmp/arc-graph-dom jsdom@27.0.1 --no-audit --no-fund
NODE_PATH=/tmp/arc-graph-dom/node_modules node tools/check_viewer_dom.cjs
```

This executes the embedded HTML and checks continuous-map navigation, camera history, keyboard search, mobile states and link-mode changes. DOM and static SVG checks do not substitute for browser rendering.
