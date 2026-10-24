#!/usr/bin/env python3
"""Build the self-contained graph/index.html. Score figure: tools/plot_score.py.

Usage: python3 -I tools/build_index.py [--github https://github.com/<owner>/<repo>/blob/<commit>/]
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    g = json.load(open(os.path.join(ROOT, "graph", "graph.json")))
    if "--github" in sys.argv: g["meta"]["github"] = sys.argv[sys.argv.index("--github") + 1]
    slim = {"meta": g["meta"], "components": g["components"], "edges": g["edges"], "submissions": g["submissions"], "best_chain": g.get("best_chain", []),
            "nodes": [{k: v for k, v in n.items() if k in ("id", "kind", "title", "date", "component", "outcome", "hedge", "claim", "measurement", "actor", "submission_ids", "files", "evidence", "terms", "tags", "links", "directory", "document", "code_status", "code_limitations")} for n in g["nodes"]]}
    for n in slim["nodes"]: n["files"] = [{"path": f["path"], "role": f["role"]} for f in n.get("files", [])]
    tpl = open(os.path.join(ROOT, "graph", "index.template.html")).read()
    data = json.dumps(slim, separators=(",", ":")).replace("</", "<\\/")
    deps = json.load(open(os.path.join(ROOT, "graph", "dependency_map.json")))
    dependency_data = json.dumps(deps, separators=(",", ":")).replace("</", "<\\/")
    guide_data = json.dumps(json.load(open(os.path.join(ROOT, "graph", "guide.json"))), separators=(",", ":")).replace("</", "<\\/")
    release_data = json.dumps(json.load(open(os.path.join(ROOT, "graph", "release_path.json"))), separators=(",", ":")).replace("</", "<\\/")
    work_data = json.dumps(json.load(open(os.path.join(ROOT, "graph", "work_index.json"))), separators=(",", ":")).replace("</", "<\\/")
    script = open(os.path.join(ROOT, "graph", "viewer.js")).read()
    d3 = open(os.path.join(ROOT, "graph", "vendor", "d3.min.js")).read().replace("</", "<\\/")
    html = tpl.replace("/*__GRAPH_JSON__*/null", data).replace("/*__DEPENDENCY_JSON__*/null", dependency_data).replace("/*__GUIDE_JSON__*/null", guide_data).replace("/*__RELEASE_PATH_JSON__*/null", release_data).replace("/*__WORK_INDEX_JSON__*/null", work_data).replace("/*__D3_JS__*/", d3).replace("/*__VIEWER_JS__*/", script)
    open(os.path.join(ROOT, "graph", "index.html"), "w").write(html)
    print(f"index.html {len(html)/1e6:.2f} MB, nodes {len(slim['nodes'])}; score figure maintained by plot_score.py")

if __name__ == "__main__":
    main()
