#!/usr/bin/env python3
"""Map graph nodes to existing work folders; never copy files or create folders.

python3 -I tools/build_work_index.py
python3 -I tools/build_work_index.py --check

Primary destinations are directories, not the graph's JSON source records.
Original research nodes are mapped only from their explicitly listed files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "graph/work_index.json"
INPUTS = ("graph/dependency_map.json", "graph/graph.json", "graph/release_path.json")
RELEASE_FOLDERS = {
    "build-parent": "submissions/337506",
    "build-fit": "release/common/readout",
    "build-export-337619": "release/337619/seal_s1.5",
    "build-runtime-337619": "release/337619/build",
    "build-final-337619": "submissions/337619",
    "build-export-337641": "release/337641/seal_s1.35",
    "build-runtime-337641": "release/337641/build",
    "build-final-337641": "submissions/337641",
}


def load(relative):
    return json.loads((ROOT / relative).read_text())


def relative_file(ref):
    relative = ref.partition("#")[0]
    path = ROOT / relative
    assert path.resolve().is_relative_to(ROOT), f"Unsafe source: {ref}"
    assert path.is_file(), f"Missing source: {ref}"
    return relative


def existing_directory(relative):
    path = ROOT / relative
    assert relative and path.resolve().is_relative_to(ROOT), f"Unsafe directory: {relative}"
    assert path.is_dir(), f"Missing directory: {relative}"
    assert not relative.endswith(".json"), f"Directory link points to JSON: {relative}"
    return path


def folder_for_file(relative, known):
    """Use the enclosing program bundle rather than its nested code/artifact folder."""
    parts = PurePosixPath(relative).parts
    for parent in PurePosixPath(relative).parents:
        if parent.as_posix() in known:
            return parent.as_posix()
    if parts[0] == "experiments" and len(parts) > 3:
        return "/".join(parts[:3])
    if parts[0] in {"submissions", "instruments"} and len(parts) > 2:
        return "/".join(parts[:2])
    if parts[:2] == ("records", "research") and len(parts) > 3:
        return "/".join(parts[:3])
    if parts[:2] == ("records", "dependencies"):
        # Historical workspaces are preserved beneath experiments/<family>/<workspace>.
        # Keep their artifact and submissions subdirectories under that same workspace.
        try:
            index = parts.index("experiments", 2)
        except ValueError:
            return PurePosixPath(relative).parent.as_posix()
        boundary = index + 3
        if len(parts) > boundary:
            return "/".join(parts[:boundary])
    return PurePosixPath(relative).parent.as_posix()


@lru_cache(maxsize=None)
def folder_info(relative):
    path = existing_directory(relative)
    total = code = 0
    for _, folders, files in os.walk(path, followlinks=False):
        folders[:] = [name for name in folders if name not in {".git", "__pycache__"}]
        for name in files:
            if name == ".DS_Store":
                continue
            total += 1
            code += name.endswith(".py")
    readme = path / "README.md"
    heading = None
    if readme.is_file():
        for line in readme.read_text(errors="replace").splitlines()[:20]:
            if line.startswith("# "):
                heading = line[2:].strip()
                break
    return {"total_files": total, "code_files": code,
            "readme": f"{relative}/README.md" if readme.is_file() else None,
            "heading": heading}


def kind(relative):
    if relative.startswith("records/research/"):
        return "research_record"
    if relative.startswith("records/dependencies/"):
        return "archived_work"
    return {"experiments": "experiment", "submissions": "submission",
            "instruments": "instrument", "release": "release"}.get(relative.split("/")[0], "documentation")


def make_entry(relative, references, basis, titles):
    info = folder_info(relative)
    title = titles.get(relative) or info["heading"]
    if not title:
        title = re.sub(r"[_-]+", " ", PurePosixPath(relative).name).strip()
        title = title[:1].upper() + title[1:]
    refs = set(references)
    return {"path": relative, "title": title, "files": len(refs),
            "linked_programs": sum(ref.endswith(".py") for ref in refs),
            "code_files": info["code_files"], "total_files": info["total_files"],
            "readme": info["readme"], "kind": kind(relative), "basis": basis}


def rank(entry):
    # Prefer a real program bundle with a README, then its direct citation strength.
    program_bundle = bool(entry["code_files"] and entry["readme"])
    return (-int(program_bundle), -int(bool(entry["linked_programs"])),
            -entry["files"], -int(bool(entry["code_files"])),
            -int(bool(entry["readme"])), entry["path"])


def file_groups(files, known):
    groups = defaultdict(set)
    for file in files:
        relative = relative_file(file["path"] if isinstance(file, dict) else file)
        groups[folder_for_file(relative, known)].add(relative)
    return groups


def original_entry(node, dated, known, titles):
    groups = file_groups(node["files"], known)
    folders = sorted((make_entry(path, refs, "direct_files", titles)
                      for path, refs in groups.items()), key=rank)
    if folders:
        return {"primary": folders[0]["path"], "directories": folders,
                "availability": "available" if any(f["code_files"] for f in folders) else "records_only",
                "note": "Folders containing this node’s directly cited files. Counts distinguish cited files from all files and Python programs in each folder."}
    # Related records are a weak association. Do not turn them into claimed source code.
    secondary = sorted({dated[related]["directory"] for related in node.get("related", []) if related in dated})
    return {"primary": None,
            "directories": [make_entry(path, [], "related_records", titles) for path in secondary],
            "availability": "unavailable",
            "note": "No work files are directly linked by the original research record. No implementation folder has been inferred."}


def build():
    dependency, graph, release = (load(relative) for relative in INPUTS)
    dated = {node["id"]: node for node in graph["nodes"]}
    titles = {node["directory"]: node["title"] for node in graph["nodes"]}
    titles.update({"release/common/readout": "Final output-correction fit and export",
                   "release/337619/seal_s1.5": "Estimator #337619 coefficient export",
                   "release/337641/seal_s1.35": "Estimator #337641 coefficient export",
                   "release/337619/build": "Estimator #337619 source assembly",
                   "release/337641/build": "Estimator #337641 source assembly",
                   "estimator": "Estimator instructions and mathematics"})
    known = set(titles) | set(RELEASE_FOLDERS.values()) | {node['directory'] for node in release['nodes'] if node.get('directory')}
    nodes = {node["id"]: original_entry(node, dated, known, titles) for node in dependency["nodes"]}
    for node in graph["nodes"]:
        primary = node["directory"]
        groups = file_groups(node["files"], known)
        groups[primary].add(relative_file(primary + "/" + node.get("document", "README.md")))
        folders = sorted((make_entry(path, refs, "dated_record", titles)
                          for path, refs in groups.items()), key=lambda entry: (entry["path"] != primary, rank(entry)))
        nodes[node["id"]] = {"primary": primary, "directories": folders,
                             "availability": "available" if any(f["code_files"] for f in folders) else "records_only"}
    for node in release["nodes"]:
        primary = node.get("directory") or RELEASE_FOLDERS[node["id"]]
        groups = file_groups([node["source"], *node["files"]], known)
        assert primary in groups, f"Release step lacks direct evidence in {primary}"
        folders = sorted((make_entry(path, refs, "release_step", titles)
                          for path, refs in groups.items()), key=lambda entry: (entry["path"] != primary, rank(entry)))
        nodes[node["id"]] = {"primary": primary, "directories": folders, "availability": "available"}
    for entry in nodes.values():
        paths = [folder["path"] for folder in entry["directories"]]
        assert len(paths) == len(set(paths)), "Duplicate folder"
        if entry["primary"] is not None:
            assert entry["primary"] in paths, "Primary folder absent from destinations"
        for folder in entry["directories"]:
            existing_directory(folder["path"])
            if folder["readme"]:
                relative_file(folder["readme"])
    research = [nodes[node["id"]] for node in dependency["nodes"]]
    assert len(nodes) == len(dependency["nodes"]) + len(graph["nodes"]) + len(release["nodes"])
    return {"meta": {"schema": 1, "sources": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in INPUTS},
                     "method": "Original research: existing folders of directly cited files; program bundles with READMEs preferred. Dated records: their declared directory. Release: the documented build-step directory. Related-record folders never become inferred source implementations.",
                     "file_counts": "files counts distinct node-cited files; code_files and total_files count the existing directory recursively.",
                     "counts": {"research_nodes": len(dependency["nodes"]), "dated_nodes": len(graph["nodes"]), "release_nodes": len(release["nodes"]),
                                "nodes": len(nodes), "directories": len({folder["path"] for entry in nodes.values() for folder in entry["directories"]}),
                                "research_with_work_folder": sum(entry["primary"] is not None for entry in research),
                                "research_without_direct_work_files": sum(entry["primary"] is None for entry in research),
                                "research_availability": dict(Counter(entry["availability"] for entry in research))}}, "nodes": nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the existing index against a fresh deterministic build")
    args = parser.parse_args()
    result = build()
    if args.check:
        assert json.loads(OUTPUT.read_text()) == result, "Work index is stale; rebuild it"
    else:
        OUTPUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(result["meta"]["counts"], indent=2))


if __name__ == "__main__":
    main()
