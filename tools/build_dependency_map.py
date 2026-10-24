#!/usr/bin/env python3
"""Extract the Phase 2 dependency tree from the read-only research graph.

Raw node objects are retained exactly as parsed, with derivation provenance.
Display parents organize the drawing; all original scoped edges remain available.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/graph.json')
PREFIX = 'pure_closure_writeup/problem_graph/'

def main():
    original = json.loads(SOURCE.read_text())
    raw = [n for n in original['nodes'] if n['id'].startswith('P2R0')]
    ids = {n['id'] for n in raw}
    edges = [e for e in original['edges'] if e['from'] in ids and e['to'] in ids]
    record = dict(source_origin=PREFIX+'graph.json', source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_updated=original['updated'], method='Exact parsed Phase 2 node objects and edges, selected by P2R0 ID prefix; Phase 1 excluded.',
        nodes=raw, edges=edges)
    text = json.dumps(record, indent=2, ensure_ascii=False)+'\n'
    (ROOT/'graph/dependency_source.json').write_text(text)
    anchors = {}
    for i, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r'\s+"id": "([^"]+)",?', line)
        if match and match[1] in ids:
            anchors[match[1]] = f'graph/dependency_source.json#L{i}'
    parents = {}
    for n in raw:
        for child in n.get('decomposition_children', []):
            if child in ids and child != n['id']:
                parents.setdefault(child, n['id'])
    for e in edges:
        if e['type'] in {'decomposes_to', 'decomposition'} and e['from'] != e['to']:
            parents.setdefault(e['to'], e['from'])
    for n in raw:
        if n.get('display_parent') in ids:
            parents[n['id']] = n['display_parent']
        if n['id'] != 'P2R0':
            parents.setdefault(n['id'], max((p for p in ids if p != n['id'] and n['id'].startswith(p)), key=len, default='P2R0'))
    parents.pop('P2R0', None)
    cycle_repairs = []
    for node in ids:
        seen, cur = set(), node
        while cur in parents:
            if cur in seen:
                cycle_repairs.append(cur)
                parents[cur] = 'P2R0'
                break
            seen.add(cur)
            cur = parents[cur]
    release = json.loads((ROOT/'graph/graph.json').read_text())
    manifest=json.loads((ROOT/'graph/manifest.json').read_text())
    paths = {c['origin']:c['dest'] for c in reversed(manifest['copies'])}
    copies = {c['dest']:c for c in manifest['copies']}
    owners = {}
    added = 0
    for n in release['nodes']:
        for f in n['files']:
            paths.setdefault(f['origin'], f['path'])
            owners.setdefault(f['origin'], set()).add(n['id'])
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for v in value: yield from strings(v)
        elif isinstance(value, dict):
            for v in value.values(): yield from strings(v)
    titles = {
        'P2R0E3A':'High-order accuracy ablations',
        'P2R0':'Mean estimation for width-1024, depth-16 networks',
        'P2R0A':'Phase 2 evaluation contract',
        'P2R0B':'Width and depth migration audit',
        'P2R0C':'Metered computation and estimator design',
        'P2R0D':'Gaussian, Hermite and Mehler approximations',
        'P2R0F':'Fitted corrections to moment propagation',
        'P2R0G':'Late-layer source reconstruction',
    }
    nodes, unresolved = [], 0
    source_dir=ROOT/'graph/dependency_records';source_dir.mkdir(exist_ok=True)
    used_names=set()
    for n in raw:
        files, related = {}, set()
        for value in strings(n.get('evidence', [])):
            candidates = [value] + re.findall(r'(?:pure_closure_writeup/|experiments/|autoresearch/|vm_work/)[^\s`\"<>;,()\[\]]+', value)
            for candidate in candidates:
                origin, sep, anchor = candidate.partition('#')
                if origin.startswith('experiments/'): origin = PREFIX+origin
                if not origin.startswith(('pure_closure_writeup/','autoresearch/','vm_work/','phase-two-release/','explainers/')) or any(len(part.encode()) > 255 for part in Path(origin).parts): continue
                source = SOURCE.parents[2] / origin
                if origin not in paths and source.is_file() and source.resolve().is_relative_to(SOURCE.parents[2]) and source.resolve()!=SOURCE and source.suffix.lower() not in {'.npy','.npz','.bin','.pt','.pth','.h5','.parquet','.zip','.pyc'} and source.stat().st_size<=25*1024*1024:
                    dest='records/dependencies/'+origin
                    paths[origin]=dest
                    copies[dest]=dict(origin=origin,dest=dest,roles=['code' if source.suffix=='.py' else 'record'],nodes=[],dependency_nodes=[],sha256=hashlib.sha256(source.read_bytes()).hexdigest(),bytes=source.stat().st_size)
                    added+=1
                if origin in paths:
                    copies[paths[origin]].setdefault('dependency_nodes',[])
                    if n['id'] not in copies[paths[origin]]['dependency_nodes']:copies[paths[origin]]['dependency_nodes'].append(n['id'])
                    path = paths[origin] + (sep+anchor if re.fullmatch(r'L\d+(?:-L?\d+)?', anchor) else '')
                    files[path] = {'path':path,'origin':origin}
                    related.update(owners.get(origin, []))
        title = titles.get(n['id'], n['label'])
        title = re.sub(r'(?i)\bladder\b', 'comparison', title)
        title = re.sub(r'^(?:SUB-\d+|N\d+)\s*[: ]\s*', '', title)
        if not files: unresolved += 1
        base=re.sub(r'[^a-z0-9]+','-',title.lower()).strip('-')[:95].rstrip('-') or 'dependency'
        name=base;k=2
        while name in used_names: name=base+'-'+str(k);k+=1
        used_names.add(name)
        entry='graph/dependency_records/'+name+'.json'
        (ROOT/entry).write_text(json.dumps(dict(source_origin=record['source_origin'],source_sha256=record['source_sha256'],source_updated=record['source_updated'],record=n),indent=2,ensure_ascii=False)+'\n')
        anchors[n['id']]=entry+'#L6'
        nodes.append(dict(id=n['id'], title=title, parent=parents.get(n['id']), status=n['status'],
            claim=n.get('summary', ''), criterion=n.get('terminal_criterion', ''), source=anchors[n['id']],
            files=list(files.values()), related=sorted(related), original_label=n['label']))
    result = dict(meta={k:record[k] for k in ('source_origin','source_sha256','source_updated','method')},
        root='P2R0', nodes=nodes, edges=edges,
        audit=dict(nodes=len(nodes),edges=len(edges),nodes_without_copied_evidence=unresolved,layout_cycle_repairs=cycle_repairs,additional_evidence_files=added))
    (ROOT/'graph/dependency_map.json').write_text(json.dumps(result, indent=1, ensure_ascii=False)+'\n')
    manifest['copies']=list(copies.values())
    (ROOT/'graph/manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
    print(json.dumps(result['audit']))

if __name__ == '__main__': main()
