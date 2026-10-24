#!/usr/bin/env python3
"""Validate graph schema, copy provenance, evidence, links and release checksums.

Usage: python3 -I graph/validate.py [--strict] [--checksums]
--strict checks file bytes, all graph/provenance links, evidence line bounds,
canonical submitted sources and unreferenced artifacts.
"""
import datetime
import functools
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEMPT_DIRS = {'.git', '_build', 'graph', 'tools'}
EXEMPT_FILES = {'.gitignore', 'README.md', 'LICENSE', 'PROVENANCE.json', 'SHA256SUMS.txt', 'WRITEUP.md', 'WRITEUP.tex', 'WRITEUP.pdf'}
GENERATED = {'NODE.md', 'README.md', 'SUBMISSIONS.md', 'ANATOMY.md', 'DATA_INDEX.md', 'INDEX.md', 'REPRODUCE.md'}
KINDS = {'submission', 'experiment', 'decision', 'finding', 'lesson', 'incident', 'tool'}
OUTCOMES = {'adopted', 'rejected', 'inconclusive', 'failed', 'informational'}
HEDGES = {'derived', 'measured', 'conjectured'}
ROLES = {'source', 'code', 'result', 'log', 'record', 'config', 'archive', 'explainer'}
EDGE_KINDS = {'enables', 'supersedes', 'rejects', 'measured_by', 'shipped_in'}
STORY = re.compile(r'\b(prehistory|journey|ladder|opening week|last day|after the bell|re-examination|codex banks)\b', re.I)

@functools.lru_cache(maxsize=None)
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

@functools.lru_cache(maxsize=None)
def line_count(path):
    with open(path, 'rb') as f:
        return sum(1 for _ in f)

def main():
    strict = '--strict' in sys.argv
    g = json.loads((ROOT / 'graph/graph.json').read_text())
    prov = json.loads((ROOT / 'PROVENANCE.json').read_text())['files'] if (ROOT / 'PROVENANCE.json').exists() else {}
    man = json.loads((ROOT / 'graph/manifest.json').read_text())['copies']
    errors, warnings = [], []
    ids, refs = {}, set()
    comps = {c['id'] for c in g['components']}

    def exists(rel, context):
        p = (ROOT / rel).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            errors.append(f'{context}: missing or unsafe path {rel}')
            return False
        refs.add(rel)
        return True

    for n in g['nodes']:
        nid = n.get('id', '?')
        for key in ('id', 'kind', 'title', 'date', 'component', 'outcome', 'hedge', 'claim'):
            if not n.get(key):
                errors.append(f'{nid}: missing {key}')
        for key, allowed in [('kind', KINDS), ('component', comps), ('outcome', OUTCOMES), ('hedge', HEDGES)]:
            if n.get(key) not in allowed:
                errors.append(f'{nid}: bad {key}: {n.get(key)}')
        if len(n['title']) > 120 or STORY.search(n['title']):
            errors.append(f'{nid}: noncompliant title: {n["title"]}')
        try:
            dt = datetime.datetime.fromisoformat(n['date'].replace('Z', '+00:00'))
            if dt.date() < datetime.date(2026, 8, 22):
                errors.append(f'{nid}: pre-Phase-2 date')
        except ValueError:
            errors.append(f'{nid}: invalid ISO date: {n["date"]}')
        if nid in ids:
            errors.append(f'duplicate node id: {nid}')
        ids[nid] = n
        for f in n.get('files', []):
            p = f['path']
            if not exists(p, nid):
                continue
            if f.get('role') not in ROLES:
                errors.append(f'{nid}: invalid file role {f.get("role")}')
            rec = prov.get(p)
            if not rec:
                errors.append(f'{nid}: no provenance for {p}')
            elif rec['origin'] != f.get('origin') or rec['sha256'] != f.get('sha256'):
                errors.append(f'{nid}: graph/provenance mismatch for {p}')
            elif strict and sha256((ROOT / p).resolve()) != rec['sha256']:
                errors.append(f'{nid}: changed bytes: {p}')
        for ref in n.get('evidence', []):
            p, _, fragment = ref.partition('#')
            if not exists(p, nid):
                continue
            if strict and fragment:
                m = re.fullmatch(r'L(\d+)(?:-L?(\d+))?', fragment)
                if m and not (1 <= int(m[1]) <= int(m[2] or m[1]) <= line_count(ROOT / p)):
                    errors.append(f'{nid}: evidence line outside file: {ref} ({line_count(ROOT / p)} lines)')
        if strict and not n.get('evidence'):
            errors.append(f'{nid}: no evidence citation')
    for e in g.get('edges', []):
        if e['from'] not in ids or e['to'] not in ids or e['kind'] not in EDGE_KINDS:
            errors.append(f'invalid edge: {e}')
    seen_sid = set()
    for s in g['submissions']:
        sid = s['submission_id']
        if sid in seen_sid:
            errors.append(f'duplicate submission #{sid}')
        seen_sid.add(sid)
        n = ids.get(s.get('node'))
        if not n or n['kind'] != 'submission' or n['submission_ids'] != [sid]:
            errors.append(f'#{sid}: wrong submission node')
        for key, digest in [('source_path', 'source_sha256'), ('archive_path', 'archive_sha256')]:
            if s.get(key) and exists(s[key], f'#{sid}') and strict and sha256(ROOT / s[key]) != s.get(digest):
                errors.append(f'#{sid}: {key} digest mismatch')
        if s.get('source_path') and s['source_path'] != f'submissions/{sid}/estimator.py':
            errors.append(f'#{sid}: noncanonical source path')
        if strict and s.get('archive_path') and (ROOT / s['archive_path']).is_file():
            try:
                with tarfile.open(ROOT / s['archive_path']) as archive:
                    matches = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name == 'estimator.py']
                    hashes = [hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in matches]
                if s['source_sha256'] not in hashes:
                    errors.append(f'#{sid}: archive does not contain the canonical estimator bytes')
            except (tarfile.TarError, OSError) as e:
                errors.append(f'#{sid}: unreadable archive: {e}')
        if not s.get('source_path'):
            warnings.append(f'#{sid}: submitted source not recovered')
    if {n['submission_ids'][0] for n in g['nodes'] if n['kind'] == 'submission'} != seen_sid:
        errors.append('submission nodes and submission index have different IDs')
    for nid in g.get('best_chain', []):
        if nid not in ids:
            errors.append(f'missing best-chain node {nid}')
    dep = json.loads((ROOT / 'graph/dependency_map.json').read_text())
    dep_ids = {n['id'] for n in dep['nodes']}
    for n in dep['nodes']:
        exists(n['source'].split('#')[0], 'dependency source')
        for f in n['files']:
            path = f['path'].split('#')[0]
            if exists(path, n['id']):
                rec = prov.get(path)
                if not rec or rec['origin'] != f['origin'] or (strict and sha256(ROOT/path) != rec['sha256']):
                    errors.append('dependency evidence provenance mismatch: '+path)
    for e in dep['edges']:
        if e['from'] not in dep_ids or e['to'] not in dep_ids: errors.append('invalid dependency edge')
    for c in man:
        rec = prov.get(c['dest'])
        if not rec or rec['origin'] != c['origin'] or rec['sha256'] != c.get('sha256'):
            errors.append(f'manifest/provenance mismatch: {c["dest"]}')
        if c['dest'] not in refs:
            errors.append(f'manifest copy not referenced: {c["dest"]}')
    if set(prov) != {c['dest'] for c in man}:
        errors.append('provenance and manifest list different files')
    derived = json.loads((ROOT / 'graph/derived_sources.json').read_text())
    for item in derived:
        path = item['path']
        if path not in refs:
            errors.append('derived evidence not referenced: ' + path)
        if exists(path, 'derived evidence') and strict:
            if sha256(ROOT / path) != item['sha256']:
                errors.append('derived evidence digest mismatch: ' + path)
            value = json.loads((ROOT / path).read_text())
            if value['source_sha256'] != item['source_sha256'] or value['source_origin'] != item['source_origin']:
                errors.append('derived evidence source metadata mismatch: ' + path)
    excerpt_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'graph/source_excerpts').glob('*.json')}
    if excerpt_paths != {d['path'] for d in derived}:
        errors.append('derived evidence index and excerpt files differ')
    orphans = []
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if rel.parts[0] in EXEMPT_DIRS or rel.as_posix() in EXEMPT_FILES or p.name in GENERATED or p.name == '.DS_Store':
            continue
        if rel.as_posix() not in refs:
            orphans.append(rel.as_posix())
    if strict:
        errors.extend('orphan file: ' + p for p in orphans)
    if '--checksums' in sys.argv:
        checked = set()
        for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
            digest, rel = line.split('  ', 1)
            checked.add(rel)
            if rel.startswith(('_build/', '.git/')) or not (ROOT / rel).is_file() or sha256(ROOT / rel) != digest:
                errors.append('checksum mismatch or excluded path: ' + rel)
        if not refs <= checked:
            errors.append('checksum list omits graph-referenced files')
        expected = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and not any(x in {'.git', '_build', '__pycache__'} for x in p.relative_to(ROOT).parts)
            and p.name != '.DS_Store' and p != ROOT / 'SHA256SUMS.txt'}
        if checked != expected:
            errors.append('checksum list and release file set differ')
    report = dict(nodes=len(ids), edges=len(g['edges']), submissions=len(seen_sid), referenced_files=len(refs), errors=errors, warnings=warnings, orphan_files=orphans)
    (ROOT / '_build').mkdir(exist_ok=True)
    (ROOT / '_build/validation_report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(f'nodes {len(ids)}; edges {len(g["edges"])}; submissions {len(seen_sid)}; referenced files {len(refs)}')
    print(f'errors {len(errors)}; warnings {len(warnings)}; orphan files {len(orphans)}')
    for e in errors[:40]:
        print('ERROR', e)
    for w in warnings[:20]:
        print('WARN', w)
    raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
