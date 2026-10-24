#!/usr/bin/env python3
"""Assemble the release graph and copy manifest without modifying source records.

Input: graph/spine.json, graph/components.json, _build/mine/*.jsonl.
Optional reviewed corrections: _build/overrides.json and each slice's
*_work/title_overrides.json. Raw mining files remain append-only.
"""
import collections
import datetime
import functools
import hashlib
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/Users/matt/arc-white-box-estimation')
MEMORY = Path('/Users/matt/.claude/projects/-Users-matt-arc-white-box-estimation/memory')
P2 = 'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
O = P2 + '/opus_reexam_20261001_v1'
KINDS = {'submission', 'experiment', 'decision', 'finding', 'lesson', 'incident', 'tool'}
OUTCOMES = {'adopted', 'rejected', 'inconclusive', 'failed', 'informational'}
HEDGES = {'derived', 'measured', 'conjectured'}
EDGE_KINDS = {'enables', 'supersedes', 'rejects', 'measured_by', 'shipped_in'}
EXCLUDED_EXT = {'.npy', '.npz', '.bin', '.pt', '.pth', '.h5', '.parquet', '.zip', '.pyc'}
LIMIT = 25 * 1024 * 1024
AICROWD = 'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{}'
RECORDS = {
    'vm_work/VM_PULL_MANIFEST.md': 'records/VM_PULL_MANIFEST.md',
    O + '/TEAM_LOG.md': 'records/TEAM_LOG_2026-10-01_to_10-22.md',
    P2 + '/fable_refit_v1/TEAM_LOG.md': 'records/TEAM_LOG_2026-09-22_to_09-26.md',
    'autoresearch/SUBMISSION_LEDGER.md': 'records/SUBMISSION_LEDGER.md',
    O + '/lead/runs/fable_oct21_final_v1/OFFICIAL_RULES_VERBATIM_PASTE_20261021.md': 'records/OFFICIAL_RULES_VERBATIM_PASTE_20261021.md',
    O + '/lead/runs/fable_oct21_final_v1/RULES_CONDENSED_RECORD_FROM_MEMORY_20260829.md': 'records/RULES_CONDENSED_RECORD_FROM_MEMORY_20260829.md',
    O + '/lead/runs/fable_oct21_final_v1/VM_SWEEP_RESULTS.md': 'records/VM_SWEEP_RESULTS_2026-10-21.md',
    O + '/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md': 'records/WRITEUP_DISCLOSURE_DRAFT_v1.md',
    O + '/SESSION_NOTES.md': 'records/SESSION_NOTES_2026-10-01_to_10-03.md',
}

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

@functools.lru_cache(maxsize=None)
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def origin_abs(origin):
    base, rel = (MEMORY, origin[7:]) if origin.startswith('MEMORY:') else (FROZEN, origin)
    p = (base / rel).resolve()
    if not p.is_relative_to(base.resolve()):
        raise ValueError('origin outside authorized source: ' + origin)
    return p

def slug(title, words=9):
    t = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode().lower()
    stop = {'the', 'a', 'an', 'of', 'at', 'on', 'in', 'to', 'and', 'for', 'with', 'vs', 'is', 'are', 'was', 'by', 'from', 'as', 'its'}
    parts = [w for w in re.sub(r'[^a-z0-9]+', ' ', t).split() if w not in stop][:words]
    while len('-'.join(parts)) > 80 and len(parts) > 1:
        parts.pop()
    return '-'.join(parts) or 'record'

def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def unique(items):
    result = []
    for x in items:
        if x not in result:
            result.append(x)
    return result

def main():
    report = collections.defaultdict(list)
    comps = load(ROOT / 'graph/components.json')
    comp_ids = {c['id'] for c in comps}
    spine = load(ROOT / 'graph/spine.json')['spine']
    by_sid = {s['submission_id']: s for s in spine}
    overrides = load(ROOT / '_build/overrides.json', {})
    titles = dict(overrides.get('titles', {}))
    aliases = {}
    node_fixes, file_additions, evidence_additions, duplicates = {}, {}, {}, {}
    code_limits, code_recovery, code_corrections = {}, {}, {}
    for p in sorted((ROOT / '_build/mine').glob('*_work/code_recovery.json')):
        for title, refs in load(p, {}).items(): code_recovery.setdefault(title, []).extend(refs)
    for p in sorted((ROOT / '_build/mine').glob('*_work/code_evidence_corrections.json')):
        code_corrections.update(load(p, {}))
    for p in sorted((ROOT / '_build/mine').glob('*_work/code_limitations.json')):
        code_limits.update(load(p, {}))
    for p in sorted((ROOT / '_build/mine').glob('*_work/evidence_additions.json')):
        for title, refs in load(p, {}).items():
            evidence_additions.setdefault(title, []).extend(refs)
    for p in sorted((ROOT / '_build/mine').glob('*_work/edge_title_aliases.json')):
        aliases.update(load(p, {}))
    for p in sorted((ROOT / '_build/mine').glob('*_work/duplicate_nodes.json')):
        for d in load(p, []):
            duplicates[d['drop_title']] = d['keep_title']
    for p in sorted((ROOT / '_build/mine').glob('*_work/node_overrides.json')):
        for title, patch in load(p, {}).items():
            node_fixes.setdefault(title, {}).update(patch)
    for p in sorted(list((ROOT / '_build/mine').glob('*_work/file_additions.json')) + list((ROOT / '_build/mine').glob('*_work/file_supplements.json'))):
        for title, refs in load(p, {}).items():
            file_additions.setdefault(title, []).extend(refs)
    for p in sorted((ROOT / '_build/mine').glob('*_work/title_overrides.json')):
        titles.update(load(p, {}))
    for p in sorted((ROOT / '_build/mine').glob('*.status.json')):
        aliases.update(load(p, {}).get('title_aliases', {}))
    raw = []
    for p in sorted((ROOT / '_build/mine').glob('*.jsonl')):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            source = f'{p.name}:{i}'
            try:
                n = json.loads(line)
            except ValueError as e:
                report['bad_json'].append(f'{source}: {e}')
                continue
            n['_src'] = source
            n.setdefault('slice', p.stem)
            n['_titles'] = [n.get('title', '')]
            n.update(node_fixes.get(n['title'], {}))
            n.update(overrides.get('nodes', {}).get(source, {}))
            n['files'] = unique(code_corrections.get(n['_titles'][0], {}).get('files', (n.get('files') or []) + file_additions.get(n['_titles'][0], [])) + code_recovery.get(n['_titles'][0], []))
            if n['_titles'][0] in code_limits: n['code_limitations'] = code_limits[n['_titles'][0]]
            if n.get('date', '').endswith('+00:00'):
                n['date'] = n['date'][:-6] + 'Z'
            n['evidence'] = unique(code_corrections.get(n['_titles'][0], {}).get('evidence', (n.get('evidence') or []) + evidence_additions.get(n['_titles'][0], [])))
            n['title'] = duplicates.get(n['title'], n['title'])
            n['title'] = titles.get(n['title'], n['title'])
            n['title'] = re.sub(r'\s+', ' ', n['title']).strip()
            for key, allowed in [('kind', KINDS), ('outcome', OUTCOMES), ('hedge', HEDGES), ('component', comp_ids)]:
                if n.get(key) not in allowed:
                    report['bad_' + key].append(f'{source}: {n.get(key)!r}')
            for key in ('date', 'title', 'claim'):
                if not n.get(key):
                    report['missing_field'].append(f'{source}: {key}')
            if len(n['title']) > 120:
                report['long_title'].append(f'{source}: {len(n["title"])} {n["title"]}')
            if not re.match(r'^2026-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z)?$', n.get('date', '')):
                report['bad_date'].append(f'{source}: {n.get("date")}')
            raw.append(n)
    # Merge duplicates while preserving every recorded file, evidence and alias.
    merged = {}
    for n in raw:
        sids = n.get('submission_ids') or []
        if n['kind'] == 'submission':
            if len(sids) != 1 or not isinstance(sids[0], int):
                report['submission_without_single_id'].append(n['_src'])
                continue
            key = ('submission', sids[0])
        else:
            key = (n['kind'], n['title'].lower(), n.get('date'))
        if key not in merged:
            merged[key] = n
        else:
            base = merged[key]
            for field in ('files', 'evidence', 'terms', 'terms_unknown', 'tags', 'edges', '_titles'):
                base[field] = unique((base.get(field) or []) + (n.get(field) or []))
            report['merged_duplicates'].append(f'{n["_src"]} -> {base["_src"]}')
    nodes = list(merged.values())
    have = {n['submission_ids'][0] for n in nodes if n['kind'] == 'submission'}
    for s in spine:
        sid = s['submission_id']
        if sid not in have:
            nodes.append(dict(kind='submission', title=f'Submission #{sid}: official evaluation record', date=s['date'], component='submission-process',
                outcome='failed' if s['status'] == 'FAILED' or s.get('failed_networks') else 'informational', hedge='measured',
                claim=f'Official evaluation record for submission #{sid}. A detailed research node has not yet been recovered.',
                measurement=f'public score {s["score_public"]}' if s.get('score_public') is not None else None,
                submission_ids=[sid], files=[], evidence=[], edges=[], tags=[], terms=[], slice='spine', _src='spine', _titles=[]))
            report['synthesised_submission'].append(str(sid))
    nodes.sort(key=lambda n: (n.get('date') or '', n['kind'] != 'submission', n['title']))
    supporting_only = {
        'Left-factor row-size input rejected: 0.16% bill for a proxy of already weak learned-input signals',
        'Fixed inner product fallback adds .1258107% FLOPs and leaves insufficient score margin',
        'Setup-bound super-views fail: strided tree operations add 8-59% CPU for little binding saving',
    }
    used, directories = set(), set()
    for n in nodes:
        if n['kind'] == 'submission':
            n['id'] = f'submission-{n["submission_ids"][0]}'
            n['directory'] = f'submissions/{n["submission_ids"][0]}'
        else:
            base = f'{n["date"][:10]}-{slug(n["title"])}'
            nid, k = base, 2
            while nid in used:
                nid, k = f'{base}-{k}', k + 1
            n['id'] = nid
            programs = [f['origin'] for f in n.get('files', []) if f.get('origin', '').endswith(('.py', '.sh'))]
            if programs and n['kind'] in {'experiment', 'tool'} and not any(t in supporting_only for t in n['_titles']):
                base = ('instruments/' if n['kind'] == 'tool' else f'experiments/{n["component"]}/') + slug(n['title'])
                n['code_root'] = os.path.commonpath([str(Path(os.path.normpath(p)).parent) for p in programs]) or '.'
                n['code_status'] = 'programs recovered; historical paths and data requirements retained'
            else:
                base = f'records/research/{n["id"]}'
                if n['kind'] == 'experiment':
                    n['code_status'] = 'result record; exact experiment program not identified (supporting code, if present, is linked separately)'
            n['document'] = 'README.md'
            d, k = base, 2
            while d in directories:
                d, k = f'{base}-{k}', k + 1
            n['directory'] = d
        used.add(n['id'])
        directories.add(n['directory'])
    by_node_sid = {n['submission_ids'][0]: n for n in nodes if n['kind'] == 'submission'}
    copies, canonical = {}, {}
    # Five Phase 2 measurements survive only in a mixed-phase research graph.
    # Generate scoped excerpts instead of copying unrelated Phase 1 records.
    excerpt_origin = 'pure_closure_writeup/problem_graph/graph.json'
    gaps = load(ROOT / '_build/mine/aug22-sep06_work/source_gaps.json', [])
    gap_by_title = {x.get('original_title', x.get('originaltitle')): x for x in gaps}
    derived = []
    if gaps:
        original_graph = load(FROZEN / excerpt_origin)
        original_nodes = {x['id']: x for x in original_graph['nodes']}
        for n in nodes:
            gap = next((gap_by_title[t] for t in n['_titles'] if t in gap_by_title), None)
            if not gap: continue
            source_node = original_nodes[gap['original_graph_node']]
            if gap['record'] not in source_node['evidence']:
                raise ValueError('Excerpt does not match frozen source: ' + n['id'])
            dest = 'graph/source_excerpts/' + n['id'] + '.json'
            value = dict(source_origin=excerpt_origin, source_sha256=sha256(FROZEN / excerpt_origin),
                source_node=gap['original_graph_node'], source_evidence=gap['evidence'],
                method='Exact Phase 2 evidence object selected from the frozen mixed-phase graph; no numerical recomputation.',
                record=gap['record'])
            path = ROOT / dest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
            n['_excerpt_path'] = dest
            n['record_limitations'] = 'Measurement retained in the historical graph; separate textual result artifact not recovered.'
            derived.append(dict(path=dest, sha256=sha256(path), source_origin=excerpt_origin, source_sha256=value['source_sha256'], source_evidence=gap['evidence']))
    (ROOT / 'graph/derived_sources.json').write_text(json.dumps(derived, indent=1) + '\n')

    def add_file(n, origin, role='record', dest=None, force=False):
        if origin == excerpt_origin:
            if n.get('_excerpt_path'): return n['_excerpt_path']
            raise ValueError('Mixed-phase graph requires an explicit Phase 2 excerpt: ' + n['id'])
        try:
            ab = origin_abs(origin)
        except (ValueError, TypeError) as e:
            report['bad_origin'].append(f'{n["id"]}: {e}')
            return None
        if not ab.is_file():
            report['missing_origin'].append(f'{n["id"]}: {origin}')
            return None
        if ab.suffix.lower() in EXCLUDED_EXT or '__pycache__' in ab.parts:
            report['excluded_tensor'].append(f'{n["id"]}: {origin}')
            return None
        if ab.stat().st_size > LIMIT:
            report['too_large'].append(f'{n["id"]}: {origin} ({ab.stat().st_size} bytes)')
            return None
        existing = next((f['path'] for f in n['files'] if f['origin'] == origin), None)
        if dest is None and existing:
            return existing
        if dest is None and n.get('code_root') and ab.suffix in {'.py', '.sh'}:
            dest = n['directory'] + '/code/' + Path(os.path.normpath(origin)).relative_to(n['code_root']).as_posix()
            force = True
        if dest is None:
            if origin.startswith('MEMORY:'):
                dest = 'records/memory/' + origin[7:]
            elif origin in RECORDS:
                dest = RECORDS[origin]
            elif origin.startswith('phase-two-release/'):
                dest = 'release/' + origin[len('phase-two-release/'):]
            elif origin.startswith('explainers/'):
                dest = 'estimator/research_notes/' + re.sub(r'^\d+_', '', Path(origin).name)
            else:
                dest = n['directory'] + '/' + ab.name
                if ab.name in ('NODE.md', 'README.md'):
                    dest = n['directory'] + '/source_' + ab.name
        dest = os.path.normpath(dest)
        if not force and origin in canonical:
            dest = canonical[origin]
        if dest in copies and copies[dest]['origin'] != origin:
            p = Path(dest)
            dest = str(p.with_name(p.stem + '-' + sha256(ab)[:8] + p.suffix))
            if dest in copies and copies[dest]['origin'] != origin:
                # Equal file bytes can still have distinct original locations.
                suffix = hashlib.sha256(origin.encode()).hexdigest()
                for width in range(8, 65, 8):
                    dest = str(p.with_name(p.stem + '-' + suffix[:width] + p.suffix))
                    if dest not in copies or copies[dest]['origin'] == origin:
                        break
                else:
                    raise ValueError('unresolved destination collision: ' + origin)
        canonical.setdefault(origin, dest)
        c = copies.setdefault(dest, dict(origin=origin, dest=dest, roles=[], nodes=[], sha256=sha256(ab), bytes=ab.stat().st_size))
        c['roles'] = unique(c['roles'] + [role])
        c['nodes'] = unique(c['nodes'] + [n['id']])
        f = dict(path=dest, role=role, origin=origin, sha256=c['sha256'])
        if f not in n['files']:
            n['files'].append(f)
        return dest

    # Register each submission's canonical artifacts before deduplicating general files.
    raw_files = {n['id']: n.pop('files', []) or [] for n in nodes}
    for n in nodes:
        n['files'] = []
    for sid, n in by_node_sid.items():
        s = by_sid.get(sid)
        if not s:
            report['submission_not_in_spine'].append(str(sid))
            continue
        sources = [f['origin'] for f in raw_files[n['id']] if f.get('role') == 'source']
        if not s.get('source_origin') and sources:
            s['source_origin'] = sources[0]
        origin = s.get('source_origin')
        if origin:
            for other in sources:
                if origin_abs(other).is_file() and sha256(origin_abs(other)) != sha256(origin_abs(origin)):
                    report['source_variant'].append(f'#{sid}: spine={origin}; node={other}')
            if s.get('date'):
                n['date'] = s['date']
            s['source_path'] = add_file(n, origin, 'source', f'submissions/{sid}/estimator.py', True)
            s['source_sha256'] = sha256(origin_abs(origin))
        if s.get('archive_origin'):
            s['archive_path'] = add_file(n, s['archive_origin'], 'archive', f'submissions/{sid}/submission.tar.gz', True)
            s['archive_sha256'] = sha256(origin_abs(s['archive_origin']))
        n['links'] = unique((n.get('links') or []) + [dict(kind='aicrowd', target=AICROWD.format(sid))])
    # Preserve the release package's layout and all of its relative links.
    rel_node = next((n for n in nodes if n['kind'] != 'submission' and 'release package' in n['title'].lower()), by_node_sid.get(337641))
    release_root = FROZEN / 'phase-two-release'
    if rel_node and release_root.is_dir():
        for p in sorted(release_root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name == '.DS_Store':
                continue
            role = 'archive' if p.name.endswith('.tar.gz') else ('code' if p.suffix == '.py' else 'record')
            release_dest = 'release/' + p.relative_to(release_root).as_posix()
            if release_dest == 'release/README.md': release_dest = 'release/ARCHIVE_README.md'
            add_file(rel_node, p.relative_to(FROZEN).as_posix(), role, release_dest, True)
    for sid in (337619, 337641):
        s, n = by_sid.get(sid), by_node_sid.get(sid)
        if s and n and s.get('source_origin'):
            add_file(n, s['source_origin'], 'source', f'estimator/submission_{sid}.py', True)
    for n in nodes:
        s = by_sid.get(n['submission_ids'][0]) if n['kind'] == 'submission' else None
        for f in raw_files[n['id']]:
            origin, role = f.get('origin'), f.get('role', 'record')
            if not origin:
                continue
            if s and role == 'source' and origin == s.get('source_origin'):
                continue
            if s and role == 'archive' and origin == s.get('archive_origin'):
                continue
            if s and role == 'source' and s.get('source_origin') and origin_abs(origin).is_file() and sha256(origin_abs(origin)) != s['source_sha256']:
                role = 'code'
            add_file(n, origin, role)
        evidence = []
        for ref in n.get('evidence') or []:
            origin, sep, line = ref.partition('#')
            if origin == 'TEAM_LOG.md':
                origin = O + '/TEAM_LOG.md'
            path = add_file(n, origin, 'record')
            if origin == excerpt_origin and n.get('_excerpt_path'):
                line, sep = 'L1', '#'
            if path:
                evidence.append(path + (sep + line if sep else ''))
            else:
                report['evidence_unresolved'].append(f'{n["id"]}: {ref}')
        n['evidence'] = unique(evidence)
    # Resolve edges through both original titles and reviewed replacement titles.
    scoped, all_titles = {}, {}
    for n in nodes:
        for title in n['_titles'] + [n['title']]:
            scoped[(n['slice'], title)] = n['id']
            all_titles.setdefault(title, n['id'])
    edges = set()
    for n in nodes:
        for e in n.pop('edges', []) or []:
            title = str(e.get('to') or '').strip()
            title = titles.get(aliases.get(title, title), aliases.get(title, title))
            m = re.match(r'^(?:Submission\s+)?#?(\d{6})(?:$|:)', title)
            tid = f'submission-{m[1]}' if m else scoped.get((n['slice'], title), all_titles.get(title))
            kind = e.get('kind')
            if kind not in EDGE_KINDS:
                report['bad_edge_kind'].append(f'{n["id"]}: {e}')
            elif tid not in used:
                report['edge_unresolved'].append(f'{n["id"]} -> {title!r}')
            elif tid != n['id']:
                edges.add((n['id'], tid, kind))
        if n['kind'] != 'submission' and n['outcome'] == 'adopted':
            for sid in n.get('submission_ids') or []:
                if f'submission-{sid}' in used:
                    edges.add((n['id'], f'submission-{sid}', 'shipped_in'))
        for key in ('_src', '_titles', 'component_other', '_excerpt_path'):
            n.pop(key, None)
    submissions = []
    for s in spine:
        row = {k: v for k, v in s.items() if k not in {'source_origin', 'archive_origin', 'bank_record', 'description', 'era'}}
        row['node'] = f'submission-{s["submission_id"]}'
        row.setdefault('source_path', None)
        row.setdefault('archive_path', None)
        submissions.append(row)
    nodes.sort(key=lambda n: (n.get('date') or '', n['kind'] != 'submission', n['title']))
    frozen_commit = subprocess.check_output(['git', '--no-optional-locks', '-C', str(FROZEN), 'rev-parse', 'HEAD'], text=True).strip()
    graph = dict(meta=dict(version=2, built_at=stamp(), frozen_repo=str(FROZEN), frozen_repo_commit=frozen_commit,
        objective='Phase 2, ARC White-Box Estimation Challenge 2026: final-layer error and metered computation on 1024-wide, 16-layer ReLU networks.'),
        components=comps, nodes=nodes, edges=[dict(zip(('from', 'to', 'kind'), x)) for x in sorted(edges)], submissions=submissions,
        best_chain=[s['node'] for s in submissions if s.get('best_so_far')])
    for name, value in [('graph/graph.json', graph), ('graph/manifest.json', dict(copies=list(copies.values()))), ('_build/assemble_report.json', dict(report))]:
        (ROOT / name).write_text(json.dumps(value, indent=1) + '\n')
    kinds = collections.Counter(n['kind'] for n in nodes)
    lines = [f'# Assemble report {graph["meta"]["built_at"]}', '', f'nodes {len(nodes)}, edges {len(edges)}, copies {len(copies)}, submissions {len(submissions)}', '',
             'kinds: ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common()), '']
    for key, values in sorted(report.items()):
        lines += [f'## {key} ({len(values)})', ''] + ['- ' + v for v in values] + ['']
    (ROOT / '_build/assemble_report.md').write_text('\n'.join(lines))
    print(lines[2])
    print(lines[4])
    for key, values in sorted(report.items()):
        print(f'{key}: {len(values)}')

if __name__ == '__main__':
    main()
