#!/usr/bin/env python3
"""Copy only manifest-listed evidence into this release repo and verify its bytes.

Usage: python3 -I tools/build_release.py [--dry-run | --checksums-only]
Preflight checks all sources and destinations before copying any file. Source
trees are read-only. Publication-edited notes retain their reviewed release bytes. Re-run --checksums-only after generating documentation.
"""
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/Users/matt/arc-white-box-estimation')
MEMORY = Path('/Users/matt/.claude/projects/-Users-matt-arc-white-box-estimation/memory')
LIMIT = 25 * 1024 * 1024
EXCLUDED = {'.npy', '.npz', '.bin', '.pt', '.pth', '.zip', '.h5', '.parquet', '.pyc'}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def checked_path(base, rel):
    p = (base / rel).resolve()
    if not p.is_relative_to(base.resolve()) or p == base.resolve():
        raise ValueError(f'path escapes {base}: {rel}')
    return p

def checksums():
    paths = [p for p in ROOT.rglob('*') if p.is_file() and not any(x in {'.git', '_build', '__pycache__'} for x in p.relative_to(ROOT).parts)
             and p.name != '.DS_Store' and p != ROOT / 'SHA256SUMS.txt']
    (ROOT / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in sorted(paths)))
    print(f'Checksums: {len(paths)} release files; build scratch excluded')

def main():
    if '--checksums-only' in sys.argv:
        checksums()
        return
    dry = '--dry-run' in sys.argv
    manifest = json.loads((ROOT / 'graph/manifest.json').read_text())
    prepared, errors, seen = [], [], {}
    source_hashes = {}
    for c in manifest['copies']:
        try:
            origin = c['origin']
            dst = checked_path(ROOT, c['dest'])
            if c.get('publication_edit'):
                if origin != 'RELEASE:' + c['dest']:
                    raise ValueError('invalid publication-edited origin: ' + origin)
                # Keep the reviewed release text; do not restore its archival source.
                src = dst
            else:
                src = checked_path(MEMORY, origin[7:]) if origin.startswith('MEMORY:') else checked_path(FROZEN, origin)
            if dst.relative_to(ROOT).parts[0] in {'.git', '_build', 'tools', 'graph'}:
                raise ValueError('evidence destination in reserved directory: ' + c['dest'])
            if not src.is_file():
                raise ValueError('missing source: ' + origin)
            if src.suffix.lower() in EXCLUDED or src.stat().st_size > LIMIT:
                raise ValueError('excluded type or size: ' + origin)
            if c['dest'] in seen and seen[c['dest']] != origin:
                raise ValueError('destination collision: ' + c['dest'])
            seen[c['dest']] = origin
            if src not in source_hashes: source_hashes[src] = sha256(src)
            digest = source_hashes[src]
            if c.get('sha256') and digest != c['sha256']:
                raise ValueError('source changed since assembly: ' + origin)
            prepared.append((c, src, dst, digest))
        except (ValueError, OSError) as e:
            errors.append(str(e))
    if errors:
        for e in errors:
            print('ERROR', e)
        raise SystemExit(1)
    total = sum(src.stat().st_size for _, src, _, _ in prepared)
    if dry:
        print(f'DRY RUN: {len(prepared)} files, {total / 1048576:.1f} MiB; preflight clean')
        return
    old = json.loads((ROOT / 'PROVENANCE.json').read_text()) if (ROOT / 'PROVENANCE.json').exists() else {'files': {}}
    provenance = dict(frozen_repo=str(FROZEN), built_at=stamp(), files={})
    copied = unchanged = 0
    for c, src, dst, digest in prepared:
        if dst.exists() and sha256(dst) == digest:
            unchanged += 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if sha256(dst) != digest:
                raise RuntimeError('copy hash mismatch: ' + c['dest'])
            copied += 1
        previous = old['files'].get(c['dest'], {})
        provenance['files'][c['dest']] = dict(origin=c['origin'], sha256=digest, bytes=src.stat().st_size,
            origin_mtime=datetime.datetime.fromtimestamp(src.stat().st_mtime, datetime.timezone.utc).isoformat(),
            copied_at=previous.get('copied_at', stamp()), roles=sorted(set(c['roles'])), nodes=sorted(set(c['nodes'])))
        if c.get('publication_edit'):
            for key in ('publication_edit', 'source_sha256', 'source_bytes'):
                provenance['files'][c['dest']][key] = c[key]
    (ROOT / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=1, sort_keys=True) + '\n')
    checksums()
    print(f'Copied {copied} files; {unchanged} byte-identical files retained; {total / 1048576:.1f} MiB total')

if __name__ == '__main__':
    main()
