#!/usr/bin/env python3
"""TAG opus-physicist T130 disk inventory (lead 15:2xZ: physicist/ is 50 GB; list, do not delete): per top-level entry of physicist/, the bytes in
generated-network weight caches (any */generated_cache_v1/weights_seed*.npy: regenerable from the seed, sha-checked against the truth on load), in result
files (results/*.json), in .npz dumps, and other; each inode counted once (hard links). Writes physicist/t130_disk_inventory.tsv; prints the top 40 and totals.
usage: t130_disk_inventory.py"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
seen, rows = set(), []
for top in sorted(HERE.iterdir()):
    acc = dict(cache=0, results=0, npz=0, other=0, ncache=0)
    paths = [top] if top.is_file() else (p for p in top.rglob('*'))
    for p in paths:
        try:
            if p.is_symlink() or not p.is_file():
                continue
            st = p.stat()
        except OSError:
            continue
        if st.st_ino in seen:
            continue
        seen.add(st.st_ino)
        s = str(p)
        if '/generated_cache_v1/' in s or '/generated_cache_shared_v1/' in s:
            acc['cache'] += st.st_size; acc['ncache'] += p.suffix == '.npy'
        elif '/results/' in s:
            acc['results'] += st.st_size
        elif p.suffix == '.npz':
            acc['npz'] += st.st_size
        else:
            acc['other'] += st.st_size
    tot = sum(acc[k] for k in ('cache', 'results', 'npz', 'other'))
    if tot:
        rows.append((tot, top.name, acc))
rows.sort(reverse=True)
G = 1024 ** 3
with (HERE / 't130_disk_inventory.tsv').open('w') as fh:
    fh.write('entry\ttotal_GiB\tweight_cache_GiB\tweight_files\tresults_GiB\tnpz_GiB\tother_GiB\n')
    for tot, name, a in rows:
        fh.write('%s\t%.3f\t%.3f\t%d\t%.3f\t%.3f\t%.3f\n' % (name, tot / G, a['cache'] / G, a['ncache'], a['results'] / G, a['npz'] / G, a['other'] / G))
T = {k: sum(a[k] for _, _, a in rows) for k in ('cache', 'results', 'npz', 'other', 'ncache')}
print('TOTAL %.1f GiB: weight caches %.1f GiB (%d files), results %.2f GiB, npz %.1f GiB, other %.1f GiB' % (
    sum(T[k] for k in ('cache', 'results', 'npz', 'other')) / G, T['cache'] / G, T['ncache'], T['results'] / G, T['npz'] / G, T['other'] / G))
for tot, name, a in rows[:40]:
    print('%-44s %6.2f GiB  cache %6.2f (%3d)  results %5.2f  npz %5.2f  other %5.2f' % (name, tot / G, a['cache'] / G, a['ncache'], a['results'] / G, a['npz'] / G, a['other'] / G))
