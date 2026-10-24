"""[opus-diet2b] diet vs base on a --sites census: requests (total and by function / line), Python instructions per request, flops,
prediction bytes (pred_<rank>.bin) per rank. usage: compare_sites_v1.py <base_sites_dir> <diet_sites_dir>"""
import collections, json, sys
from pathlib import Path
A, B = Path(sys.argv[1]), Path(sys.argv[2])
ra = {json.loads(l)['rank']: json.loads(l) for l in (A/'rows.jsonl').read_text().splitlines()}
rb = {json.loads(l)['rank']: json.loads(l) for l in (B/'rows.jsonl').read_text().splitlines()}
for k in sorted(ra):
    x, y = ra[k], rb[k]
    same = (A/f'pred_{k}.bin').read_bytes() == (B/f'pred_{k}.bin').read_bytes()
    print(f"rank {k}: requests {x['requests']} -> {y['requests']} ({y['requests'] - x['requests']:+d}); flops {x['flops']} -> {y['flops']} ({y['flops'] - x['flops']:+d}); "
          f"instr/req {x['residual_instructions'] / x['requests']:,.0f} -> {y['residual_instructions'] / y['requests']:,.0f} "
          f"(x{(y['residual_instructions'] / y['requests']) / (x['residual_instructions'] / x['requests']):.4f}); residual instr {100 * (y['residual_instructions'] / x['residual_instructions'] - 1):+.2f}%; "
          f"predictions {'BYTE-IDENTICAL' if same else 'DIFFER'}")
    fa, fb = collections.Counter(), collections.Counter()
    for key, n in x['by_site'].items(): fa[key.split(' ')[1] + ' ' + key.split(' ', 2)[2]] += n
    for key, n in y['by_site'].items(): fb[key.split(' ')[1] + ' ' + key.split(' ', 2)[2]] += n
    d = {f: fb[f] - fa[f] for f in set(fa) | set(fb) if fb[f] != fa[f]}
    for f, v in sorted(d.items(), key=lambda t: t[1])[:30]:
        print(f"    {v:+5d} {f}")
