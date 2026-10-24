"""[opus-stack] 8-net worker-like session comparison (rank 35 cold, 100 second, then 37,39,120,140,160,180 warm) of a build vs S2tr:
per-net flops ratio, requests, residual instructions (Python) ratio; public-mix bill = 0.1 cold + 0.9 mean(warm 7); projected platform
residual per net = #336412's platform median (.356 warm, cold nets x first/warm request ratio) x the net's residual-instruction ratio.
usage: compare8_v1.py <build_dir> <s2tr_dir>"""
import json, statistics as st, sys
from pathlib import Path
B, S = Path(sys.argv[1]), Path(sys.argv[2])
rows = lambda d: {json.loads(l)['rank']: json.loads(l) for l in (d/'rows.jsonl').read_text().splitlines()}
b, s = rows(B), rows(S)
ranks = [35, 100, 37, 39, 120, 140, 160, 180]
print(f"{'rank':>4s} {'flops ratio':>11s} {'req b':>6s} {'req s2tr':>8s} {'resid instr b':>13s} {'ratio':>7s} {'instr/req b':>11s}")
fr, rr = {}, {}
for k in ranks:
    x, y = b[k], s[k]
    fr[k] = x['flops'] / y['flops'] - 1; rr[k] = x['residual_instructions'] / y['residual_instructions']
    print(f"{k:4d} {100 * fr[k]:+10.3f}% {x['requests']:6d} {y['requests']:8d} {x['residual_instructions'] / 1e9:12.3f}G {rr[k]:7.4f} {x['residual_instructions'] / x['requests']:11.0f}")
warm = [fr[k] for k in ranks[1:]]
mix = .1 * fr[35] + .9 * st.mean(warm)
print(f"public-mix bill vs S2tr (0.1 cold + 0.9 mean of 7 warm): {100 * mix:+.3f}%  (cold {100 * fr[35]:+.3f}%, second {100 * fr[100]:+.3f}%, later-6 mean {100 * st.mean(warm[1:]):+.3f}%)")
print(f"residual-instruction ratio vs S2tr: cold {rr[35]:.4f}, second {rr[100]:.4f}, warm median {st.median([rr[k] for k in ranks[1:]]):.4f}, max {max(rr.values()):.4f}")
print(f"projected platform residual (x #336412 .356 warm median): warm median {.356 * st.median([rr[k] for k in ranks[1:]]):.4f} s, max warm {.356 * max(rr[k] for k in ranks[1:]):.4f} s; cold {.374 * rr[35]:.4f} s (.374 = #336412 first-net median)")
