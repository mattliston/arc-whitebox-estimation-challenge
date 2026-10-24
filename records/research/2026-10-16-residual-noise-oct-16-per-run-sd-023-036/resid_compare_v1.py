"""[opus-pass-release] first/warm-network residual and request comparison of two parity-release boundaries (normal + slow rows):
requests, free trips, residual seconds, residual instructions per request; a platform estimate = the reference fire's platform residual
(#336412 S2tr: first-network and warm medians from its eval, by participant_setup_wall_s) x the local residual ratio.
usage: resid_compare_v1.py <boundary_dir_new> <boundary_dir_ref> <eval_ref.json>"""
import json, statistics as st, sys
from pathlib import Path
B1, B0, EV = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = lambda d, m: {json.loads(l)['rank']: json.loads(l) for l in (d/m/'rows.jsonl').read_text().splitlines()}
e = json.load(open(EV))['evaluation']['results']['per_mlp']
first = [m['telemetry']['residual_wall_time_s'] for m in e if (m.get('telemetry') or {}).get('participant_setup_wall_s') is not None and m['telemetry'].get('residual_wall_time_s')]
warm = [m['telemetry']['residual_wall_time_s'] for m in e if (m.get('telemetry') or {}).get('participant_setup_wall_s') is None and (m.get('telemetry') or {}).get('residual_wall_time_s')]
print('reference platform residual: first nets n %d median %.3f max %.3f | warm n %d median %.3f p90 %.3f max %.3f' % (
    len(first), st.median(first), max(first), len(warm), st.median(warm), sorted(warm)[int(.9 * len(warm))], max(warm)))
for m in ('normal', 'slow'):
    a, b = rows(B1, m), rows(B0, m)
    for k in sorted(a):
        x, y = a[k], b[k]
        rr = x['residual_s'] / y['residual_s']; ri = (x['residual_instructions'] / x['requests']) / (y['residual_instructions'] / y['requests'])
        base = st.median(first) if k == 35 else st.median(warm)
        print(f"{m:6s} rank {k:3d} requests {x['requests']:6d} vs {y['requests']:6d} ({x['requests'] - y['requests']:+d}) free trips {x['free_trips']:5d} vs {y['free_trips']:5d} | "
              f"local residual {x['residual_s']:.3f} vs {y['residual_s']:.3f} s (x{rr:.3f}), instr/req x{ri:.3f} | platform est {base * x['requests'] / y['requests'] * ri:.3f} s (ref median {base:.3f})")
