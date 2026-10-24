"""[opus-repair] the lead's CONFIDENCE BAR (2026-10-16 18:03Z) for a repaired iffy build, from MEASURED numbers only:
 (a) projection <= 1.4635e-9 with STORE-POOLED part values (--dmse, % vs S2tr) on the 6-net session bill (0.1 cold + 0.9 mean of 5 warm);
 (b) 6-net session: every net's instruction-weighted residual <= .325 (warm: .356 x residual-instruction ratio vs S2tr's same net;
     cold: .374 x ratio, .374 = #336412 first-net median) and the cold net <= .335;
 (c) P(any fail) <= 3% on S2tr-family noise (count_gates_v4 of the build's release);
 (d) worst wall <= 106 s (count_gates_v4, normal judge);
 (e) cold-net first-call shapes: reported from the refit simulation (lazy binds on the first network) - judged by hand.
usage: eval_bar_v1.py <release_dir> <6net_dir> <dmse_pct> [label]"""
import json, statistics as st, sys
from pathlib import Path
REL, MN, DMSE = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]); LABEL = sys.argv[4] if len(sys.argv) > 4 else REL.name
S2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_passrel_v1/lr12_multinet_v1/S2TR_merged')
rows = lambda d: {json.loads(l)['rank']: json.loads(l) for l in (d/'rows.jsonl').read_text().splitlines()}
b, s = rows(MN), rows(S2)
ranks = [r for r in (35, 100, 37, 39, 120, 160) if r in b]
fr = {k: b[k]['flops'] / s[k]['flops'] - 1 for k in ranks}
rr = {k: b[k]['residual_instructions'] / s[k]['residual_instructions'] for k in ranks}
req = {k: (b[k]['requests'], s[k]['requests']) for k in ranks}
proj_res = {k: (.374 if k == 35 else .356) * rr[k] for k in ranks}
warm = [k for k in ranks if k != 35]
mix = .1 * fr[35] + .9 * st.mean(fr[k] for k in warm) if 35 in fr and warm else float('nan')
proj = 1.475902e-9 * (1 + DMSE / 100) * (1 + mix)
cg = json.loads(next(iter(sorted(REL.glob('COUNT_GATES_v4_*.json')))).read_text())
fam = cg['p_fail']['THIS BUILD']['family']['either']; post = cg['p_fail']['THIS BUILD']['post']['either']
famr = cg['p_fail']['this build, residual = warm requests x 7.77 us']['family']['either']
res = {
 'a': (proj <= 1.4635e-9, f"projection {proj:.4e} (dmse {DMSE:+.3f}%, 6-net bill {100 * mix:+.3f}% vs S2tr; {len(ranks)} nets)"),
 'b': (all(proj_res[k] <= .325 for k in warm) and proj_res.get(35, 9) <= .335,
       'per-net ' + ', '.join(f"{k}:{proj_res[k]:.4f}" for k in ranks) + f" (warm max {max(proj_res[k] for k in warm):.4f} <= .325, cold {proj_res.get(35, float('nan')):.4f} <= .335)"),
 'c': (fam['any'] <= .03, f"P(any/public) family {100 * fam['any']:.1f}/{100 * fam['public']:.1f}% (request-based {100 * famr['any']:.1f}/{100 * famr['public']:.1f}%), Oct-16 {100 * post['any']:.1f}/{100 * post['public']:.1f}%"),
 'd': (cg['worst_wall_s'] <= 106, f"worst wall {cg['worst_wall_s']} s (normal judge)"),
}
print(f"== BAR {LABEL}")
for k, (ok, txt) in res.items():
    print(f"  ({k}) {'PASS' if ok else 'FAIL'}: {txt}")
print(f"  requests normal {cg['normal_35_requests']:,}/{cg['normal_100_requests']:,}, slow {cg['slow_35_requests']:,}/{cg['slow_100_requests']:,}; medians "
      f"{cg['residual_median_gate']['median']:.4f} instr-weighted / {cg['residual_projection'].get('warm_req_model', float('nan')) if isinstance(cg.get('residual_projection'), dict) else 'n/a'}; "
      f"instr/request first {cg['python_instr_per_request']['35']['build']:.0f} warm {cg['python_instr_per_request']['100']['build']:.0f}")
print('  6-net: ' + ', '.join(f"{k}: req {req[k][0]:,} ({req[k][0] - req[k][1]:+,}) flops {100 * fr[k]:+.3f}% instr x{rr[k]:.4f}" for k in ranks))
# the warm-request cut that would put the worst warm net at .325 (residual ~ requests x per-request client cost, u64's call-form law)
worst = max(warm, key=lambda k: proj_res[k])
need = 1 - .325 / proj_res[worst]
print(f"  (b) needs the worst warm net ({worst}) at {proj_res[worst]:.4f} -> .325: -{100 * need:.1f}% client instructions ~ -{need * b[worst]['requests']:,.0f} requests "
      f"at its {b[worst]['residual_instructions'] / b[worst]['requests']:,.0f} instr/request (two-strip rate 0.21% bill per 1,000 -> +{.21 * need * b[worst]['requests'] / 1000:.2f}% bill)")
