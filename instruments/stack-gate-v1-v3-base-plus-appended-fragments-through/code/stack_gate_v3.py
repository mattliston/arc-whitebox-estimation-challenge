"""[opus-stack] v3 = v2 on release_pipeline_v5b (parity replay v2: also for setups that bind the whole PB list), writing stack_v3/
(user-approved 2026-10-16). v2 = v1 with count_gates_v4 (hard instruction-weighted median residual <= .32 s; Python instructions per request).
clean base + fragment(s) -> fire-ready in one command (~7-10 min): builds base + fragments (appended LAST, in order,
blank-line separated, like build_lowres_v3), runs release_pipeline_v5b (static release: boundary normal+slow, official120, stage A,
metadata, captured-setup native parity, RELEASE_CONFIRMED, prefire), then count_gates_v4 (requests 42,700 / 41,200 normal+slow, worst
wall 106 s, setup 4.60 s, P(fail) on S2tr-family / all-Oct-16 / pre noise), the exact bill on the boundary nets vs the base's release
and vs S2tr (public mix 0.1 first + 0.9 warm), and the pre-registered projected public score vs 336412 (1.475902e-9) from --dmse (the
build's expected final-MSE change vs S2tr in %, base + fragment, e.g. -0.234 for the DC35 base alone). Writes stack_v3/STACK_<name>.json
(never overwrites) and prints the fire argv only when everything passes. Never fires.
usage: stack_gate_v3.py --base BASE.py --base-rel REL_DIR --frag F1[,F2..] --name NAME --dmse PCT [--desc '...'] [--ignore-gate-fail]"""
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
H = Path(__file__).resolve().parent; S = H/'stack_v3'; REL = O/'lead/runs/opus_release_v1'; PIPE = REL/'pipeline'
PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'; S2TR = REL/'rel_s2tr_20261016_opus_v1'; EV = O/'fire/s2tr_20261016_opus_v1/eval_336412.json'
BASE_SCORE = 1.475902e-9
sha = lambda b: hashlib.sha256(b).hexdigest()
utc = lambda: subprocess.check_output(['/bin/date', '-u', '+%FT%TZ'], text=True).strip()


def rows(d, m='normal'):
    return {json.loads(l)['rank']: json.loads(l) for l in (d/'boundary'/m/'rows.jsonl').read_text().splitlines()}


def bill(new, ref):
    a, b = rows(new), rows(ref)
    f, w = a[35]['flops'] / b[35]['flops'] - 1, a[100]['flops'] / b[100]['flops'] - 1
    return dict(first_pct=100 * f, warm_pct=100 * w, public_mix_pct=100 * (.1 * f + .9 * w))


ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True, type=Path); ap.add_argument('--base-rel', required=True, type=Path)
ap.add_argument('--frag', required=True); ap.add_argument('--name', required=True); ap.add_argument('--dmse', required=True, type=float)
ap.add_argument('--desc', default=''); ap.add_argument('--ignore-gate-fail', action='store_true')
a = ap.parse_args()
assert re.fullmatch(r'[a-z0-9]+', a.name), 'name: lowercase letters and digits only'
S.mkdir(exist_ok=True)
base = a.base.resolve().read_text(); frags = [Path(x).resolve() for x in a.frag.split(',') if x]
src = base
for p in frags:
    t = p.read_text(); src += ('' if t.startswith('\n') else '\n\n') + t
compile(src, a.name, 'exec')
cand = S/f'cand_{a.name}.py'
with cand.open('x') as f:
    f.write(src)
h = sha(src.encode()); tag = f"stk{a.name}_{utc()[:10].replace('-', '')}_opus_v1"
desc = a.desc or f"[opus-stack] {a.name}: clean base {sha(base.encode())[:8]} + {', '.join(p.name for p in frags)}"
print(f'{utc()} STACK {a.name} candidate {h[:8]} tag {tag}', flush=True)
r = subprocess.run([PY, '-B', 'release_pipeline_v5b.py', '--source', str(cand), '--tag', tag, '--desc', desc] + (['--ignore-gate-fail'] if a.ignore_gate_fail else []),
                   cwd=PIPE, capture_output=True, text=True, timeout=5400)
D = REL/f'rel_{tag}'
(S/f'pipeline_{a.name}.out').write_text(r.stdout + r.stderr)
done = (D/'PIPELINE.log').read_text() if (D/'PIPELINE.log').exists() else ''
confirmed = 'DONE status=CONFIRMED prefire_clear=True' in done
fire = next((l for l in r.stdout.splitlines() if l.startswith('FIRE (lead only')), None)
cg = subprocess.run(['python3', str(H/'count_gates_v4.py'), str(D), str(S2TR), str(EV)], capture_output=True, text=True) if (D/'boundary').exists() else None
cgj = next(iter(sorted(D.glob('COUNT_GATES_v4_*.json'))), None) if D.exists() else None
cgr = json.loads(cgj.read_text()) if cgj else {}
vs_base, vs_s2tr = (bill(D, a.base_rel.resolve()), bill(D, S2TR)) if (D/'boundary').exists() else (None, None)
proj = BASE_SCORE * (1 + a.dmse / 100) * (1 + vs_s2tr['public_mix_pct'] / 100) if vs_s2tr else None
passed = bool(confirmed and cgr.get('count_gates_v4_pass'))
fam = (cgr.get('p_fail') or {}).get('THIS BUILD', {})
res = dict(utc=utc(), name=a.name, tag=tag, candidate=str(cand), candidate_sha256=h, base=str(a.base.resolve()), base_sha256=sha(base.encode()),
           fragments={str(p): sha(p.read_bytes()) for p in frags}, pipeline_rc=r.returncode, confirmed=confirmed, count_gates_v4=cgr.get('count_gates_v4_pass'),
           requests={k: cgr.get(k) for k in ('normal_35_requests', 'normal_100_requests', 'slow_35_requests', 'slow_100_requests')},
           worst_wall_s=cgr.get('worst_wall_s'), residual_median_gate=cgr.get('residual_median_gate'), python_instr_per_request=cgr.get('python_instr_per_request'), setup_s=cgr.get('setup_s'), slow_setup_s=cgr.get('slow_setup_s'),
           p_fail_pct={g: dict(any=round(100 * v['either']['any'], 1), public=round(100 * v['either']['public'], 1)) for g, v in fam.items() if isinstance(v, dict) and 'either' in v},
           bill_vs_base=vs_base, bill_vs_s2tr=vs_s2tr, dmse_vs_s2tr_pct=a.dmse, projected_public=proj,
           projected_vs_336412_pct=(100 * (proj / BASE_SCORE - 1)) if proj else None, passed=passed, fire_argv_for_lead=fire if passed else None)
out = S/f'STACK_{a.name}.json'
with out.open('x') as f:
    json.dump(res, f, indent=1)
if cg is not None:
    print('\n'.join(l for l in cg.stdout.splitlines() if not l.startswith('P(fail) [check')))
print(f"{utc()} STACK {a.name} {'PASS' if passed else 'FAIL'}: confirmed={confirmed} count_gates_v4={cgr.get('count_gates_v4_pass')} | bill vs base "
      f"{vs_base and round(vs_base['public_mix_pct'], 3)}% vs S2tr {vs_s2tr and round(vs_s2tr['public_mix_pct'], 3)}% | dmse {a.dmse}% -> projected {proj and f'{proj:.4e}'} "
      f"({res['projected_vs_336412_pct'] and round(res['projected_vs_336412_pct'], 3)}% vs 336412) | wrote {out}")
if passed:
    print(fire)
sys.exit(0 if passed else 1)
