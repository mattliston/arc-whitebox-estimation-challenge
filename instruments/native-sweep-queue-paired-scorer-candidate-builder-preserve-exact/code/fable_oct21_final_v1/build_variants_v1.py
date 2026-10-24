"""Build dial variants = exact deployed science bytes + appended tail; write a plan with the variants added.
usage: python3 build_variants_v1.py <plan_out_name> <variants.json>
variants.json: {"<name>": ["<CONST> = <value>", ...], ...}  (tail lines; _FAST_RESCUE_FRAC = {} appended automatically)
"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
REL = 'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/'
DEPLOYED = RUNS / 'astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py'
BASE_PLAN = RUNS / 'astra_oct20_bill_v1/runtime_first128_frozen_census_v1/NATIVE_PLAN_LINUX_v1.json'
sha = lambda b: hashlib.sha256(b).hexdigest()
base = DEPLOYED.read_bytes()
assert sha(base) == '49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80' and len(base) == 2896123
plan = json.loads(BASE_PLAN.read_text())
plan_out, variants = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
for name, lines in variants.items():
    tail = '\n# fable 2026-10-21 variant %s\n%s\n_FAST_RESCUE_FRAC = {}\n' % (name, '\n'.join(lines))
    data = base + tail.encode()
    out = HERE / ('cand_%s.py' % name)
    if out.exists():
        assert out.read_bytes() == data, name
    else:
        out.write_bytes(data)
    plan['candidates'][name] = dict(path=REL + out.name, bytes=len(data), role='frozen_candidate', sha256=sha(data), effective_sha256=sha(data))
    print(name, len(data), sha(data))
text = json.dumps(plan, indent=2) + '\n'
(HERE / plan_out).write_text(text)
print('PLAN', plan_out, sha(text.encode()))
