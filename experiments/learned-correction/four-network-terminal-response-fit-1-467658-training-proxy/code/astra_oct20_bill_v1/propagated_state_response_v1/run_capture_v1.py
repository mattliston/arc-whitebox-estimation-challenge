"""First-seven-network diagnostic, unchanged shared capacity-v7 guard."""
import argparse
import json
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PORT = HERE.parents[1] / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import sha, verified
p = argparse.ArgumentParser(add_help=False)
for key in ['recipe', 'root', 'plan']:
    p.add_argument('--'+key, type=Path, required=True)
p.add_argument('--candidate', required=True)
a, rest = p.parse_known_args()
assert '--warmup-each' not in rest and '--groups' not in rest
recipe, plan = json.loads(a.recipe.read_text()), json.loads(a.plan.read_text())
assert plan['groups'] == [[35,150,200],[250]]
assert set(plan['rows']) == {'35','150','200','250'}
assert all(set(row) == {'weights'} for row in plan['rows'].values())
assert plan['candidates'][a.candidate] == recipe['teacher']
for rec in recipe['harness'] + [recipe['hook']]:
    verified(a.root, rec)
os.environ['ARC_FULL_STATE_RECIPE'] = str(a.recipe.resolve())
os.environ['ARC_FULL_STATE_RECIPE_SHA'] = sha(a.recipe)
args = sys.argv[1:]; i = args.index('--recipe'); del args[i:i+2]
sys.argv = [sys.argv[0]] + args
BASE = PORT/'run_science_linux_v7.py'
assert sha(BASE) == '0c625815bb1753daa761395056453e38e5327d519fad86400d1d0acbab798e53'
source = BASE.read_text(); old = "str(here / 'run_group_linux_v7.py')"
assert source.count(old) == 1
source = source.replace(old, repr(str(HERE/'run_group_capture_v1.py')))
__file__ = str(BASE)
exec(compile(source, str(BASE)+':full_state_response_capture', 'exec'), globals())
