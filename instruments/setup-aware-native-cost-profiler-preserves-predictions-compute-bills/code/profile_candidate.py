"""Local observation only: keep every original budget charge, attribute by stack.

Usage: profile_candidate.py CANDIDATE RANKS TAG
Runs the unchanged setup-aware evaluator. Profiling overhead is not runtime
evidence. Source code and output predictions are untouched.
"""
from collections import Counter
from pathlib import Path
import json
import runpy
import sys

from flopscope import _budget

HERE = Path(__file__).resolve().parent
candidate = str(Path(sys.argv[1]).resolve())
tag = sys.argv[3]
profile_path = HERE / (tag + '_profile.json')
assert not profile_path.exists()
charges = Counter()
requests = Counter()
original_charge = _budget.BudgetContext._charge_op


def charged(self, op_name, flop_cost, subscripts, shapes, **kw):
    before = self._flops_used
    result = original_charge(self, op_name, flop_cost, subscripts, shapes, **kw)
    cost = self._flops_used - before
    stack = []
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename == candidate:
            stack.append(f'{frame.f_code.co_name}:{frame.f_lineno}')
        frame = frame.f_back
    key = (tuple(stack), op_name)
    charges[key] += cost
    requests[key] += 1
    return result


_budget.BudgetContext._charge_op = charged
sys.argv[0] = str(HERE / 'native_eval.py')
runpy.run_path(sys.argv[0], run_name='__main__')
rows = [dict(stack=list(stack), op=op, flops=cost, calls=requests[(stack, op)])
        for (stack, op), cost in charges.most_common()]
with profile_path.open('x') as f:
    json.dump(rows, f, indent=2)
print(json.dumps(dict(profile=str(profile_path), total_flops=sum(charges.values()),
                     operation_calls=sum(requests.values()), stack_sites=len(charges))), flush=True)
