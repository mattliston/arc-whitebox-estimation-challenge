"""TAG opus_engineer a40 (light: setup only, no prediction): which module globals of a candidate hold flopscope arrays
  (a) before setup, (b) after setup, (c) after setup and the reset path (_pb_reset((None, None)): what _pb_predict does when the
  setup-made arena is gone at the first predict).
A cache that setup fills and the reset does not clear would hold handles of a server state that no longer exists. The check: after
the reset, no global holds an array that it did not hold before setup. With a second file (the parent), the two lists are compared.
usage: reset_check.py <candidate.py> [<parent candidate.py>]"""
from pathlib import Path
import json
import sys
import types
import flopscope as flops
from flopscope import _ndarray


def arrays_by_global(mod):
    """global name -> number of distinct flopscope arrays reachable from it (through dicts, lists, tuples, sets, instances)."""
    out = {}
    for name, value in list(mod.__dict__.items()):
        seen, stack, count = set(), [value], 0
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            if isinstance(x, _ndarray.FlopscopeArray):
                count += 1
            elif isinstance(x, dict):
                stack.extend(x.values())
            elif isinstance(x, (list, tuple, set, frozenset)):
                stack.extend(x)
            elif hasattr(x, '__dict__') and not isinstance(x, (type, types.ModuleType, types.FunctionType)):
                stack.extend(vars(x).values())
            elif hasattr(x, '__slots__'):
                stack.extend(getattr(x, s) for s in x.__slots__ if hasattr(x, s))
        if count:
            out[name] = count
    return out


def run(path):
    candidate = Path(path).resolve()
    stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    mod = types.ModuleType('candidate_reset_check'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
    exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
    est = mod.Estimator()
    before = arrays_by_global(mod)
    with flops.BudgetContext(flop_budget=2**42, quiet=True):
        est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
    after_setup = arrays_by_global(mod)
    log = list(mod._PB_LOG)
    mod._pb_reset((None, None))
    after_reset = arrays_by_global(mod)
    return dict(before=before, after_setup=after_setup, after_reset=after_reset, pb_log=log)


results = [run(p) for p in sys.argv[1:3]]
for path, r in zip(sys.argv[1:3], results):
    left = {k: v for k, v in r['after_reset'].items() if r['before'].get(k, 0) != v}
    print(json.dumps(dict(file=Path(path).name[-60:], arrays_before_setup=sum(r['before'].values()), arrays_after_setup=sum(r['after_setup'].values()),
                          globals_filled_by_setup=sorted(k for k in r['after_setup'] if r['after_setup'][k] != r['before'].get(k, 0)),
                          left_after_reset=left, pb_log=r['pb_log'])))
if len(results) == 2:
    a, b = ({k: v for k, v in r['after_reset'].items() if r['before'].get(k, 0) != v} for r in results)
    new = {k: v for k, v in a.items() if b.get(k) != v}
    print('left after the reset by the first file and not by the second:', new or 'nothing', '->', 'PASS' if not new else 'FAIL')
else:
    left = {k: v for k, v in results[0]['after_reset'].items() if results[0]['before'].get(k, 0) != v}
    print('left after the reset:', left or 'nothing', '(give the parent file as second argument to judge: the parent itself keeps its old arena in _VIEW_SPECS)')
