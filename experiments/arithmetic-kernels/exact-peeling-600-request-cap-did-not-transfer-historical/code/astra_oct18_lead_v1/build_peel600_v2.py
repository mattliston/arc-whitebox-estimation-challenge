"""Restore a frozen, metered subset of exact peel products on the current union path."""
import ast
import hashlib
import json
from pathlib import Path
import sys

D = Path(__file__).resolve().parent
R = D.parent
parent, output = map(Path, sys.argv[1:])
assert not output.exists()
frontier_path = R/'astra_oct18_bill_v1/PEEL_DEUNION_FRONTIER_v1.json'
frontier = json.loads(frontier_path.read_text())
choice = dict(next(x for x in frontier['frontier'] if x['budget'] == 600))
excluded = {('read',(992,6144,896,0)), ('push',(1024,992,8064,928))}
choice['shapes'] = [x for x in choice['shapes'] if (x['kind'], tuple(x['shape'])) not in excluded]
choice['parent_mean_pct_before_slow_route_removal'] = choice.pop('mean_pct')
choice['excluded_known_slow_routes'] = [(kind,list(shape)) for kind,shape in sorted(excluded)]
for key in ('pct','request_delta','individual_knapsack_upper_pct'): choice.pop(key, None)
push = {tuple(x['shape']): max(0, x['request_delta']) for x in choice['shapes'] if x['kind'] == 'push'}
read = {tuple(x['shape']): max(0, x['request_delta']) for x in choice['shapes'] if x['kind'] == 'read'}
s = parent.read_text()
tree = ast.parse(s)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_rf192_bind')
old = ast.get_source_segment(s, node)
new = old.replace('live = {plan[1] for plan in _SM12_PLANS.values()}',
'''push = {k:v for k,v in saved_push.items() if k in _O18_PEEL_PUSH_SHAPES}
    read = {k:v for k,v in saved_read.items() if k in _O18_PEEL_READ_COST}
    live = {plan[1] for plans in (_SM12_PLANS, push, read) for plan in plans.values()}''')
new = new.replace('_PEEL_PUSH_PLANS, _PEEL_READ_PLANS = {}, {}',
                  '_PEEL_PUSH_PLANS, _PEEL_READ_PLANS = push, read')
assert new != old and new.count('_O18_PEEL_') == 2
s = s.replace(old, new, 1)
fragment = '''

# Astra Oct18: fixed public-shape exact peel selection. Full numerical content
# is retained. A fixed incremental request allowance keeps parent fallbacks.
_O18_PEEL_PUSH_COST = __PUSH_VALUES__
_O18_PEEL_READ_COST = __READ_VALUES__
_O18_PEEL_PUSH_SHAPES = frozenset(k[:3] for k in _O18_PEEL_PUSH_COST)
_O18_PEEL_STATE = dict(left=600, push_cost=None, push=0, read=0, rejected=0)
_O18_PEEL_PARENT_NEWBORN = _astra_bcc_newborn_impl
_O18_PEEL_PARENT_PREDICT = Estimator.predict

def _peel_take(kind, key):
    if kind == 'push':
        cost = _O18_PEEL_STATE['push_cost']
        if key not in _O18_PEEL_PUSH_SHAPES:
            return False
    else:
        cost = _O18_PEEL_READ_COST.get(key)
    if cost is None:
        return False
    if cost > _O18_PEEL_STATE['left']:
        _O18_PEEL_STATE['rejected'] += 1
        return False
    _O18_PEEL_STATE['left'] -= cost
    _O18_PEEL_STATE[kind] += 1
    return True

def _astra_bcc_newborn_impl(w, p, previous, newborn):
    if newborn is not None:
        old_columns, selected, base, t, s = newborn
        key = (int(w.shape[1]), int(previous.shape[0]),
               int(old_columns), int(base.shape[1]))
        cost = _O18_PEEL_PUSH_COST.get(key)
        if (cost is not None and cost <= _O18_PEEL_STATE['left']
                and _AB_ON and _WINOGRAD and _SMX_ON and _SCRATCH is not None
                and int(p.shape[0]) == key[1]
                and w.dtype == p.dtype == base.dtype == 'float32'):
            _O18_PEEL_STATE['push_cost'] = cost
            try:
                # The pre-union implementation pays the common product and
                # uses the exact peel only for the inherited product.
                return _JUN_PARENT_NEWBORN(w, p, previous, newborn)
            finally:
                _O18_PEEL_STATE['push_cost'] = None
    return _O18_PEEL_PARENT_NEWBORN(w, p, previous, newborn)

def _o18_peel_predict(self, mlp, budget):
    _O18_PEEL_STATE.update(left=600, push_cost=None, push=0, read=0, rejected=0)
    return _O18_PEEL_PARENT_PREDICT(self, mlp, budget)

Estimator.predict = _o18_peel_predict
'''.replace('__PUSH_VALUES__', repr(push), 1).replace('__READ_VALUES__', repr(read), 1)
# Replacement tokens must not accidentally consume the identifier suffix.
assert '_O18_PEEL_PUSH_COST = ' in fragment and '_O18_PEEL_READ_COST = ' in fragment
s += fragment
compile(s, str(output), 'exec')
output.write_text(s)
receipt = dict(parent=str(parent), parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               source=str(output), sha256=hashlib.sha256(s.encode()).hexdigest(),
               frontier_sha256=hashlib.sha256(frontier_path.read_bytes()).hexdigest(),
               status='UNTESTED_SOURCE', routes=choice, hard_request_allowance=600)
output.with_suffix('.source.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(receipt['sha256'])
