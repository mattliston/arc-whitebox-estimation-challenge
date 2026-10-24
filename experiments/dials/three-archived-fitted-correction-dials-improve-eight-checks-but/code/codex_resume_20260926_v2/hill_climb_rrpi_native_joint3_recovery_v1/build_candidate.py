"""Inactive replay of the single archived TRAIN4 proposal; no fit or model execution."""
from pathlib import Path
import ast
import hashlib
import json
import sys

D = Path(__file__).resolve().parent


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def load(record):
    assert pin(record['path']) == record, record['path']
    return json.loads(Path(record['path']).read_text())


def main():
    assert len(sys.argv) == 4
    plan_pin = pin(D/'recovery_plan.json'); assert plan_pin['sha256'] == sys.argv[1]
    plan = load(plan_pin)
    selection_pin = pin(sys.argv[2]); assert selection_pin['sha256'] == sys.argv[3]
    selection = load(selection_pin)
    assert selection['root_selected'] is True and selection['recovery_plan'] == plan_pin
    assert selection['builder'] == plan['builder'] == pin(__file__)
    assert selection['values'] == plan['values']
    fit = load(plan['original_fit']); closed = load(plan['original_closeout'])
    assert fit['passed_integrity'] and fit['endpoint_admitted'] is False and fit['endpoint'] is None
    assert fit['fit_ranks'] == plan['fit_ranks'] == [35,100,36,45]
    assert fit['values'] == plan['values']
    assert closed['no_endpoint_exported'] and closed['no_actual_endpoint_or_check_or_broad_admitted']
    assert pin(plan['parent']['path']) == plan['parent']
    assert pin(plan['original_parent']['path']) == plan['original_parent']
    def assignments(path):
        out = {}
        for n in ast.parse(Path(path).read_text()).body:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name): out[t.id] = n.value
        return out
    old = assignments(plan['original_parent']['path']); new = assignments(plan['parent']['path'])
    for name in plan['compatible_assignments']:
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False), name
    values = plan['values']
    assert values == dict(bscale=1.0360701883865824, newborn=0.9946877785393429, c1=0.5127473198109436)
    tail = ('\n# Replay only the archived fixed TRAIN4 joint3 step on current RRPI.\n'
            '_S19_BSCALE = '+repr(values['bscale'])+'\n'
            '_S19_LAM = (_S19_LAM[0], '+repr(values['newborn'])+')\n'
            '_K31R_TABLE = {l: (c[0], '+repr(values['c1'])+') for l, c in _K31R_TABLE.items()}\n').encode()
    parent = Path(plan['parent']['path']).read_bytes(); candidate = parent+tail
    assert candidate[:-len(tail)] == parent
    assert len(ast.parse(tail.decode()).body) == 3
    compile(candidate, str(D/'candidate.py'), 'exec')
    assert not (D/'candidate.py').exists() and not (D/'build.json').exists()
    with (D/'candidate.py').open('xb') as f: f.write(candidate)
    record = dict(passed=True, candidate=pin(D/'candidate.py'), parent=plan['parent'],
        builder=pin(__file__), recovery_plan=plan_pin, selection=selection_pin,
        original_fit=plan['original_fit'], original_closeout=plan['original_closeout'],
        values=values, suffix=tail.decode(), source_inverse_exact=True,
        old_forecast_failure_preserved=True, new_fit_performed=False, numerical_execution=False,
        model_executed=False, retained=False, bank_qualified=False, submitted=False)
    with (D/'build.json').open('x') as f: json.dump(record,f,indent=2); f.write('\n')
    print(json.dumps(dict(candidate=record['candidate'],build=pin(D/'build.json'))))


if __name__ == '__main__':
    main()
