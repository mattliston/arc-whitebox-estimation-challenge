"""Root-only multiplier=1.0 diagnostic; no fitting or candidate admission."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

D = Path(__file__).resolve().parent


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def checked(record):
    assert pin(record['path']) == record
    return Path(record['path']).read_bytes()


def main():
    assert len(sys.argv) == 3
    pp = pin(sys.argv[1]); assert pp['sha256'] == sys.argv[2]
    p = json.loads(checked(pp))
    assert p['schema'] == 'uniform-gain-identity-fit4-diagnostic-v1'
    assert p['multiplier'] == 1.0 and p['ranks'] == [35,100,36,45]
    assert p['assessor'] == pin(__file__)
    for record in p['pins']: checked(record)
    spec = importlib.util.spec_from_file_location('gain_identity_existing_helpers', p['fit_source']['path'])
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    q, cp, c, builder = f.configuration(p['protocol']['path'], p['protocol']['sha256'])
    assert p['protocol'] == cp and p['fit_source'] == c['fit_source']
    assert p['parent'] == c['parent'] and c['parent_role'] == 'rrpi'
    assert p['control'] == c['controls']['fit4'] and p['builder'] == c['builder']
    source, proof = builder.edit_source(checked(p['parent']).decode(), 1.0)
    assert source.encode() == checked(p['arm']['source']) and proof == p['source_inverse']
    truth = f.truths(q, c, 'fit4')
    base, ba, bp = f.panel(q, c, p['control']['arm'], f.FIT, truth, p['control']['artifacts'])
    rows, aa, ap = f.panel(q, c, p['arm'], f.FIT, truth)
    comparison = []
    for b, a in zip(base, rows):
        rank = b['rank']; assert a['rank'] == rank
        layers = [ba[rank][l].tobytes() == aa[rank][l].tobytes() for l in range(16)]
        comparison.append(dict(rank=rank, layer_bytes_equal=layers,
            prediction_file_bytes_equal=checked(bp['predictions'][str(rank)]) == checked(ap['predictions'][str(rank)]),
            all16_bytes_equal=all(layers), final_mse_exact=a['mse'] == b['mse'],
            base_flops=b['flops'], identity_flops=a['flops'], flops_delta=a['flops']-b['flops']))
    result = dict(schema='uniform-gain-identity-fit4-assessment-v1', passed_integrity=True,
        plan=pp, protocol=cp, assessor=pin(__file__), candidate=p['arm']['source'], parent=p['parent'],
        ranks=f.FIT, multiplier=1.0, comparison=comparison,
        identity_predictions_equal=all(r['all16_bytes_equal'] for r in comparison),
        final_mse_exact=all(r['final_mse_exact'] for r in comparison),
        artifacts=dict(base=bp, identity=ap), evidence=list(q.EVIDENCE.values()),
        diagnostic_only=True, fitting_performed=False, accuracy_selection=False,
        retained=False, bank_qualified=False, model_executed_by_assessor=False)
    q.once(D/'identity_assessment.json', result)
    print(json.dumps({k:result[k] for k in ('passed_integrity','identity_predictions_equal','final_mse_exact','comparison')}))


if __name__ == '__main__': main()
