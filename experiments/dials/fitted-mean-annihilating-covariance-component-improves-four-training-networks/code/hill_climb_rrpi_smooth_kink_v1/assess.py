"""Root-only paired screen; unchanged native panel/truth/metric helpers."""
from pathlib import Path
import importlib.util, json, os, sys

D = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    assert len(sys.argv) == 3
    b = module(D/'prepare.py', 'smooth_kink_preflight')
    pp, c, prerequisites = b.configuration(Path(sys.argv[1]).resolve(), sys.argv[2])
    f = module(c['helper']['path'], 'smooth_kink_saved_helpers')
    q = module(c['engine']['path'], 'smooth_kink_saved_metrics')
    assert str(Path(sys.executable).absolute()) == c['runtime'] == str(q.NATIVE)
    assert q.np.__version__ == '2.4.6' and all(os.environ.get(k) == '1' for k in q.THREADS)
    stage, ranks = c['stage'], c['ranks']; truth = f.truths(q, c, stage)
    control = c['controls'][stage]
    assert control['arm']['source'] == c['parent'] and c['arm']['source'] == c['candidate']
    base, ba, bp = f.panel(q, c, control['arm'], ranks, truth, control['artifacts'])
    rows, aa, ap = f.panel(q, c, c['arm'], ranks, truth)
    for rank in ranks: assert ba[rank][0].tobytes() == aa[rank][0].tobytes()
    metrics = q.metrics(base, rows)
    gates = dict(raw_gain=metrics['raw_pct'] < 0, adjusted_gain=metrics['adjusted_pct'] < 0,
                 all16_finite=True, layer0_exact=True)
    result = dict(schema='fixed-smooth-kink-paired-assessment-v1', stage=stage, passed_integrity=True,
        admitted=all(gates.values()), check8_permitted=stage == 'fit4' and all(gates.values()),
        component_point_passed=stage == 'check8' and all(gates.values()), plan=pp, assessor=q.pin(__file__),
        parent=c['parent'], candidate=c['candidate'], ranks=ranks, metrics=metrics, gates=gates,
        artifacts=dict(base=bp, candidate=ap), prerequisites=prerequisites, evidence=list(q.EVIDENCE.values()),
        fitting_performed=False, old_failures_preserved=True, adaptive_reused_panels=True,
        wins_and_ci_descriptive_only=True, retained=False, broad_admitted=False, bank_qualified=False, model_executed=False)
    q.once(D/(stage+'_assessment.json'), result)
    print(json.dumps({k:result[k] for k in ('stage','passed_integrity','admitted','metrics','gates')}))


if __name__ == '__main__': main()
