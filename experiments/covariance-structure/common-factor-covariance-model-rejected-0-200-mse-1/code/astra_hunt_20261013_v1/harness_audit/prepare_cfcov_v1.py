"""Stdlib-only preparation of the fixed common-factor covariance arm; no eval."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent
O = A.parent / 'opus_reexam_20261001_v1'
SIM = A / 'physics_alternative/cutsim_cfcov_v1'
BASE = O / 'physicist/t134_refit_screen'
CTL, ARM = HERE / 'run_cfcov_control_v1', HERE / 'run_cfcov_arm_v1'
RULE = dict(cfcov=1.0, cfcov_lo=8, cfcov_hi=14)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable(req):
    r = json.loads(json.dumps(req))
    for k in ('id', 'rank', 'keep_final', 'scale', 'truth_plan'):
        r.pop(k, None)
    pl = r.pop('pl', {})
    for k in ('k4la3', 'k4la4'):
        if k in pl:
            assert pl.pop(k) == [-1.0] * 16
    if pl:
        r['pl'] = pl
    r['cutsim'] = {'opts': r['cutsim'].get('opts', {})}
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pbase-sha256', required=True)
    args = parser.parse_args()
    assert sha(SIM / 'pbase_cutsim_v1.py') == args.pbase_sha256
    for path in (CTL, ARM, HERE / 'cfcov_manifest_v1.json', HERE / 'allow_cfcov_v1.txt'):
        assert not path.exists(), ('Will not overwrite; version the preparation', path)
    source = runpy.run_path(str(O / 'scout_card/make_rb_v1.py'))
    source['main']([str(CTL), 'astracfctl', '7100584', '1', '--sim', str(SIM), 'ctl'])
    source['main']([str(ARM), 'astracfcov', '7100584', '8', '--sim', str(SIM), 'gk'])
    control = next((CTL / 'queue').glob('*.json'))
    rows = []
    for q in sorted((ARM / 'queue').glob('*.json')):
        req = json.loads(q.read_text())
        seed = req['rank']
        claim = BASE / 'claimed' / f't134_a_ctl_on_{seed}.w0.json'
        basepath = BASE / 'results' / f't134_a_ctl_on_{seed}.json'
        baseline = json.loads(basepath.read_text())
        assert comparable(req) == comparable(json.loads(claim.read_text())), ('Baseline request mismatch', seed)
        assert req['params'].endswith('/S51/point_v1/params.pt')
        assert baseline['rank'] == seed and baseline['params'] == req['params']
        assert len(baseline['mse_layers']) == 16
        pb = next(x for x in req['ncar'] if x['module'] == 'pbase')
        assert pb['gsk'] == pb['gk4'] == 1.0
        pb.update(RULE)
        req['id'] = req['id'].removesuffix('_gk') + '_cfcov'
        dest = q.with_name(req['id'] + '.json')
        dest.write_text(json.dumps(req))
        q.unlink()
        rows.append(dict(seed=seed, arm_id=req['id'], arm_request=str(dest), arm_request_sha256=sha(dest),
                         base=str(basepath), base_sha256=sha(basepath), base_mse=baseline['mse'],
                         base_request=str(claim), base_request_sha256=sha(claim)))
    assert comparable(json.loads(control.read_text())) == comparable(json.loads(Path(rows[0]['base_request']).read_text()))
    manifest = dict(rule=RULE, threshold_mean_log_percent=-2.0, rows=rows,
                    control_request=str(control), control_request_sha256=sha(control),
                    control_result=str(CTL / 'results' / control.name),
                    sources=source['private'](source['TMPL']['cutsim'], str(SIM)),
                    generator_sha256=sha(O / 'scout_card/make_rb_v1.py'),
                    worker_sha256=sha(HERE / 'worker_astra_v1.py'),
                    runtime_sha256=sha(HERE / 'otr_cut_runtime_astra_v1.py'),
                    safety_sha256=sha(HERE / 'astra_runtime_safety_v1.py'),
                    entry_sha256=sha(HERE / 'screen_entry_v2.py'),
                    comparator_sha256=sha(HERE / 'compare_cfcov_v1.py'),
                    launcher_sha256=sha(HERE / 'launch_cfcov_v1.sh'),
                    required_before_eval=['source peer review', 'guarded cfcov_toy_cert_v1.py certificate'],
                    toy_certificate_source=str(A / 'physics_alternative/cfcov_toy_cert_v1.py'),
                    toy_certificate_source_sha256=sha(A / 'physics_alternative/cfcov_toy_cert_v1.py'))
    (HERE / 'cfcov_manifest_v1.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (HERE / 'allow_cfcov_v1.txt').write_text(','.join(str(s) for s in range(7100584, 7100592)) + '\n')
    print(json.dumps(dict(control=str(CTL), arm=str(ARM), request_count=9,
                         pbase_sha256=args.pbase_sha256, comparable_to='true S51+gk11 t134 learnedON'), indent=2))


if __name__ == '__main__':
    main()
