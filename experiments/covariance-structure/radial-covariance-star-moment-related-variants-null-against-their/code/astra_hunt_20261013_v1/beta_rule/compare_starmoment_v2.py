"""Read-only stdlib audit: control, then all paired8. Optional worker log telemetry.

Usage: python -B compare_starmoment_v1.py control
       python -B compare_starmoment_v1.py [worker_log]
No output files are written. Root owns invocation of every analysis command.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE/'starmoment_manifest_v2.json').read_text())
    for name, expected in manifest['source_private'].items():
        assert hashlib.sha256((HERE/'cutsim_starmoment_v2'/name).read_bytes()).hexdigest() == expected
    for name, key in (('otr_m_starmoment_v1.py','m_private_sha256'), ('worker_starmoment_v1.py','worker_sha256'), ('otr_cut_runtime_astra_v1.py','runtime_sha256'), ('astra_runtime_safety_v1.py','safety_sha256')):
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == manifest[key]
    for row in manifest['rows']:
        for key in ('base', 'base_request'):
            assert hashlib.sha256(Path(row[key]).read_bytes()).hexdigest() == row[key+'_sha256']
    control = json.loads(Path(manifest['control_result']).read_text())
    base0 = json.loads(Path(manifest['rows'][0]['base']).read_text())
    assert control['params'] == base0['params']
    assert control['mse'] == base0['mse'] and control['mse_layers'] == base0['mse_layers'], 'No-op control mismatch'
    print('No-op control: all 16 layer MSEs exactly match true S51+gk11 t134 control')
    if len(sys.argv) > 1 and sys.argv[1] == 'control':
        return
    records, effects, by_id = [], [], {}
    for row in manifest['rows']:
        result = HERE/'run_starmoment_arm_v2/results'/(row['arm_id']+'.json')
        arm, base = json.loads(result.read_text()), json.loads(Path(row['base']).read_text())
        assert arm['rank'] == base['rank'] and arm['params'] == base['params']
        assert arm['cutsim']['sources'] == manifest['source_private']
        args = next(m['arguments'] for m in arm['ncar'] if m['module']=='pbase')
        assert all(args[k] == v for k,v in manifest['rule'].items())
        effect = math.log(arm['mse']/base['mse'])
        effects.append(effect)
        record = dict(seed=arm['rank'], log_percent=100*effect, arm_mse=arm['mse'], base_mse=base['mse'])
        records.append(record); by_id[row['arm_id']] = record
    n = len(effects); mean = sum(effects)/n
    se = math.sqrt(sum((x-mean)**2 for x in effects)/(n-1)/n)
    if len(sys.argv) > 1:
        pattern = re.compile(r'^astra starmoment L(\d+) s ([+\-0-9.]+) J0 (\S+) Jm (\S+) Jp (\S+) eta (\S+) ssb (\S+) fallback (none|uclamp)$')
        pending = []
        pending_j21 = []
        for line in Path(sys.argv[1]).read_text().splitlines():
            match = pattern.fullmatch(line)
            if match:
                l,s,j0,jm,jp,eta,ssb,fallback = match.groups()
                chosen = {0.0:0,-1.0:1,1.0:2}[float(s)]
                losses = [float(j0), float(jm), float(jp)] if fallback == 'none' else None
                if losses is not None:
                    assert losses[chosen] == min(losses)
                else:
                    assert chosen == 0
                pending.append(dict(layer=int(l), s=float(s), losses=losses,
                                    chosen_loss_ratio=losses[chosen]/losses[0] if losses and losses[0] else None,
                                    eta=float(eta), ssb=float(ssb), fallback=fallback))
            elif line.startswith('astra starmoment_j21 '):
                pending_j21.append(json.loads(line[len('astra starmoment_j21 '):]))
            elif line.startswith('{'):
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                if event.get('event') == 'done' and event.get('id') in by_id:
                    assert [x['layer'] for x in pending] == list(range(8,15)), (event['id'], pending)
                    assert [x['layer'] for x in pending_j21] == list(range(8,15)), (event['id'], pending_j21)
                    for j3, j21 in zip(pending, pending_j21):
                        assert j3['layer'] == j21['layer'] and j3['s'] == j21['j3_selected_s']
                        assert j3['fallback'] == j21['fallback']
                    by_id[event['id']]['telemetry'] = pending
                    by_id[event['id']]['j21_telemetry'] = pending_j21
                    pending = []
                    pending_j21 = []
                elif event.get('event') == 'error':
                    pending = []
                    pending_j21 = []
        assert all('telemetry' in r for r in records), 'Worker log lacks telemetry for some arm results'
    print(json.dumps(dict(n=n, mean_log_percent=100*mean, se_percent=100*se,
                          pooled_percent=100*(sum(r['arm_mse'] for r in records)/sum(r['base_mse'] for r in records)-1),
                          lower=sum(e<0 for e in effects), per_network=records,
                          satisfies_torch_screen=100*mean <= manifest['threshold_mean_log_percent']), indent=2))


if __name__ == '__main__':
    main()
