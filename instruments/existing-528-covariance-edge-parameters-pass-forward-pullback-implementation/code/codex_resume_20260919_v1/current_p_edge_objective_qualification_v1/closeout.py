from common import *
verify()
r=read(HERE/'result.json');assert r['completed'] and r['passed'] and r['error'] is None
for k in ('unit','full'):check(r[k])
f=read(HERE/'full_result.json');u=read(HERE/'unit_result.json')
for p in f['records']:check(p)
for k in ('state_boundary','restored_boundary'):check(f[k])
b=read(f['state_boundary']['path']);check(b['payload']);assert not b['candidate_eligible']
guards=[clean_watch(n) for n in ('unit','full')]
resource=dict(guards=2,full_calls=4,backward_calls=3,all_groups_retired=True,
    peak_rss_bytes=max(g['max_rss_bytes'] for g in guards),added_swap_bytes=0,
    free_disk_bytes=shutil.disk_usage(HERE).free,controller_session14954_exit_code=0)
write(HERE/'resources.json',resource)
report=f'''# Existing covariance-edge parameters: implementation qualified

The exact official331760 source initializes all parameters. Only the existing
five covariance-edge leaves (528 FP32 values) are eligible for learning.
All P arrays, A, cstar and edge normalizers/scales stay fixed. The saved source
inverse recovers the prior model_v2 exactly; only trainability and checks change.

The7-node independent forward check has relative error
{u['forward_relative_error']:.12g}. The largest independent analytic pullback
error is{max(u['analytic_pullback_relative_errors'].values()):.12g}, below the
fixed2e-5 gate. Four full calls (frozen35, gradient35/100/35) preserve frozen /
checkpointed outputs and repeated35 prediction, gradient, variance, selector
and state bytes. All parameters remain unchanged during these calls.

The real528-leaf Adam boundary, settings and RNG round trip exactly. A second
identical synthetic-gradient update agrees byte-for-byte. These two synthetic
updates are a transport check and are not an eligible model or fit.

Both guards retired, peak{resource['peak_rss_bytes']/1e9:.6f}GB, zero added swap,
and post-graph retirement below2GB. No reference mean array was opened, no
production candidate was exported, and no accuracy or native-derivative
qualification is claimed. The separate fixed training stage must initialize
from the official parent, never the synthetic qualification checkpoint.

This is related to the older failed16-output native SPSA and joint covariance
experiments; their negative results remain preserved. Training and actual
native validation are still required. Official331760 remains3.507738856975084e-9.
'''
with (HERE/'RESULT.md').open('x') as h:h.write(report)
write(HERE/'closeout.json',dict(completed=True,implementation_passed=True,scientific_accuracy_claim=False,
    report=pin(HERE/'RESULT.md'),resources=pin(HERE/'resources.json'),result=pin(HERE/'result.json'),
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
    official_changed=False,goal_achieved=False))
print(dict(closed=True,implementation_passed=True,peak_rss_bytes=resource['peak_rss_bytes']),flush=True)
