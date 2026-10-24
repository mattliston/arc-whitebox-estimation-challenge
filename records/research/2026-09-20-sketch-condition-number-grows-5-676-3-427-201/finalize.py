from datetime import datetime, timezone
import shutil
import subprocess
from common import HERE, D, read, write, pin, verify, seal_check

seal_check(); assert read(HERE/'completed.json')['completed']
a=read(HERE/'analysis.json');assert a['capture_qualification_passed']
guards=[read(p) for p in sorted(HERE.glob('*_watch.json'))]
assert len(guards)==7 and all(g['completed'] and g['whole_group_retired'] for g in guards)
live={int(x) for x in subprocess.check_output(['/bin/ps','-axo','pgid='],text=True).split()}
assert not any(g['pid'] in live for g in guards)
for p in read(HERE/'sources_sealed.json')['pins']:verify(p)
size=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert size < 1024**3
resources=dict(guarded_groups=len(guards),all_groups_retired=True,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_native_self_rss_bytes=max(read(p)['max_self_rss_bytes'] for p in HERE.glob('capture_*/result.json')),
    max_swap_growth_bytes=max(g['max_swap_growth_bytes'] for g in guards),
    guarded_wall_s=sum(g['elapsed_s'] for g in guards),full_native_calls=5,
    full_native_flops=a['full_native_flops'],diagnostic_copy_flops=a['diagnostic_copy_flops'],
    artifact_bytes_before_closeout=size,disk_free_bytes=shutil.disk_usage(HERE).free,
    fit_steps=0,new_samples=0,new_downloads=0,truth_arrays_opened=0)
assert resources['max_swap_growth_bytes']==0
write(HERE/'resources.json',resources)
text='''# Current shared-sketch rounding diagnostic completed

Five rank38 cold replays reproduce all16 original output bytes and all full
bills after subtracting explicitly metered snapshots. Original base repeats
all saved trace arrays exactly. Three ownership/QR/Gram fixtures pass. All40
actual native QR results equal FP64 refactorization rounded toFP32 exactly.

Frozen-current-input FP64 sketch reconstruction differs from the native
sketch by3.91e-6..5.56e-6 relative on the base trajectory. Native condition
numbers grow from5,676 at layer5 to3,427,201 at layer14. Corresponding
relative Frobenius projector differences grow from0.000887 to0.212442.
At the first changed sketch (layer10), native centered projector response
norm is0.581818, versus0.000165167 for the independently reconstructed
same-input FP64 sketch. Their forward/center discrepancies are1.80812 and
0.0697395, respectively. Later FP64 references still inherit perturbed
native states; their remaining nonlinearity cannot isolate a root cause.

This establishes local sketch arithmetic sensitivity on this network and
these original head perturbations, not a causal end-to-end MSE improvement.
The diagnostic changes no estimator or parameters and opens no targets.
The previous canonical-order and original response experiments remain failed.
No fitting or submission is licensed by this diagnostic alone. A separate
full native sketch-precision intervention is needed to test response stability
and actual own cost. Merely declaring QR64 is not the proposed mechanism:
NumPy already widens QR internally, while the sketch producer isFP32.

Layers5..9 have zero cross-perturbation sketch response. In analysis.json the
field base_sketch_rounding_over_centered_response uses a1e-300 numerical
denominator floor; its large values at those layers are NOT finite response
ratios and must be interpreted as undefined. The response fields themselves
correctly use null for zero denominators. No scientific conclusion uses the
floored field.

Seven guarded groups retired;5 full calls,2.072718626895TF including
159,583,360 diagnostic copyF; peak sampled6.342GB, zero added swap.
About639MB artifacts; no new samples, downloads, GPU or training.
Official331684 stays3.5844144212297072e-9; remaining milestones unmet.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(text)
paths=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='closeout.json')
write(HERE/'closeout.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
    pins=[pin(p) for p in paths],external_pins=read(HERE/'sources_sealed.json')['pins'],
    resources=resources,goal_active=True,accuracy_benefit_proven=False))
for p in read(HERE/'closeout.json')['pins']:verify(p)
print(dict(completed=True,artifacts=len(paths),resources=resources))
