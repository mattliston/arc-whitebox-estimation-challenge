from datetime import datetime, timezone
import shutil
import subprocess
from common import HERE,D,DIAG,PARENT,read,write,pin,verify,seal_check

seal_check();assert read(HERE/'completed.json')['completed']
r=read(HERE/'response_result.json');outcome=read(HERE/'outcome.json')
guards=[read(p) for p in sorted(HERE.glob('*_watch.json'))]
assert len(guards)==17
assert all(g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0 for g in guards)
live={int(x) for x in subprocess.check_output(['/bin/ps','-axo','pgid='],text=True).split()}
assert not any(g['pid'] in live for g in guards)
calls=[read(p)['records'][0] for p in HERE.glob('r*/result.json')]
assert len(calls)==15
resources=dict(guarded_groups=17,all_groups_retired=True,full_native_calls=15,
    full_native_flops=r['full_native_flops'],peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_native_self_rss_bytes=max(c['max_rss_bytes'] for c in calls),
    max_swap_growth_bytes=max(g['max_swap_growth_bytes'] for g in guards),
    guarded_wall_s=sum(g['elapsed_s'] for g in guards),
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free,fit_steps=0,new_samples=0,new_downloads=0,
    GPU_calls=0,truth_arrays_opened=0)
assert resources['artifact_bytes_before_closeout']<50_000_000
assert max(resources['peak_group_rss_bytes'],resources['peak_native_self_rss_bytes'])<8_000_000_000
write(HERE/'resources.json',resources)
lines=['# FP64 sketch producer: actual native response result','',
    'The single intervention widens only the10 shared range sketches and their QR input.',
    'Q returns toFP32; core, factors, physical trajectory, selectors and all fitted values otherwise retain original code.',
    'All7 complete-source inverses pass. The43 tiny/wide FNP helper cases pass against the FP64 reference.',
    'Fifteen actual cold native calls pass domain/cache checks; base first5 original rows and probe first8 own-base rows are exact.',
    'The repeated36/head0plus output bytes and full bills match exactly.','',
    '| Rank | Head | Original asymmetry | FP64 sketch asymmetry | Original curvature | FP64 sketch curvature | Pass |',
    '| --- | --- | --- | --- | --- | --- | --- |']
for c in r['cells']:
    lines.append(f"| {c['rank']} | {c['coordinate']} | {c['original_asymmetry']:.9g} | {c['forward_vs_centered_relative']:.9g} | {c['original_curvature']:.9g} | {c['double_curvature_relative']:.9g} | {c['passed']} |")
lines += ['',f"All four fixed response gates pass: {r['passed']}.",
    'Each cell requires spacing SNR>=20 and both response ratios<=.10.',
    f"Complete own-cold cost ratios range {min(x['cost_ratio'] for x in r['rows']):.12g}..{max(x['cost_ratio'] for x in r['rows']):.12g}.",
    'These are genuine additional billed operations, not a free precision/reference change.',
    'No targets, fitting, accuracy scores, client qualification, package or upload are included.',
    ('A separately declared actual-cost supervised proposal can be designed; no end-to-end gain is established.' if r['passed'] else
     'The fixed intervention is closed at its response gate. No epsilon/coordinate/precision/scope rescue or fit follows.'),
    'The frozen-input diagnostic remains valid; neither conditioning nor improved response alone proves better final MSE.',
    '',f"All17 guarded groups retired;15 full native calls;peak sampled{resources['peak_group_rss_bytes']/1e9:.3f}GB;zero added swap.",
    f"Artifacts {resources['artifact_bytes_before_closeout']/1e6:.3f}MB;no new samples/downloads/GPU.",
    'Official331684 remains3.5844144212297072e-9,100/100 complete. The3/2/~1e-9 milestones remain unmet.','']
with (HERE/'RESULT.md').open('x') as f:f.write('\n'.join(lines))
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='closeout.json')
write(HERE/'closeout.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
    artifact_pins=[pin(p) for p in files],external_pins=[pin(PARENT),pin(D/'official_best_331684.json'),pin(DIAG/'closeout.json')],
    outcome=outcome,resources=resources,goal_active=True,goal_complete=False))
for p in read(HERE/'closeout.json')['artifact_pins']:verify(p)
print(dict(completed=True,outcome=outcome,resources=resources),flush=True)
