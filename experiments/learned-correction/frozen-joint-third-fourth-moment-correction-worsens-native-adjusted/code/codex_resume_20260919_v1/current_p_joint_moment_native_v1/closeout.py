"""Verify actual results and preserve a scoped negative without changing gates."""
import ast
import datetime
import math
import struct
from common import *

verify()
controller=read(HERE/'controller_result.json')
assert controller['completed'] and controller['error'] is None
check(controller['score'])
score=read(HERE/'score.json')
assert score['completed'] and not score['passed']
check(score['prediction_seal'])
seal=read(HERE/'predictions_sealed.json')
for p in seal['pins']:
    check(p)
assert seal['completed'] and seal['disabled_adapter_exact'] and seal['all_calls_retired']

def npy_payload(p):
    data=Path(p).read_bytes()
    assert data[:6]==b'\x93NUMPY'
    major=data[6]
    fmt,off=('<H',10) if major==1 else ('<I',12)
    n=struct.unpack(fmt,data[8:off])[0]
    header=ast.literal_eval(data[off:off+n].decode('latin1').strip())
    assert header==dict(descr='<f4',fortran_order=False,shape=(16,1024))
    payload=data[off+n:]
    assert len(payload)==16*1024*4
    return payload

roles={k:read(p['path'])['records'] for k,p in seal['roles'].items()}
causality=[]
for a,b in zip(roles['parent'],roles['trained']):
    av=npy_payload(a['prediction']['path']);bv=npy_payload(b['prediction']['path'])
    exact_layers=[av[l*4096:(l+1)*4096]==bv[l*4096:(l+1)*4096] for l in range(16)]
    assert all(exact_layers[:8])
    causality.append(dict(index=a['index'],rank=a['rank'],layer_bytes_exact=exact_layers))
raw=math.fsum(r['roles']['trained']['final_raw'] for r in score['rows'])/math.fsum(r['roles']['parent']['final_raw'] for r in score['rows'])
adjusted=math.fsum(r['roles']['trained']['final_adjusted'] for r in score['rows'])/math.fsum(r['roles']['parent']['final_adjusted'] for r in score['rows'])
assert abs(raw-score['raw_ratio'])<1e-14 and abs(adjusted-score['adjusted_ratio'])<1e-14
assert score['passed']==(adjusted<=.98 and all(r['adjusted_ratio']<1 for r in score['rows']))
watches=[(p,clean_watch(p.name.removesuffix('_watch.json'))) for p in sorted(HERE.glob('*_watch.json'))]
assert len(watches)==6
resources=dict(guards=len(watches),full_native_calls=9,all_groups_retired=True,
    max_rss_bytes=max(w['max_rss_bytes'] for p,w in watches),max_swap_growth_bytes=0,
    guard_wall_s=math.fsum(w['elapsed_s'] for p,w in watches),
    actual_native_flops=sum(r['flops'] for rs in roles.values() for r in rs),
    disk_free_bytes=shutil.disk_usage(HERE).free,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
write(HERE/'resources.json',resources)
write(HERE/'causal_prefix_check.json',dict(passed=True,unchanged_layers=list(range(8)),records=causality))
report=f'''# Frozen joint-moment source: native adjusted regression

The fixed full-amplitude candidate fails its separate direct-native gate.
Against exact official331760 on the two reused development networks:

| Network | Raw error ratio | Adjusted error ratio |
| --- | ---: | ---: |
'''
for r in score['rows']:
    report+=f"| {r['rank']} | {r['raw_ratio']:.9f} | {r['adjusted_ratio']:.9f} |\n"
report+=f'''
Pooled raw ratio is{raw:.12f}; adjusted ratio is{adjusted:.12f}.
The added mean cost is{(score['means']['trained']['flops']/score['means']['parent']['flops']-1)*100:.4f}%.
Both rows fail the per-network improvement condition, and the pooled result
fails the0.98 development gate. No native40, confirmation, independent panel,
client qualification, package or submission follows. No model, amplitude,
layer window, fit, or scientific threshold was changed after these results.

The earlier model's proxy0.90 gate remains failed. This new experiment was
explicitly undertaken despite that scoped failure to test its7–8% next-mean
response signal in the real estimator. Its frozen learned arrays are byte
exact. The effect includes transfer from the earlier parent to current331760
and all autonomous downstream interactions; it does not isolate one cause.
These two heavily reused networks are not independent validation or an
official score. The result rules out this fixed frozen deployment, not every
joint-moment representation or jointly trained estimator.

All three disabled-adapter calls match parent output bytes, complete bills,
variance domains and selectors. Enabled35 repeats exactly. The first8 output
layers of every enabled call also match the parent exactly, establishing the
intended pre-layer8 boundary. Both raw errors then grow through the changed
suffix. Input/prediction budget-scope ASTs match the qualified native harness;
the source inverse and helper AST inverse are verified. All predictions and
their own complete costs were sealed before the1B target arrays were opened.

All9 calls and6 guarded stages retired. Peak sampled tree RSS was
{resources['max_rss_bytes']/1e9:.6f}GB, with zero added swap; total guarded
wall time{resources['guard_wall_s']:.3f}s. New artifacts before closeout were
{resources['artifact_bytes_before_closeout']}bytes. No data download, new
sampling, GPU work, refit, or external write occurred.

Official331760 remains3.507738856975084e-9 adjusted,100/100,zero failures.
The4e-9 bank is complete;3e-9,2e-9 and approximately1e-9 are not achieved.
The overall goal stays active. Current-P and signed-message sources, all
receipts, and this frozen source model remain preserved.
'''
with (HERE/'RESULT.md').open('x') as f:
    f.write(report)
write(HERE/'closeout.json',dict(completed=True,scientific_passed=False,score=pin(HERE/'score.json'),
    resources=pin(HERE/'resources.json'),report=pin(HERE/'RESULT.md'),causality=pin(HERE/'causal_prefix_check.json'),
    original_proxy_gate_preserved=True,source_seal=pin(HERE/'sources_sealed.json'),
    closeout_source=pin(HERE/'closeout.py'),official_changed=False,goal_achieved=False))
print(dict(completed=True,adjusted_ratio=adjusted,raw_ratio=raw,resources=resources))
