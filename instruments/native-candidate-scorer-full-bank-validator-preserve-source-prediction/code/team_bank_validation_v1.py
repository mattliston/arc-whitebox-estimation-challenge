"""Full-cost matched VALID40 for a candidate that passed its fixed screens."""
from pathlib import Path
import hashlib,json,subprocess,sys

H=Path(__file__).resolve().parent; P2=H.parent; ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python';NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
read=lambda p:json.loads(Path(p).read_text())
def pin(p):
    p=Path(p);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def once(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
d=Path(sys.argv[1]).resolve();assert d.parent==H
role=d.name
r=read(d/'result.json');assert r['passed'] and r['stage']=='TRAIN8'
candidate=r['candidate'];admission=[pin(d/'result.json')]
assert pin(candidate['path'])==candidate
out=H/(role+'_bank_validation_v1');out.mkdir()
baseline=[H/f'both_tail_valid40_b{i}_v6' for i in range(1,5)]
rosters=[]
for p in baseline:
    m=read(p/'manifest.json')
    assert m['sha256']=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
    assert pin(m['candidate'])['sha256']==m['sha256'] and m['numpy']=='2.4.6'
    rows=[json.loads(x) for x in (p/'rows.jsonl').read_text().splitlines()]
    assert [r['rank'] for r in rows]==m['ranks']
    rosters.append(m['ranks'])
once(out/'protocol.json',dict(candidate=candidate,admission=admission,rosters=rosters,
    gates=dict(pooled_adjusted_pct_at_most=-2,at_least_wins=30,paired95_upper_below=0),
    baseline_process_orders_matched=True,reused_validation_not_pristine=True,
    sources=[pin(__file__),pin(H/'native_eval.py'),pin(H/'compare_native.py')],
    baselines=[dict(manifest=pin(p/'manifest.json'),rows=pin(p/'rows.jsonl')) for p in baseline]))
def run(name,script,args,runtime):
    print(json.dumps(dict(event='launch',name=name)),flush=True)
    with (out/(name+'.log')).open('x') as f:
        proc=subprocess.run([str(LIGHT),str(GUARD),'--record',str(out/(name+'_guard.json')),
            '--seconds','360',str(runtime),str(script),*args],stdout=f,stderr=subprocess.STDOUT)
    g=read(out/(name+'_guard.json'))
    assert proc.returncode==0 and g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
    assert g['max_swap_growth_bytes']==0
    print(json.dumps(dict(event='retired',name=name,seconds=g['elapsed_s'],peak_rss=g['max_rss_bytes'])),flush=True)
tags=[]
for i,ranks in enumerate(rosters,1):
    assert pin(candidate['path'])==candidate
    tag=role+f'_bank_valid40_b{i}_v1';tags.append(tag)
    run(f'valid_b{i}',H/'native_eval.py',[candidate['path'],','.join(map(str,ranks)),tag],NATIVE)
comparison=role+'_bank_valid40_comparison_v1.json'
run('compare',H/'compare_native.py',[','.join(p.name for p in baseline),','.join(tags),comparison],LIGHT)
r=read(H/comparison)
gates=dict(pooled_adjusted=r['adjusted_pct']<=-2,wins=r['adjusted_wins']>=30,paired_interval=r['adjusted_ci95_pct'][1]<0)
once(out/'result.json',dict(passed=all(gates.values()),gates=gates,candidate=candidate,comparison=pin(H/comparison),
    adjusted_pct=r['adjusted_pct'],raw_pct=r['raw_pct'],wins=r['adjusted_wins'],ci95=r['adjusted_ci95_pct'],worker_qualified=False))
print(json.dumps(dict(event='VALID40_COMPLETE',passed=all(gates.values()),adjusted_pct=r['adjusted_pct'],
    raw_pct=r['raw_pct'],wins=r['adjusted_wins'],ci95=r['adjusted_ci95_pct'])),flush=True)
