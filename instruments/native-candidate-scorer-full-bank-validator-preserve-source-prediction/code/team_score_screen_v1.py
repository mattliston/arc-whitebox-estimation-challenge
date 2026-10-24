"""Fixed teammate candidate screen with full native score; one guarded worker at a time."""
from pathlib import Path
import hashlib,json,subprocess,sys

H=Path(__file__).resolve().parent; HERE=Path(sys.argv[1]).resolve(); P2=H.parent; ROOT=P2.parents[3]
assert HERE.parent==H and HERE.is_dir()
PREFIX=HERE.name
assert PREFIX.replace('_','').isalnum()
LIGHT=ROOT/'.venv/bin/python'
NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
read=lambda p:json.loads(Path(p).read_text())
def pin(p):
    p=Path(p); return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def once(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
frozen=read(HERE/'candidates.json')
for e in [frozen['bank'],frozen['component'],frozen['protocol']]+[x['candidate'] for x in frozen['arms']]:
    assert pin(e['path'])==e
once(HERE/'run_seal.json',dict(sources=[pin(__file__),pin(HERE/'candidates.json'),pin(HERE/'PROTOCOL.md')]
    +[pin(H/'native_eval.py'),pin(H/'compare_native.py')],fixed_actual_score_screen=True))
def run(name,script,args,runtime=NATIVE):
    print(json.dumps(dict(event='launch',name=name)),flush=True)
    with (HERE/(name+'.log')).open('x') as f:
        proc=subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/(name+'_guard.json')),
            '--seconds','300',str(runtime),str(script),*args],stdout=f,stderr=subprocess.STDOUT)
    g=read(HERE/(name+'_guard.json'))
    assert proc.returncode==0 and g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
    assert g['max_swap_growth_bytes']==0
    print(json.dumps(dict(event='retired',name=name,seconds=g['elapsed_s'],peak_rss=g['max_rss_bytes'])),flush=True)
def evaluate(label,source,ranks):
    tag=PREFIX+'_' +label+'_v1'
    run(label,H/'native_eval.py',[str(source),ranks,tag]); return tag
def compare(label,base,candidate):
    path=PREFIX+'_' +label+'_comparison_v1.json'
    run('compare_'+label,H/'compare_native.py',[base,candidate,path],LIGHT)
    result=read(H/path)
    print(json.dumps(dict(event='score',name=label,raw_pct=result['raw_pct'],adjusted_pct=result['adjusted_pct'],wins=result['adjusted_wins'])),flush=True)
    return result
base='tail_cut_short_parent_train2_v1'
m=read(H/base/'manifest.json');assert m['sha256']==frozen['bank']['sha256'] and m['ranks']==[35,100] and m['numpy']=='2.4.6'
old=H/'both_tail_train2_v6'
for rank in (35,100):assert pin(H/base/f'pred_{rank}.npy')['sha256']==pin(old/f'pred_{rank}.npy')['sha256']
bg=read(H/'incumbent_tail_cut_screens_v1/parent_train2_guard.json');assert bg['completed'] and bg['whole_group_retired'] and bg['max_swap_growth_bytes']==0
rows=[]; eligible=[]
for arm in frozen['arms']:
    name=arm['name']; tag=evaluate(name+'_train2',arm['candidate']['path'],'35,100')
    result=compare(name+'_train2',base,tag)
    passed=result['adjusted_wins']==2 and result['adjusted_pct']<=-1 and max(r['raw_pct'] for r in result['per_network'])<=5
    row=dict(name=name,tag=tag,passed=passed,comparison=pin(H/(PREFIX+'_' +name+'_train2_comparison_v1.json')),
        adjusted_pct=result['adjusted_pct'],raw_pct=result['raw_pct'])
    rows.append(row)
    if passed:eligible.append((result['adjusted_pct'],name,tag,arm['candidate']))
once(HERE/'train2_result.json',dict(rows=rows,eligible=len(eligible)))
if not eligible:
    once(HERE/'result.json',dict(passed=False,stage='TRAIN2',rows=rows,validation_opened=False))
    print('SHORT_SCREENS_CLOSED_NO_SURVIVOR',flush=True)
else:
    _,name,tag,candidate=min(eligible)
    once(HERE/'selected.json',dict(name=name,candidate=candidate,selection='lowest TRAIN2 adjusted score among fixed eligible arms'))
    base6='last_join_shift_parent_train6_v1'
    bm=read(H/base6/'manifest.json');assert bm['sha256']==frozen['bank']['sha256'] and bm['ranks']==[36,45,52,58,64,70] and bm['numpy']=='2.4.6'
    bg6=read(H/'last_join_shift_screens_v1/parent_train6_guard.json');assert bg6['completed'] and bg6['whole_group_retired'] and bg6['max_swap_growth_bytes']==0
    cand6=evaluate(name+'_train6',candidate['path'],'36,45,52,58,64,70')
    result=compare(name+'_train8',base+','+base6,tag+','+cand6)
    gates=dict(pooled_gain=result['adjusted_pct']<=-2,wins=result['adjusted_wins']>=6,
        max_regression=max(r['adjusted_pct'] for r in result['per_network'])<=5,
        paired_interval=result['adjusted_ci95_pct'][1]<0)
    once(HERE/'result.json',dict(passed=all(gates.values()),stage='TRAIN8',gates=gates,name=name,candidate=candidate,
        comparison=pin(H/(PREFIX+'_' +name+'_train8_comparison_v1.json')),validation_opened=False))
    print(json.dumps(dict(event='SHORT_SCREENS_COMPLETE',passed=all(gates.values()),name=name,
        adjusted_pct=result['adjusted_pct'],gates=gates)),flush=True)
