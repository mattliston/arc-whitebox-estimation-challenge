"""Small source/data/guard bindings; importing this module is stdlib-only."""
from pathlib import Path
import ast, hashlib, importlib.util, json, os, sys

D=Path(__file__).resolve().parent
H=D.parent
EXCLUDED=[35,100,36,45,52,58,64,70,76,80,84,88]
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
TRAINABLE=('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')
FROZEN=('centers','scales','rms','first')
OPT=dict(epochs=5,lr=1e-4,lr_min=1e-5,weight_decay=2e-5,clip_norm=5.,seed=40,device='cpu')
REPLAY_GATES=dict(delta_relative_rms_max=2e-4,hidden_relative_rms_max=1e-4)

def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)

def read(p):return json.loads(Path(p).read_text())
def checked(p):
    assert pin(p['path'])==p,p['path']
    return Path(p['path'])
def once(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def plan(path,digest):
    pp=pin(path);assert pp['sha256']==digest;c=read(checked(pp))
    assert c['optimizer']==OPT and c['replay_gates']==REPLAY_GATES and c['trainable']==list(TRAINABLE) and c['frozen']==list(FROZEN)
    assert c['mean_scale']==.9997558892136928 and c['qualification_rank']==35
    index=read(checked(c['data_index']));roster=[r for r in index['ranks'] if r not in EXCLUDED][:40]
    assert c['ranks']==roster and c['fit_ranks']==roster[:32] and c['heldout_ranks']==roster[32:]
    assert [r['rank'] for r in c['data']]==roster and not set(roster)&set(EXCLUDED)
    for p in c['sources'].values():checked(p)
    checked(c['candidate']);checked(c['guard_source']);checked(c['authority'])
    authority=read(checked(c['authority']))
    assert set(roster)<=set(authority['train']) and not set(roster)&set(authority['valid'])
    indexed={r['rank']:r for r in index['rows']}
    assert c['qualification_fixture']==dict(rank=35,weights=indexed[35]['weights'])
    for r in c['data']:
        assert r==dict(rank=r['rank'],weights=indexed[r['rank']]['weights'],truth=indexed[r['rank']]['truth'])
    return pp,c

def stage(c,pp,name):
    if name=='passive':
        cmd=[c['runtime'],'-B',c['sources']['collect']['path'],pp['path'],pp['sha256'],name,'0'];seconds=120
    elif name.startswith('capture_'):
        block=int(name.split('_')[1]);assert 0<=block<8
        cmd=[c['runtime'],'-B',c['sources']['collect']['path'],pp['path'],pp['sha256'],'capture',str(block)];seconds=240
    else:
        assert name in ('replay','train');seconds=120 if name=='replay' else 600
        cmd=[c['torch_runtime'],'-B',c['sources'][name]['path'],pp['path'],pp['sha256']]
    gp=D/(name+'_guard.json')
    return dict(command=cmd,seconds=seconds,guard=str(gp),guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',str(gp),'--seconds',str(seconds),'--rss-limit','7500000000',*cmd],log=str(D/(name+'.log')))

def guard(c,pp,name):
    job=stage(c,pp,name);p=pin(job['guard']);g=read(checked(p))
    assert g['command']==job['command'] and g['completed'] and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
    assert g['rss_trigger_bytes']==7500000000 and 0<g['max_rss_bytes']<7500000000
    assert 0<g['elapsed_s']<=job['seconds'] and g['samples']>0
    return p

def native_prerequisite(c):
    """The changed physics must finish its native integrity diagnostic first."""
    checked(c['angular_builder']);builder=load(c['angular_builder']['path'],'current_angular_admission')
    ap,ac,_=builder.configuration(c['angular_plan']['path'],c['angular_plan']['sha256'])
    a=read(checked(c['angular_assessment']))
    assert a['passed_integrity'] and a['all16_finite'] and a['candidate']==c['candidate']
    assert a['plan']==ap and a['stage']=='fit4' and a['assessor']==ac['assessor']
    assert c['angular_assessment_command']==[ac['runtime'],'-B',ac['assessor']['path'],ap['path'],ap['sha256']]
    for p in a['evidence']:checked(p)
    # The sealed assessment guard/plan are supplied by root after actual completion.
    g=read(checked(c['angular_assessment_guard']))
    assert g['completed'] and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
    assert g['rss_trigger_bytes']==7500000000 and 0<g['max_rss_bytes']<7500000000
    assert g['command']==c['angular_assessment_command'] and 0<g['elapsed_s']<=120
    return a

def passive_inputs(c,pp):
    gp=guard(c,pp,'passive');rp=pin(D/'passive/complete.json');r=read(checked(rp))
    assert r['passed'] and r['plan']==pp and r['source']==c['candidate'] and r['mode']=='passive'
    assert r['rows'][0]['independent_all16_and_bill_exact'] and not r['targets_accessed']
    for p in (r['rows'][0]['packet'],r['parameters']):checked(p)
    return r,rp,gp

def replay_prerequisite(c,pp):
    native_prerequisite(c);r,rp,gp=passive_inputs(c,pp);p=pin(D/'replay_result.json');v=read(checked(p));vg=guard(c,pp,'replay')
    assert v['passed'] and v['plan']==pp and v['passive_complete']==rp and v['parameters']==r['parameters']
    return dict(passive=rp,passive_guard=gp,replay=p,replay_guard=vg)

def capture_inputs(c,pp):
    prerequisites=replay_prerequisite(c,pp);rows=[];receipts=[]
    for block in range(8):
        name=f'capture_{block}';g=guard(c,pp,name);p=pin(D/name/'complete.json');v=read(checked(p))
        assert v['passed'] and v['mode']=='capture' and v['plan']==pp and v['source']==c['candidate']
        assert v['ranks']==c['ranks'][5*block:5*block+5] and v['targets_accessed']
        assert v['prerequisites']==prerequisites
        for row in v['rows']:checked(row['packet']);rows.append(row)
        receipts.append(dict(complete=p,guard=g))
    assert [r['rank'] for r in rows]==c['ranks']
    return rows,receipts,prerequisites

def environment(c,torch=False):
    assert str(Path(sys.executable).absolute())==c['torch_runtime' if torch else 'runtime']
    assert all(os.environ.get(k)=='1' for k in THREADS) and os.getpriority(os.PRIO_PROCESS,0)>=10

def edit_literal(source,literal):
    tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_GNN_LITERAL' for x in n.targets)]
    assert len(nodes)==1 and set(literal)==set(TRAINABLE+FROZEN)
    n=nodes[0];lines=source.splitlines(keepends=True);before=''.join(lines[n.lineno-1:n.end_lineno]);after='_GNN_LITERAL = '+repr(literal)+'\n'
    assert source.count(before)==1
    result=source.replace(before,after,1);assert result.replace(after,before,1)==source
    compile(result,'angular_refit_candidate','exec')
    return result,dict(whole_source_inverse_exact=True,only_assignment='_GNN_LITERAL',parent_assignment_sha256=hashlib.sha256(before.encode()).hexdigest(),candidate_assignment_sha256=hashlib.sha256(after.encode()).hexdigest())
