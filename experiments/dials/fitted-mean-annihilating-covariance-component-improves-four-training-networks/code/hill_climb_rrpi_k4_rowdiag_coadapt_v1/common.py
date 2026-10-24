"""Small, stdlib-only bindings shared by the fixed two-arm head study."""
from pathlib import Path
import hashlib, importlib.util, json, os

D=Path(__file__).resolve().parent
H=D.parent
TRAIN=[38,40,41,43,44,46,47,49,50,51,53,54,55,56,57,60,61,62,65,68]
ARMS=('rrpi','k4')
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
GATES=dict(physical_gain_vs_rrpi_original_pct_min=.2,physical_gain_vs_rrpi_fitted_pct_min=.2,
           physical_folds_vs_rrpi_fitted_min=4,control_gain_vs_rrpi_original_pct_min=.2,
           control_folds_vs_rrpi_original_min=4,raw_descriptive=True,all_finite=True)


def pin(p):
    p=Path(p).resolve(); digest=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20): digest.update(b)
    return dict(path=str(p),sha256=digest.hexdigest(),bytes=p.stat().st_size)


def read(p): return json.loads(Path(p).read_text())


def checked(p):
    assert pin(p['path'])==p,p['path']
    return Path(p['path'])


def once(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def plan(path,digest):
    pp=pin(path);assert pp['sha256']==digest;c=read(checked(pp))
    assert c['train']==TRAIN and c['ridge']==.001 and c['fold_rule']=='roster_index_modulo_5'
    assert c['cv_gates']==GATES
    for entry in c['sources'].values():checked(entry)
    for entry in c['arms'].values():checked(entry)
    assert set(c['arms'])==set(ARMS)
    return pp,c


def stage(c,pp,name):
    if name.startswith('control_'):
        mode='control';arm=name[8:];block=0;seconds=90
    else:
        _,arm,b=name.split('_');mode='capture';block=int(b);seconds=180
    assert arm in ARMS and 0<=block<4
    command=[c['runtime'],'-B',c['sources']['collect']['path'],pp['path'],pp['sha256'],mode,arm,str(block)]
    return dict(name=name,seconds=seconds,command=command,guard=str(D/(name+'_guard.json')),
                guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/(name+'_guard.json')),
                               '--seconds',str(seconds),'--rss-limit','7500000000',*command],log=str(D/(name+'.log')))


def guard(path,command,seconds):
    p=pin(path);g=read(checked(p))
    assert g['command']==command and g['completed'] and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
    assert g['rss_trigger_bytes']==7500000000 and 0<g['max_rss_bytes']<7500000000
    assert 0<g['elapsed_s']<=seconds and g['samples']>0
    return p


def prerequisites(c):
    K=H/'hill_climb_rrpi_k4_rowdiag_v1';pp=pin(K/'physical_native2_plan.json');p=read(pp['path'])
    assert pp==c['physical_native2_plan'] and p['candidate']==c['arms']['k4'] and p['parent']==c['arms']['rrpi']
    a=pin(K/'physical_native2_assessment.json');r=read(a['path'])
    assert r['passed_integrity'] and r['all16_finite'] and r['plan']==pp and r['candidate']==c['arms']['k4']
    assert r['exact_parity_layers']==[0,1] and r['assessor']==c['physical_assessor']
    cmd=[c['runtime'],'-B',c['physical_assessor']['path'],pp['path'],pp['sha256']]
    g=guard(K/'physical_native2_assess_guard.json',cmd,120)
    return [a,g]


def environment(c):
    import sys
    assert str(Path(sys.executable).absolute())==c['runtime']
    assert all(os.environ.get(k)=='1' for k in THREADS) and os.getpriority(os.PRIO_PROCESS,0)>=10


def capture_inputs(c,pp):
    result={arm:[] for arm in ARMS}
    for arm in ARMS:
        for block in range(4):
            job=stage(c,pp,f'capture_{arm}_{block}');gp=guard(job['guard'],job['command'],job['seconds'])
            rp=pin(D/job['name']/'complete.json');r=read(rp['path'])
            assert r['passed'] and r['arm']==arm and r['source']==c['arms'][arm] and r['plan']==pp
            assert r['ranks']==TRAIN[5*block:5*block+5] and not r['targets_accessed']
            for row in r['rows']:
                checked(row['packet']);result[arm].append(dict(row=row,complete=rp,guard=gp))
        assert [x['row']['rank'] for x in result[arm]]==TRAIN
    return result
