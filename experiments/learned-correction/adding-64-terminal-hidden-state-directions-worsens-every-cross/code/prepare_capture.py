"""Stdlib-only bindings for a fixed passive RRPI20 terminal-128 acquisition."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
D=Path(__file__).resolve().parent
H=D.parent
OLD=H/'hill_climb_rrpi_k4_rowdiag_coadapt_v1'
TRAIN=[38,40,41,43,44,46,47,49,50,51,53,54,55,56,57,60,61,62,65,68]
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
PARENT_SHA='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
OLD_PROTOCOL_SHA='93eea5cc3a3ff6d7e889798ba57e78b73f466500b23e8b3801b3897537e06ef3'


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


def guard(path,command,seconds):
    p=pin(path);g=read(checked(p))
    assert g['command']==command and g['completed'] and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
    assert g['rss_trigger_bytes']==7500000000 and 0<g['max_rss_bytes']<7500000000
    assert 0<g['elapsed_s']<=seconds and g['samples']>0
    return p


def plan(path,digest):
    pp=pin(path);assert pp['sha256']==digest;c=read(checked(pp))
    assert c['schema']=='passive-rrpi-terminal128-v1' and c['ranks']==TRAIN
    assert c['blocks']==[TRAIN[i:i+5] for i in range(0,20,5)]
    assert c['parent']['sha256']==PARENT_SHA and c['old_protocol']['sha256']==OLD_PROTOCOL_SHA
    for p in [c['parent'],c['old_protocol'],c['guard_source'],*c['sources'].values()]:checked(p)
    assert [x['rank'] for x in c['data']]==TRAIN
    assert c['targets_accessed_during_capture'] is False and c['all20_byte_and_bill_parity_required'] is True
    return pp,c


def environment(c):
    assert str(Path(sys.executable).absolute())==c['runtime']
    assert all(os.environ.get(k)=='1' for k in THREADS) and os.getpriority(os.PRIO_PROCESS,0)>=10


def stage(c,pp,block):
    assert 0<=block<4;name=f'capture_{block}'
    command=[c['runtime'],'-B',c['sources']['collect']['path'],pp['path'],pp['sha256'],str(block)]
    return dict(name=name,seconds=240,command=command,guard=str(D/(name+'_guard.json')),log=str(D/(name+'.log')),
        guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/(name+'_guard.json')),
                       '--seconds','240','--rss-limit','7500000000',*command])


def controls(c,block):
    """Check only saved target-free controls, never load a truth array."""
    binding=c['controls'][block];r=read(checked(binding['complete']))
    op=read(checked(c['old_protocol']))
    command=[op['runtime'],'-B',op['sources']['collect']['path'],c['old_protocol']['path'],
             c['old_protocol']['sha256'],'capture','rrpi',str(block)]
    assert guard(binding['guard']['path'],command,180)==binding['guard']
    assert r['passed'] and r['arm']=='rrpi' and r['mode']=='capture' and r['plan']==c['old_protocol']
    assert r['source']==c['parent'] and r['ranks']==c['blocks'][block] and not r['targets_accessed']
    for row in r['rows']:
        assert row['source']==c['parent'] and not row['targets_accessed'] and row['all16_finite'] and row['readout_replay_exact']
        checked(row['packet'])
    assert [x['rank'] for x in r['rows']]==c['blocks'][block]
    return r['rows']


def capture_inputs(c,pp):
    result=[]
    for block in range(4):
        job=stage(c,pp,block);gp=guard(job['guard'],job['command'],240)
        rp=pin(D/job['name']/'complete.json');r=read(checked(rp))
        assert r['passed'] and r['source']==c['parent'] and r['plan']==pp and r['ranks']==c['blocks'][block]
        assert not r['targets_accessed'] and r['capture_references_retired']
        for row in r['rows']:
            assert row['independent_all16_and_bill_exact'] and row['terminal_state_and_delta_replay_exact']
            assert row['all16_finite'] and not row['targets_accessed'] and row['source']==c['parent']
            checked(row['packet']);result.append(dict(row=row,complete=rp,guard=gp))
    assert [x['row']['rank'] for x in result]==TRAIN
    return result


def prepare():
    op=pin(OLD/'protocol.json');assert op['sha256']==OLD_PROTOCOL_SHA;o=read(op['path'])
    assert o['train']==TRAIN and o['arms']['rrpi']['sha256']==PARENT_SHA
    for p in [o['arms']['rrpi'],o['guard_source'],o['data_index'],o['training_authority']]:checked(p)
    c=dict(schema='passive-rrpi-terminal128-v1',ranks=TRAIN,blocks=[TRAIN[i:i+5] for i in range(0,20,5)],
        parent=o['arms']['rrpi'],old_protocol=op,data=o['data'],data_index=o['data_index'],training_authority=o['training_authority'],
        runtime=o['runtime'],light=o['light'],guard_source=o['guard_source'],
        sources=dict(prepare=pin(__file__),collect=pin(D/'collect.py')),
        controls=[dict(complete=pin(OLD/f'capture_rrpi_{b}'/'complete.json'),guard=pin(OLD/f'capture_rrpi_{b}_guard.json')) for b in range(4)],
        targets_accessed_during_capture=False,all20_byte_and_bill_parity_required=True,
        replay='outside BudgetContext: H126 -> two native-order float32 SiLUs -> hidden64 -> original delta; bytes exact',
        acquisition='20 existing RRPI ranks/order, four serial cold blocks; no unwrapped reruns, no training or candidate',
        native_calls=20,root_only_execution=True)
    for b in range(4):
        rows=controls(c,b)
        for row,fixture in zip(rows,c['data'][b*5:b*5+5]):
            assert row['rank']==fixture['rank'] and row['weights']==fixture['weights']
    once(D/'protocol.json',c);pp=pin(D/'protocol.json')
    once(D/'capture_commands.json',dict(protocol=pp,environment={k:'1' for k in THREADS},nice_at_least=10,
        serial=True,jobs=[stage(c,pp,b) for b in range(4)]))
    print(json.dumps(dict(protocol=pp,commands=pin(D/'capture_commands.json'))))


if __name__=='__main__':
    assert sys.argv[1:]==['--prepare'];prepare()
