import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
A=P2/'astra_phase2_reexamination_v1'
FIT=D/'current_p_signed_branch_fit_v1'
QUAL=D/'current_p_batch_process_qualification_v1'
Q=D/'current_p_signed_branch_qualification_v2'
RUNTIME=Path('/Users/matt/anaconda3/bin/python')

def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def resource_floor():
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<1_000_000_000
def verify(data=False):
    seal=read(HERE/'sources_sealed.json')
    for entry in seal['pins']+(seal['data_pins'] if data else []):check(entry)
    resource_floor()
def qualified_modules():
    assert 'common' not in sys.modules
    sys.path.insert(0,str(Q))
    import common
    import model_branch
    import allocator
    assert Path(common.__file__).resolve()==Q/'common.py'
    assert Path(model_branch.__file__).resolve()==Q/'model_branch.py'
    assert Path(allocator.__file__).resolve()==Q/'allocator.py'
    common.verify()
    assert read(Q/'result.json')['passed']
    assert read(Q/'native_qualification.json')['passed'] and read(Q/'torch_qualification.json')['passed']
    return model_branch,allocator
def numerical_workers():
    exes={str(RUNTIME),str(A/'openblas_runtime_env_v1/bin/python'),str(P2/'data_gate/.client-venv/bin/python')}
    result=[]
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=5).splitlines():
        fields=line.strip().split(None,2)
        if len(fields)==3 and fields[2].split()[0] in exes:result.append(line)
    return result
def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['error'] is None
    assert w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    return w
def guarded(name,seconds,script,*args):
    verify()
    assert not numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as log:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    watch=clean_watch(name)
    assert not numerical_workers()
    resource_floor()
    print(dict(completed=name,wall_s=watch['elapsed_s'],peak_rss_bytes=watch['max_rss_bytes']),flush=True)
    return watch

GRAPH_KEYS=['index','rank','update','within_batch','weights','target','final_mse','earlier_mse',
    'batch_scaled_loss','parameter_hashes_during_graph','domain','selectors','states','visits',
    'fixed_parameters_unchanged','validation_targets_accessed','card_retired']
UPDATE_KEYS=['update','ranks','learning_rate','pre_clip_gradient_norm','fixed_parameters_unchanged','parameter_hashes']
CHECKPOINT_KEYS=['complete','completed_updates','tensors','parameter_keys','trainable_keys','optimizer_groups',
    'round_trip_exact','live_values_unchanged','mandatory_boundary_only','candidate_eligible']
def projection(value,keys):return {k:value[k] for k in keys}
def contents(value):return {k:value[k] for k in ('bytes','sha256')}
def prefix_graph(record):
    index=record['index'];checks=dict(old_completed_graph_exact=False,qualified_gradient_graph_exact=False)
    if index<134:
        old=read(FIT/f'train_{index:03d}_result.json')
        assert projection(record,GRAPH_KEYS)==projection(old,GRAPH_KEYS),(index,'old_graph')
        assert contents(record['prediction'])==contents(old['prediction']),(index,'old_prediction')
        checks['old_completed_graph_exact']=True
    if index<8:
        old=read(QUAL/'reference'/f'train_{index:03d}_result.json')
        assert projection(record,GRAPH_KEYS+['accumulated_gradient_hashes'])==projection(old,GRAPH_KEYS+['accumulated_gradient_hashes']),(index,'qualified_graph')
        assert contents(record['prediction'])==contents(old['prediction']),(index,'qualified_prediction')
        checks['qualified_gradient_graph_exact']=True
    return checks
def prefix_update(record):
    number=record['update'];checks=dict(old_completed_update_exact=False,qualified_state_boundary_exact=False)
    if number<=33:
        old=read(FIT/f'update_{number:02d}.json')
        assert projection(record,UPDATE_KEYS)==projection(old,UPDATE_KEYS),(number,'old_update')
        checks['old_completed_update_exact']=True
    if number<=2:
        old=read(QUAL/'reference'/f'update_{number:02d}.json')
        assert projection(record,UPDATE_KEYS+['clipped_gradient_hashes'])==projection(old,UPDATE_KEYS+['clipped_gradient_hashes']),(number,'qualified_update')
        check(record['checkpoint']);check(old['checkpoint'])
        current_state=read(record['checkpoint']['path']);qualified_state=read(old['checkpoint']['path'])
        check(current_state['payload']);check(qualified_state['payload'])
        assert projection(current_state,CHECKPOINT_KEYS)==projection(qualified_state,CHECKPOINT_KEYS),(number,'qualified_boundary')
        checks['qualified_state_boundary_exact']=True
    return checks
