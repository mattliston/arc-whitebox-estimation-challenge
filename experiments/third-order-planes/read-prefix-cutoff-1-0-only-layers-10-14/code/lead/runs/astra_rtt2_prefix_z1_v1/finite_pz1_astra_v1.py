"""Fixed PZ1 ON-only pilot/confirmation; reuse exact QR/e406 controls."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
QP=O/'lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py'
assert sha(QP)=='8bd0cb93f83ddb22223b8b8f1e8f1017dcb458661a0934637107437a732c15ae'
s=importlib.util.spec_from_file_location('_pz1_owned',QP);smoke=importlib.util.module_from_spec(s);s.loader.exec_module(smoke)
q=smoke.q;q.guard.JUDGE=D/'native_pz1_judge_astra_v1.py'
STATS=q.STATS+('_YR7_STATS','_YR7_ACTIVE','_YR7_BIND_STATS','_RTT_STATS','_RTT_ACTIVE','_RTT_CONTEXT')
def run(phase,shard):
    p=D/'PROTOCOL_REGISTERED_astra_v1.json';c=json.loads(p.read_text());digest=sha(p)
    assert c['approved_by']=='root' and c['purpose']=='pz1_qr_e406_pilot8_fresh16_oos8'
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert phase in ('pilot','remaining') and shard in range(4)
    if phase=='remaining':
        r=json.loads((D/'PILOT_DECISION_astra_v1.json').read_text());assert r['passed'] is True and r['all_actual_retired'] is True
    assert all(os.environ.get(k)=='1' for k in q.guard.THREADS)
    proto=dict(pinned_files=c['source_pins'],candidates=c['candidates'])
    q.guard.pins(proto);seeds=c['phase_seeds'][phase][shard::4]
    h=D/f'run_pz1_{phase}_s{shard}_v1';h.mkdir(exist_ok=False)
    q.write(h/'PLAN_v1.json',dict(phase=phase,shard=shard,controller_pid=os.getpid(),seeds=seeds,protocol_sha256=digest))
    evaluations=[]
    try:
        for seed in seeds:
            q.admit(h,'ON',seed)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),GOP2_MODE='1',GOP2_SCIENCE_PROTOCOL_PATH=str(p),GOP2_SCIENCE_PROTOCOL_SHA256=digest,PYTHONDONTWRITEBYTECODE='1')
            item=q.guard.run_native(h,proto,'ON',seed,env)
            a=q.current(c,item,seed);st=a['row']['stats']
            assert st['_RTT_ACTIVE'] is False and st['_RTT_CONTEXT']==[None]
            assert st['_YR7_ACTIVE']==[False] and st['_YR7_BIND_STATS']==dict(keys=17,retired=True)
            assert st['_YR7_STATS']['retired'] and st['_YR7_STATS']['value_cache_entries']==0
            evaluations.append(dict(role='ON',seed=seed,**item))
            print(json.dumps(dict(phase=phase,shard=shard,completed=len(evaluations),total=len(seeds))),flush=True)
        q.write(h/'READY_v1.json',dict(controller_pid=os.getpid(),protocol_sha256=digest,evaluations=evaluations,all_guards_retired=q.guard.OWNED_RETIREMENT_CONFIRMED,means_not_read=True))
    except BaseException as e:
        q.write(h/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,owned_retirement=q.guard.OWNED_RETIREMENT_CONFIRMED));raise
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
