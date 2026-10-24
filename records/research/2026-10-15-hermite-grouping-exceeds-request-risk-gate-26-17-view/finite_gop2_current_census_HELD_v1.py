"""HELD fixed own20 for qualified current4e+GOP2.
Actual keys/PB coverage and finite all16 arrays only; no old-physics identity or price reuse.
PB649/source remain unchanged. No downstream action or retry.
"""
import argparse,ast,hashlib,json,math,os,re,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];P2=O.parent;R=O/'release'
Q=D.parent/'astra_gop2_current_delivery_v1'
PROV=Q/'candidate_current4e_GOP2_ON_HELD_v1.py'
PARENT=D.parent/'astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py'
PY=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
RT=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
N=D/'numerical_v1'
PARTS={'a':[651,378,424,328,287,641,548,557,316,390],'660':list(range(660,670))}
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');ENV.update({k:'1'for k in THREADS})
for flag in('PB_ALL','MODE','DRY','STATS'):ENV.pop(flag,None)
RETIRED=True
ACTIVE_PROTOCOL=D/'CENSUS_PROTOCOL_HELD_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stamp():
    return subprocess.check_output(['/bin/date', '-u', '+%FT%TZ'], text=True).strip()

def read(path):
    return json.loads(Path(path).read_text())

def exclusive(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')

def clean_guard(path, command=None):
    g = read(path)
    assert g['completed'] is True and g['exit_code'] == 0 and g.get('error') is None
    assert g['whole_group_retired'] is True and g['remaining_rows'] == []
    assert g['rss_backstop_bytes'] == g['footprint_trigger_bytes'] == 7500000000
    assert g['max_rss_bytes'] < 7500000000 and g['max_footprint_bytes'] < 7500000000
    if command is not None:
        assert g['command'] == command
    return g

def resources():
    text = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', text).group(1))
    rows = subprocess.check_output(['/bin/ps', '-axo', 'rss=,comm='], text=True, timeout=8)
    heavy = sum(int(s.split(None, 1)[0]) >= 1000000 and 'python' in s.lower() for s in rows.splitlines() if s.strip())
    return dict(utc=stamp(), memory_free_percent=free, heavy_python_ge1GB=heavy, counted=heavy + 1, truth_reserve=1)

def done(step):
    p = D / f'GOP2_STEP_{step}_DONE_v1.json'
    if not p.exists():
        return False
    r = read(p)
    for path, expected in r['output_sha256'].items():
        assert sha(path) == expected, ('completed output changed', path)
    return True

def finish(step, paths, details):
    exclusive(D / f'GOP2_STEP_{step}_DONE_v1.json', dict(utc=stamp(), details=details,
              output_sha256={str(p): sha(p) for p in paths}))

def admission_event(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')

def admit(step):
    # Keep the granted slot reserved; no native child exists during this wait.
    assert RETIRED is True and step in ['census_' + p for p in PARTS]
    started=time.monotonic();attempt=0;last_status=None
    while True:
        state=resources();elapsed=time.monotonic()-started;attempt+=1
        okay=state['memory_free_percent']>=60 and state['counted']<8 and elapsed<=1800
        record=dict(step=step,controller_pid=os.getpid(),attempt=attempt,
                    elapsed_seconds=elapsed,admitted=okay,**state)
        admission_event(N/'admission_wait.jsonl',record)
        if okay:
            admission_event(N/'admission.jsonl',record)
            return record
        if last_status is None or elapsed-last_status>=30:
            print(json.dumps(dict(status='waiting_for_prospective_headroom_no_native_child',
                  step=step,memory_free_percent=state['memory_free_percent'],counted=state['counted'],
                  elapsed_seconds=elapsed,slot_reserved=True)),flush=True)
            last_status=elapsed
        assert elapsed<1800, 'Bounded admission wait expired; no automatic retry'
        time.sleep(min(15,1800-elapsed))

def admissions(controller_pid=None):
    controller_pid=os.getpid() if controller_pid is None else controller_pid
    records=[json.loads(x)for x in (N/'admission.jsonl').read_text().splitlines()]
    assert [x['step']for x in records]==['census_'+p for p in PARTS]
    assert all(x['controller_pid']==controller_pid and x['memory_free_percent']>=60 and x['counted']<8 for x in records)
    attempts=[json.loads(x)for x in (N/'admission_wait.jsonl').read_text().splitlines()]
    assert [x for x in attempts if x['admitted']] == records
    assert all(x['admitted'] is True and 0<=x['elapsed_seconds']<=1800 for x in records)
    return dict(path=str(N/'admission.jsonl'),sha256=sha(N/'admission.jsonl'),records=records,
                wait_path=str(N/'admission_wait.jsonl'),wait_sha256=sha(N/'admission_wait.jsonl'))

def numerical(step, command, seconds, extra_env=None):
    global RETIRED
    pins()
    assert os.getpriority(os.PRIO_PROCESS, 0) == 0 and all(os.environ.get(k) == '1' for k in THREADS)
    guard, log = N / f'{step}_guard.json', N / f'{step}.log'
    started, owner = D / f'GOP2_STEP_{step}_STARTED_v1.json', D / f'GOP2_STEP_{step}_OWNER_v1.json'
    assert not any(p.exists() for p in [guard, log, started, owner]), 'unfinished step needs root review/new version'
    before = admit(step)
    pins()
    env = dict(ENV, **(extra_env or {}))
    args = [str(PY), '-B', str(R / 'guard_footprint_v3.py'), '--record', str(guard),
            '--seconds', str(seconds), '--footprint-limit', '7500000000', '--rss-backstop', '7500000000', *command]
    exclusive(started, dict(utc=stamp(), command=args, resources=before, source_sha256=sha(PROV),
                           controller_sha256=sha(__file__), env={k: env.get(k) for k in THREADS + ('PB_ALL', 'STATS')}))
    child, failure = None, None
    with log.open('x') as stream:
        try:
            child = subprocess.Popen(args, cwd=P2, env=env, stdout=stream, stderr=subprocess.STDOUT)
            RETIRED = False
            exclusive(owner, dict(utc=stamp(), controller_pid=os.getpid(), owned_guard_pid=child.pid, command=args,
                                  controller_nice=0, guard_nice=0, native_nice=10))
            while child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current = resources()
                    with (D / 'census_resources_v1.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=step, **current)) + '\n')
                    assert current['memory_free_percent'] >= 50 and current['counted'] <= 10, ('live resource bound', current)
        except BaseException as error:
            failure = error
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child is None:
                RETIRED = True
            else:
                try:
                    g = read(guard)
                    RETIRED = child.poll() is not None and g.get('whole_group_retired') is True and g.get('remaining_rows') == []
                except (OSError, ValueError):
                    RETIRED = False
            if failure is not None:
                exclusive(D / f'GOP2_STEP_{step}_INTERRUPTED_v1.json', dict(utc=stamp(), error=repr(failure),
                          owned_guard_pid=child.pid if child else None, owned_retirement_verified=RETIRED,
                          slot_reservation_retained=not RETIRED))
    if failure is not None:
        raise failure
    assert child.returncode == 0 and RETIRED
    clean_guard(guard, command)
    pins()
    return guard

def census(part):
    step, ranks = 'census_' + part, PARTS[part]
    out = N / (step + '.json')
    if done(step):
        return out
    assert not out.exists()
    command = [str(RT), '-B', str(O / 'engineer/quick.py'), str(PROV), ','.join(map(str, ranks)), str(out), '--keys']
    guard = numerical(step, command, 9120)
    d = read(out)
    assert Path(d['candidate']) == PROV and [r['rank'] for r in d['rows']] == ranks
    assert all('key_uses' in r and 'error' not in r and 'errors' not in r and type(r['flops'])is int and r['flops']>0 and type(r['ops'])is int and r['ops']>0 for r in d['rows'])
    import numpy as np
    predictions = [N / f'{step}_pred_{r}.npy' for r in ranks]
    for path in predictions:
        a = np.load(path, allow_pickle=False)
        assert a.shape == (16, 1024) and a.dtype == np.dtype('float32') and np.isfinite(a).all()
    finish(step, [out, guard, *predictions], dict(source_sha256=sha(PROV), ranks=ranks, all16_predictions_complete=True))
    return out

def asts(path):return [ast.dump(n,include_attributes=False)for n in ast.parse(Path(path).read_text()).body]

def pb(path):
    ns=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_PB_SINGLE'for t in n.targets)]
    return ast.literal_eval(ns[-1].value)

def metadata(path):return json.loads(Path(path).read_text(),parse_float=lambda _:None)


def source_context():
 manifest=read(Q/'SOURCE_MANIFEST_HELD_v1.json')
 assert sha(PARENT)==manifest['parent_sha256']=='4e1bf080bb6751b487c62842e369a427633e0c4686f089fd3f8e6448bb416f0e'
 assert sha(PROV)==manifest['files'][str(PROV)]=='0170697df8866cff2bc43b13fa7a41faf5a8d73a9c527985e12ac8a29a4c89e4'
 assert PROV.read_bytes()==PARENT.read_bytes()+b'\n\n'+Path(manifest['fragment_path']).read_bytes()
 assert sha(manifest['fragment_path'])==manifest['fragment_sha256']
 assert pb(PROV)==pb(PARENT)and len(pb(PROV))==649
 assert manifest['dense_context_unchanged']is True and manifest['identity_assertions_preserved']is True
 assert manifest['FHM_dense_GTR_IOP_OFF_preserved']is True
 return dict(candidate_sha256=sha(PROV),parent_sha256=sha(PARENT),PB_order_sha256=hashlib.sha256(repr(pb(PROV)).encode()).hexdigest())

def pins():
 c=read(ACTIVE_PROTOCOL)
 assert c['purpose']=='GOP2_current4e_actual_own20_preservedPB_v1'and c['parts']==PARTS
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 for p,h in c['source_pins'].items():assert sha(p)==h,('changed pin',p)
 assert sha(PROV)==c['candidate_sha256']and sha(RT)==c['current_runtime_binary_sha256']
 assert c['historical_predictions_keys_bills_timings_reused']is False
 assert c['old_physics_prediction_equality_gate']is False
 source_context();return c

def qualification(g):
 rp=Path(g['qualification_root_acceptance_path']).resolve();qp=Path(g['qualification_protocol_path']).resolve()
 assert rp.parent==Q and qp.parent==Q
 assert sha(rp)==g['qualification_root_acceptance_sha256']and sha(qp)==g['qualification_protocol_sha256']
 a=read(rp)
 assert a['approved']is True and a['reviewed_by']=='root'
 assert a['actual_managed_exit_observed']is True and a['exit_code']==0
 assert a['all_owned_PID_PPID_PGID_absent']is True and a['all_fixed_qualification_checks_recomputed']is True
 assert a['candidate_sha256']==sha(PROV)and a['protocol_sha256']==sha(qp)
 assert a['complete_calls']==2 and a['all16_finite']is True and a['first_layer_exact']is True
 assert a['fifteen_routes_and_cleanup']is True and a['setup_identical']is True
 q=read(qp);assert q['candidates']['ON']['sha256']==sha(PROV)
 assert q['purpose']=='gop2_current4e_qualification2'and q['selection']=='none_fixed_implementation_and_bill_only'
 for p,h in q['source_pins'].items():assert sha(p)==h,('qualification source drift',p)
 return dict(root_path=str(rp),root_sha256=sha(rp),protocol_path=str(qp),protocol_sha256=sha(qp),candidate_sha256=sha(PROV))

def grant(path):
 global ACTIVE_PROTOCOL
 g=read(path);assert Path(path).resolve().parent==D
 assert g['approved']is True and g['reviewed_by']=='root'and g['purpose']=='GOP2_current4e_actual_own20_preservedPB_v1'
 ACTIVE_PROTOCOL=Path(g['protocol_path']).resolve();assert ACTIVE_PROTOCOL.parent==D
 assert g['controller_sha256']==sha(__file__)and g['protocol_sha256']==sha(ACTIVE_PROTOCOL)
 c=pins();assert c['status']=='REGISTERED'
 q=qualification(g)
 assert g['numerical_slots']==1 and g['candidate_sha256']==sha(PROV)
 assert g['preserve_PB_order']is True and g['no_followup_authorization']is True
 assert g['fresh_all20']is True and g['historical_predictions_keys_bills_timings_reused']is False
 assert g['old_physics_prediction_equality_gate']is False
 assert type(g['logical_reserved_after_launch'])is int and 1<=g['logical_reserved_after_launch']<=8
 assert all(os.environ.get(k)=='1'for k in THREADS)and all(os.environ.get(k)is None for k in('PB_ALL','MODE','DRY','STATS'))
 assert os.getpriority(os.PRIO_PROCESS,0)==0
 return c,g,q

def census_report(paths):
 import numpy as np
 import collections
 assert len(paths)==len(PARTS)
 reports=[];allkeys=collections.Counter();ctx=source_context()
 for part,path in zip(PARTS,paths):
  ranks=PARTS[part];a=metadata(path)
  assert Path(path)==N/f'census_{part}.json'and a['candidate']==str(PROV)
  assert [r['rank']for r in a['rows']]==ranks
  for row in a['rows']:
   assert 'error'not in row and 'errors'not in row
   rank=row['rank'];keys=row['key_uses']
   assert keys and all(len(k)==5 and all(type(v)is int for v in k)and k[2]>0 and k[3]>0 and k[4]>0 for k in keys)
   assert len({tuple(k[:4])for k in keys})==len(keys)
   for k in keys:allkeys[tuple(k[:4])]+=k[4]
   pred=Path(path).with_name(Path(path).stem+'_pred_'+str(rank)+'.npy')
   x=np.load(pred,allow_pickle=False)
   assert x.shape==(16,1024)and x.dtype==np.dtype('float32')and np.isfinite(x).all()
   assert hashlib.sha256(x.tobytes()).hexdigest()[:16]==row['pred_sha']
   reports.append(dict(rank=rank,all16_complete_finite=True,prediction_sha256=sha(pred),key_uses_sha256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),distinct_keys=len(keys),key_requests=sum(k[4]for k in keys)))
 assert len(reports)==20 and len({r['rank']for r in reports})==20
 listed=set(pb(PROV));observed=set(allkeys)
 return dict(**ctx,complete_networks=20,reports=reports,PB_order_preserved=True,PB_single_count=649,distinct_observed_keys=len(allkeys),observed_key_uses=[list(k)+[allkeys[k]]for k in sorted(allkeys)],ordinary_lazy_keys=[list(k)for k in sorted(observed-listed)],listed_keys_observed=len(listed&observed),listed_keys_unobserved=[list(k)for k in sorted(listed-observed)],MSE_fields_inspected=False,historical_predictions_keys_bills_timings_reused=False,old_physics_prediction_equality_gate=False,production_eligibility_not_inferred_from_coverage=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['numeric']);p.add_argument('root_grant',type=Path);a=p.parse_args()
 c,g,q=grant(a.root_grant);assert not N.exists()and not(D/'CENSUS_COMPLETE_HELD_v1.json').exists()
 lock=D/'OWNER_LOCK_v1';lock.mkdir();N.mkdir()
 exclusive(D/'CENSUS_OWNER_v1.json',dict(utc=stamp(),controller_pid=os.getpid(),controller_sha256=sha(__file__),root_grant_sha256=sha(a.root_grant),candidate_sha256=sha(PROV)))
 error=None
 def interrupted(signum,frame):raise KeyboardInterrupt('Owned GOP2 census signal%s'%signum)
 previous=signal.signal(signal.SIGTERM,interrupted)
 try:
  files=[census(part)for part in PARTS]
  expected=[N/f'census_{part}_guard.json'for part in PARTS];assert sorted(N.glob('*_guard.json'))==sorted(expected)
  admission_evidence=admissions();report=census_report(files);pins();qualification(g)
  exclusive(D/'CENSUS_COMPLETE_HELD_v1.json',dict(utc=stamp(),status='COMBINED_OWN20_ACTUAL_KEYS_COMPLETE_PB_RETAINED_HELD',controller_pid=os.getpid(),candidate=str(PROV),candidate_sha256=sha(PROV),controller_sha256=sha(__file__),protocol_sha256=sha(ACTIVE_PROTOCOL),root_grant_sha256=sha(a.root_grant),qualification=q,actual_coverage=report,admission_evidence=admission_evidence,expected_guards=[str(p)for p in expected],done_receipts={str(D/f'GOP2_STEP_census_{p}_DONE_v1.json'):sha(D/f'GOP2_STEP_census_{p}_DONE_v1.json')for p in PARTS},all_owned_guards_retired=RETIRED,managed_controller_reaping_by_root_still_required=True,no_followup_authorization=True))
 except BaseException as e:
  error=e;exclusive(D/'CENSUS_STOPPED_v1.json',dict(utc=stamp(),error=repr(e),all_owned_guards_retired=RETIRED,slot_reservation_retained=not RETIRED,no_retry=True));raise
 finally:
  exclusive(D/'CENSUS_OWNER_RETIREMENT_v1.json',dict(utc=stamp(),controller_pid=os.getpid(),error=repr(error)if error else None,all_owned_guards_retired=RETIRED,slot_reservation_retained=not RETIRED))
  if RETIRED:lock.rmdir()
  signal.signal(signal.SIGTERM,previous)
if __name__=='__main__':main()

