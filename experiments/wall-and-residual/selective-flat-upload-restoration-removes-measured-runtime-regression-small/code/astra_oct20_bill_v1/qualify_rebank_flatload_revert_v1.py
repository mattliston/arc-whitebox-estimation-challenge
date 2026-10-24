"""Lead-owned bank retry qualification. Original guards/gates; no fire."""
import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

D=Path(__file__).resolve().parent
RUNS=D.parent
K=RUNS/'fable_oct19_kc_v1'
A=K/'agent_astra_joint_v1'
O=RUNS.parent.parent
ROOT=next(p for p in D.parents if (p/'.venv/bin/python').exists())
PY=ROOT/'.venv/bin/python'
SCI=D/'candidate_eh14_flatload_extra_revert_sci_v1.py'
REL=D/'candidate_eh14_flatload_extra_revert_rel_v1.py'
SCI_SHA='9af5617cf1a8be8e8a6d0734c5779d277a0febc1f0c6a73d777bd204d2318e44'
REL_SHA='c7b39747c2dcaaa3dba6575bab963075416ea7a9cd0aef435ef05b25bbd39a80'
JUDGE=A/'run_current32_parallel5_v2.py'
JUDGE_SHA='096ec6c5670c3303ccc8111bee2b274f7e82bb5293a27a2d6128ba038228d993'
ISOLATE=A/'qualify_runtime_companion_isolated_v3.py'
HELPER=A/'prepare_runtime_pair_parity_v3.py'
BANK=K/'candidate_rel_gram_pair_kclm_copy_astra.py'
BANK_SHA='fd8739582ffdd04decc5fe6bbb2f99fb0b3dc02a2ee786d684bd21bbf52d974d'
LAST=O/'release/runs/astra_micro_copyall_ecut_nogov_4b8baf51_v1/submission.tar.gz'
LAST_SHA='faf49095644316e9a5738fecd39cdce33d85b2e03287b9cfb5238e01ef16afa9'
GATE=RUNS/'fable_oct19_l13rank_v1/release_min_v1.sh'
SCI_OUT=D/'science32_eh14_flatload_revert_parallel5_v1'
OUT=D/'rebank_flatload_revert_v1'
TAG='astra_eh14_flatload_revert_c7b39747_bank_v1'
QUAL=D/'qual_eh14_flatload_revert_c7b39747_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
stamp=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
rows=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def main():
    assert sys.argv[1:]==['--run']
    fixed={SCI:SCI_SHA,REL:REL_SHA,JUDGE:JUDGE_SHA,BANK:BANK_SHA,LAST:LAST_SHA,
           HELPER:'f0292b0045e5e21f661265e4807199fc437908e025706c5dc792d363a5ffa1b6'}
    for p,h in fixed.items():assert sha(p)==h,(str(p),'pin')
    fixed.update({p:sha(p) for p in (ISOLATE,GATE,O/'release/stage_a_package.py',O/'fire/prefire_check_v1.py',Path(__file__))})
    for p in (OUT,SCI_OUT,QUAL,O/'release/runs'/TAG):assert not p.exists(),str(p)
    isolation=module(ISOLATE,'rebank_isolation')
    helper=module(HELPER,'rebank_validation')
    initial=isolation.processes()
    assert 77740 in initial and 'run_queue_astra_v6.sh' in initial[77740]['command']
    isolation.deny_external_heavy(initial,isolation.descendants(initial,77740))
    OUT.mkdir()
    rec=dict(utc=stamp(),status='STARTED',authorization='Matt authorized one bank resubmission before 5pm Pacific Oct20, then next must project below J2W.',source_pins={str(p):h for p,h in fixed.items()},runs=[],no_auto_fire=True)
    paused={};active=None;status=1
    def save():
        (OUT/'QUALIFICATION_v1.json').write_text(json.dumps(rec,indent=2)+'\n')
    def interrupt(sig,_):raise KeyboardInterrupt('signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def run(command,log,timeout):
        nonlocal active
        with log.open('x') as f:
            active=subprocess.Popen([str(x) for x in command],cwd=O,stdout=f,stderr=subprocess.STDOUT)
            try:code=active.wait(timeout=timeout)
            finally:
                if active.poll() is None:
                    owned=isolation.descendants(isolation.processes(),active.pid)
                    active.send_signal(signal.SIGINT)
                    for pid in owned-{active.pid}:
                        try:os.kill(pid,signal.SIGINT)
                        except ProcessLookupError:pass
                    active.wait(timeout=30)
                active=None
        rec['runs'].append(dict(command=[str(x) for x in command],exit_code=code))
        save()
        assert code==0,(str(log),code)
    try:
        os.kill(77740,signal.SIGSTOP);paused[77740]=initial[77740]
        table=isolation.processes()
        for pid,row in table.items():
            if row['ppid']==77740 and 'run_current32_' in row['command'] and '_fable_v1.py ' in row['command']:
                os.kill(pid,signal.SIGSTOP);paused[pid]=row
        rec['paused']=paused;save();print(stamp(),'DRAINING',len(paused),flush=True)
        deadline=time.monotonic()+210
        while True:
            table=isolation.processes();tree=isolation.descendants(table,77740)
            isolation.deny_external_heavy(table,tree)
            remaining={pid for pid in tree-set(paused) if not table[pid]['state'].startswith('Z')}
            if not remaining:break
            assert time.monotonic()<deadline,remaining
            time.sleep(2)
        assert isolation.pressure()>=60
        rec['drained_utc']=stamp();save()
        print(stamp(),'ORIGINAL_FULL32_START',flush=True)
        run([PY,'-B',JUDGE,SCI,SCI_SHA,SCI_OUT,'--run'],OUT/'full32.log',1800)
        verdict=read(SCI_OUT/'VS_BEST_GRADED_v1.json')
        assert verdict['reference_submission_id']==337393
        assert verdict['all32']['adjusted_pct']<0 and verdict['heldout25']['adjusted_pct']<0
        rec['science32']=verdict
        parent=K/'science32_micro_copyall_ecut_eh14_fg_astra_m15'
        exact=[]
        for pred in (SCI_OUT/'candidate_all32').glob('pred_*.bin'):
            other=parent/'candidate_all32'/pred.name
            assert pred.read_bytes()==other.read_bytes(),('parent byte mismatch',pred.name)
            exact.append(pred.name)
        assert len(exact)==32
        rec['all32_byte_exact_to_eh14_fg_parent']=True;save()
        for label,source,digest in [('bank',BANK,BANK_SHA),('candidate',REL,REL_SHA)]:
            dest=OUT/label;dest.mkdir()
            assert isolation.pressure()>=60
            print(stamp(),'QUIET_BOUNDARY_START',label,flush=True)
            run([PY,'-B',helper.GUARD,'--record',dest/'GUARD_v1.json','--seconds','125',
                 '--footprint-limit','7500000000','--rss-backstop','7500000000',
                 PY,'-B',helper.BOUNDARY,source,'35,150,200',dest,'--short-setup'],dest/'run.log',150)
            helper.valid_guard(dest/'GUARD_v1.json',original=True)
            assert read(dest/'manifest.json')['sha256']==digest
            rr=rows(dest/'rows.jsonl');assert [r['rank'] for r in rr]==[35,150,200]
            reference=SCI_OUT if label=='candidate' else Path(read(K/'BEST_GRADED_REFERENCE_v1.json')['science_run'])
            for row in rr:
                rank=row['rank'];raw=helper.valid_row(row,dest/f'pred_{rank}.bin')
                original=reference/'candidate_part01'/f'pred_{rank}.bin'
                assert raw==original.read_bytes(),('production/science bytes',label,rank)
                sr=next(x for x in rows(reference/'candidate_part01/rows.jsonl') if x['rank']==rank)
                assert row['flops']==sr['flops'],('production/science tariff',label,rank,row['flops'],sr['flops'])
            rec[label+'_production_science_bytes_tariff_pass']=True;save()
        print(stamp(),'ORIGINAL_RELEASE_GATES_START',flush=True)
        for p,h in fixed.items():assert sha(p)==h
        run(['bash',GATE,REL,TAG,QUAL,LAST],OUT/'release_min.log',240)
        archive=O/'release/runs'/TAG/'submission.tar.gz'
        rec.update(status='FULL32_EXACT_PARENT_AND_QUIET_PRODUCTION_AND_ORIGINAL_GATES_PASS',
                   completed_utc=stamp(),release_tag=TAG,archive=str(archive),archive_sha256=sha(archive),
                   no_runtime_guarantee=True,submission_authorized=False)
        status=0
    except BaseException as exc:
        rec.update(status='STOPPED',error=repr(exc),stopped_utc=stamp())
        raise
    finally:
        table=isolation.processes();resumed=[]
        # A failed controller must not resume research over surviving native work.
        isolation.deny_external_heavy(table,isolation.descendants(table,77740))
        for pid,old in reversed(list(paused.items())):
            now=table.get(pid)
            if now and now['started']==old['started'] and now['command']==old['command']:
                os.kill(pid,signal.SIGCONT);resumed.append(pid)
        rec['resumed']=resumed;rec['resumed_utc']=stamp();save()
        print(stamp(),rec['status'],'RESUMED',resumed,flush=True)
    return status

if __name__=='__main__':
    sys.exit(main())

