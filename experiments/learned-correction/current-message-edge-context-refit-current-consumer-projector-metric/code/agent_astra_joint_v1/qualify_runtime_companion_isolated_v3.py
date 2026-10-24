"""Generic quiet three-rank parity + original release gates. No auto-fire.

Usage: python this.py RT_TAG --check    # read-only preparation validation
       python this.py RT_TAG --run      # lead schedules this explicitly
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

A=Path(__file__).resolve().parent; K=A.parent; R=K.parent; O=R.parent.parent
PY=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
ROOT_PID=77740
PARITY=A/'prepare_runtime_pair_parity_v3.py'
PARITY_SHA='f0292b0045e5e21f661265e4807199fc437908e025706c5dc792d363a5ffa1b6'
BANK=K/'candidate_rel_gram_pair_kclm_copy_astra.py'
BANK_SHA='fd8739582ffdd04decc5fe6bbb2f99fb0b3dc02a2ee786d684bd21bbf52d974d'
LAST_ARCHIVE=O/'release/runs/astra_micro_kclm_copyall_4a40feb6_v1/submission.tar.gz'
LAST_ARCHIVE_SHA='73e3d0af6c898dd00bcfc33813d9578da6a7ee7abb30fe7659536c6d44bd23a9'
RELEASE_MIN=R/'fable_oct19_l13rank_v1/release_min_v1.sh'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
stamp=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def processes():
    result={}
    for line in subprocess.check_output(['ps','-axo','pid=,ppid=,lstart=,stat=,command='],text=True).splitlines():
        p=line.split(None,8)
        if len(p)==9:result[int(p[0])]=dict(ppid=int(p[1]),started=' '.join(p[2:7]),state=p[7],command=p[8])
    return result

def descendants(table,parent):
    found={parent}
    while True:
        more={p for p,r in table.items() if r['ppid'] in found}
        if not more-found:return found
        found|=more

def pressure():
    value=subprocess.check_output(['memory_pressure'],text=True)
    return int(value.rsplit('System-wide memory free percentage:',1)[1].strip().strip('%'))

def deny_external_heavy(table,primary_tree):
    denied=[]
    for pid,r in table.items():
        c=r['command']
        project_python=c.startswith(str(PY)) or c.startswith(str(R/'astra_crunch_runtime_v1/failure336631_v1'))
        if pid in primary_tree or pid==os.getpid() or not project_python or r.get('state','').startswith('Z'):continue
        # Do not let a waiting auxiliary/native controller launch into quiet time.
        auxiliary='run_current32_aux30_' in c or 'guard_memory30_aux_' in c
        native='eval_fresh_arm_' in c or bool(re.search(r'/(?:[^ /]*fresh[^ /]*|gate_coadapt[^ /]*)/[^ ]*\.py(?: |$)',c))
        guarded='guard_memory' in c or 'guard_footprint' in c
        boundary='boundary_run' in c or 'run_boundary_census' in c or 'boundary_client' in c or '/server_light.py ' in c
        if auxiliary or native or guarded or boundary:denied.append(dict(pid=pid,**r))
    assert not denied, ('External auxiliary/native work must finish or be retired by lead before quiet qualification.',denied)

def read_science(path,helper):
    result=read(path/'RESULT32_v1.json')
    assert result['all_groups_retired'] and result['all32']['n']==32
    grade={r['rank']:r for r in read(path/'candidate_all32/GRADE_STORE_v1.json')}
    data={};cold=set();manifest_hashes={}
    for part in sorted(path.glob('candidate_part*')):
        f=part/'rows.jsonl'
        if not f.exists():continue
        rr=rows(f);m=read(part/'manifest.json')
        assert m['sha256']==result['candidate_sha256'] and sha(m['candidate'])==m['sha256']
        helper.valid_guard(part/'GUARD_v1.json')
        assert [r['rank'] for r in rr]==m['ranks']
        cold.add(rr[0]['rank']);manifest_hashes[str(part/'manifest.json')]=sha(part/'manifest.json')
        for r in rr:
            assert r['rank'] not in data
            helper.valid_row(r,part/f"pred_{r['rank']}.bin")
            data[r['rank']]=r
    assert set(data)==set(grade) and len(data)==32 and len(cold)==12
    return result,grade,data,cold,manifest_hashes

def summary(values):
    x=[r['adjusted_pct'] for r in values]
    return dict(n=len(x),adjusted_pct=statistics.mean(x),adjusted_se=statistics.stdev(x)/math.sqrt(len(x)),
                lower=sum(v<0 for v in x),raw_pct=statistics.mean(r['raw_pct'] for r in values),
                bill_pct=statistics.mean(r['bill_pct'] for r in values))

def transfer(plan,helper,paired,parity_result):
    parent=Path(plan['variant']['physics_science_run'])
    actual,g,parent_rows,cold,manifests=read_science(parent,helper)
    assert parity_result['status']=='WHOLE_THREE_RANK_PARITY_AND_TARIFF_PASS'
    fg_extra=401408 if 'features grouped' in plan['exact_additions'] else 0
    charges={rank:fg_extra+114688+(8 if rank in cold else 0) for rank in parent_rows}
    references={'best337393':Path(read(K/'BEST_GRADED_REFERENCE_v1.json')['science_run']),
                'frozen_v4a':K/'science32_rel_v4a_m15'}
    compared={}
    for label,path in references.items():
        _,rg,rr,_,_=read_science(path,helper)
        output=[]
        for rank in sorted(g):
            raw=g[rank]['u1']/rg[rank]['u1'];bill=(parent_rows[rank]['flops']+charges[rank])/rr[rank]['flops']
            output.append(dict(rank=rank,raw_pct=100*(raw-1),bill_pct=100*(bill-1),adjusted_pct=100*(raw*bill-1)))
        seven={35,150,200,250,259,277,287}
        compared[label]=dict(all32=summary(output),held25=summary([r for r in output if r['rank'] not in seven]),rows=output)
    assert compared['best337393']['all32']['adjusted_pct']<0 and compared['best337393']['held25']['adjusted_pct']<0
    assert compared['frozen_v4a']['all32']['adjusted_pct']<=-.35 and compared['frozen_v4a']['held25']['adjusted_pct']<0
    return dict(utc=stamp(),status='EXACT_TRANSFORM_32_TRANSFER_SUPPORTED_BY_SOURCE_TINY_AND_WHOLE3_PARITY',
       interpretation='Parent32 was actually measured. Decorated32 was NOT rerun; its predictions transfer through exact-transform proof plus whole3 parity, with explicit deterministic bill additions.',
       parent_actual32_run=str(parent),parent_actual32_sha256=actual['candidate_sha256'],parent_result_sha256=sha(parent/'RESULT32_v1.json'),
       parent_actual_comparison=paired,source_manifest_hashes=manifests,variant=plan['variant'],
       full3_parity=parity_result,features_addition_per_prediction=fg_extra,pair_warm_addition=114688,
       pair_first_of_group_addition=114696,cold_first_ranks=sorted(cold),sm_flop_addition=0,
       per_rank_added_flops=charges,transferred_comparisons=compared,
       no_auto_fire=True,no_new32_measurement=True,no_runtime_or_failure_guarantee=True)

def validate_reused_bank(path,helper,plan):
    path=Path(path).resolve()
    guard=helper.valid_guard(path/'GUARD_v1.json',original=True)
    m,l=read(path/'manifest.json'),read(path/'launcher.json')
    assert m['sha256']==l['sha256']==BANK_SHA
    assert m['ranks']==[35,150,200] and m['client_flopscope']=='0.12.3'
    assert m['diagnostic_harness_sha256']==plan['pins'][str(helper.B/'boundary_client_dual_v3.py')]
    assert m['thread_helper_sha256']==plan['pins'][str(helper.B/'thread_counts_v3.py')]
    assert l['client_exit']==l['server_exit']==0 and l['handle_cap']==100000
    rr=rows(path/'rows.jsonl');assert [r['rank'] for r in rr]==[35,150,200]
    science=Path(read(K/'BEST_GRADED_REFERENCE_v1.json')['science_run'])
    for row in rr:
        rank=row['rank'];raw=helper.valid_row(row,path/f'pred_{rank}.bin')
        references=list(science.glob(f'candidate_part*/pred_{rank}.bin'))
        assert len(references)==1 and raw==references[0].read_bytes()
    files=[path/name for name in ('GUARD_v1.json','manifest.json','launcher.json','rows.jsonl','pred_35.bin','pred_150.bin','pred_200.bin')]
    return dict(path=str(path),pins={str(p):sha(p) for p in files},guard_elapsed_s=guard['elapsed_s'],
                bank_whole3_prediction_bytes_match_own_science=True,
                timing_caveat='Previously completed quiet bank reused; candidate is measured later. No contemporaneous timing guarantee.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('tag')
    parser.add_argument('--reuse-bank',type=Path,help='verified complete quiet bank35,150,200 directory; read-only reuse')
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--check',action='store_true');modes.add_argument('--run',action='store_true')
    args=parser.parse_args()
    assert sha(PARITY)==PARITY_SHA and sha(BANK)==BANK_SHA and sha(LAST_ARCHIVE)==LAST_ARCHIVE_SHA
    helper=module(PARITY,'runtime_parity_v3')
    out=A/'runtime_pair_sm_parity_v3'/args.tag;manifest=out/'MANIFEST_v1.json';plan=read(manifest);v=plan['variant']
    assert args.tag==plan['tag']
    for p,h in plan['pins'].items():assert sha(p)==h,('pin changed',p)
    for name in ('science','release'):assert sha(v[name])==v[name+'_sha256']
    sys.path.insert(0,str(K));import compare_best_v1
    paired=compare_best_v1.report(Path(v['physics_science_run']))
    assert paired['reference_submission_id']==337393
    assert paired['all32']['adjusted_pct']<0 and paired['heldout25']['adjusted_pct']<0
    old=paired['frozen_v4a_qualification']
    assert old['science_bar_cleared'] and old['all32']['adjusted_pct']<=-.35 and old['heldout25']['adjusted_pct']<0
    read_science(Path(v['physics_science_run']),helper)
    release_tag='astra_'+args.tag.removesuffix('_astra')+'_'+v['release_sha256'][:8]+'_v1'
    q=K/f"qual_{args.tag}_{v['release_sha256'][:8]}_v1"
    record=out/'QUALIFIER_v1.json';candidate=Path(plan['output']);bank=args.reuse_bank.resolve() if args.reuse_bank else out/'bank'
    reused_bank=validate_reused_bank(bank,helper,plan) if args.reuse_bank else None
    assert candidate.is_dir() and not any(candidate.iterdir())
    assert not record.exists() and (reused_bank is not None or not bank.exists()) and not q.exists() and not (O/'release/runs'/release_tag).exists()
    gate_pins={str(p):sha(p) for p in [RELEASE_MIN,O/'release/stage_a_package.py',O/'fire/prefire_check_v1.py',LAST_ARCHIVE]}
    if args.check:
        print(json.dumps(dict(status='READ_ONLY_PREPARATION_PASS_NO_NATIVE',tag=args.tag,release_tag=release_tag,
                              parent32=paired['all32'],held25=paired['heldout25'],qualification=str(q),manifest=str(manifest),reused_bank=reused_bank),indent=2));return 0
    initial=processes();assert ROOT_PID in initial and 'run_queue_astra_v6.sh' in initial[ROOT_PID]['command']
    deny_external_heavy(initial,descendants(initial,ROOT_PID))
    if reused_bank is None:bank.mkdir()
    rec=dict(start_utc=stamp(),tag=args.tag,manifest=str(manifest),manifest_sha256=sha(manifest),helper_sha256=sha(__file__),
       parity_helper_sha256=sha(PARITY),gate_pins=gate_pins,science=paired,reference_submission=337393,
       sources=[dict(label='bank',path=str(BANK),sha256=BANK_SHA),dict(label='candidate',path=v['release'],sha256=v['release_sha256'])],
       scope='Candidate ranks35,150,200 under original guard, validated quiet bank (fresh or explicitly reused), whole3 parity then original release_min. No submission.',
       reused_bank=reused_bank,
       timing_caveat='Research dispatchers are paused only between guarded groups. Worker/server jitter remains; no runtime guarantee.',
       runs=[],no_capacity_or_queue_edit=True)
    paused={};active=None;began=time.monotonic();exitcode=1
    def save():record.write_text(json.dumps(rec,indent=2)+'\n')
    def interrupted(signum,frame):raise KeyboardInterrupt('signal %s'%signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def run(command,log,timeout):
        nonlocal active
        with log.open('x') as stream:
            active=subprocess.Popen(command,cwd=O,stdout=stream,stderr=subprocess.STDOUT)
            try:return active.wait(timeout=timeout)
            finally:
                if active.poll() is None:
                    active.send_signal(signal.SIGINT);active.wait(timeout=30)
                active=None
    try:
        os.kill(ROOT_PID,signal.SIGSTOP);paused[ROOT_PID]=initial[ROOT_PID]
        table=processes()
        judges=[pid for pid,r in table.items() if r['ppid']==ROOT_PID and 'run_current32_' in r['command'] and '_fable_v1.py ' in r['command']]
        for pid in judges:os.kill(pid,signal.SIGSTOP);paused[pid]=table[pid]
        rec['paused']=paused;save();print(stamp(),'DRAINING_ACTIVE_GROUPS',len(judges),flush=True)
        deadline=time.monotonic()+180
        while True:
            table=processes();tree=descendants(table,ROOT_PID)
            deny_external_heavy(table,tree)
            active_descendants={pid for pid in tree-set(paused) if not table[pid].get('state','').startswith('Z')}
            if not active_descendants:break
            assert time.monotonic()<deadline,active_descendants
            time.sleep(2)
        rec.update(drained_utc=stamp(),free_pct_before=pressure());save()
        assert rec['free_pct_before']>=60,rec['free_pct_before']
        run_specs=[('candidate',Path(v['release']),v['release_sha256'],candidate)]
        if reused_bank is None:run_specs.insert(0,('bank',BANK,BANK_SHA,bank))
        else:
            for p,h in reused_bank['pins'].items():assert sha(p)==h
            rec['bank_whole3_prediction_bytes_match_own_science']=True
        for label,source,digest,destination in run_specs:
            assert sha(source)==digest
            table=processes();deny_external_heavy(table,descendants(table,ROOT_PID))
            command=[str(PY),'-B',str(helper.GUARD),'--record',str(destination/'GUARD_v1.json'),'--seconds','125',
                     '--footprint-limit','7500000000','--rss-backstop','7500000000',str(PY),'-B',str(helper.BOUNDARY),
                     str(source),'35,150,200',str(destination),'--short-setup']
            print(stamp(),'BOUNDARY_START',label,flush=True)
            status=run(command,destination/'run.log',140)
            rec['runs'].append(dict(label=label,command=command,status=status));save();assert status==0,(label,status)
            guard=helper.valid_guard(destination/'GUARD_v1.json',original=True)
            rr=rows(destination/'rows.jsonl');assert [r['rank'] for r in rr]==[35,150,200]
            for r in rr:helper.valid_row(r,destination/f"pred_{r['rank']}.bin")
            assert read(destination/'manifest.json')['sha256']==digest
            if label=='bank':
                science_bank=Path(read(K/'BEST_GRADED_REFERENCE_v1.json')['science_run'])
                for r in rr:
                    matches=list(science_bank.glob(f"candidate_part*/pred_{r['rank']}.bin"))
                    assert len(matches)==1 and matches[0].read_bytes()==(bank/f"pred_{r['rank']}.bin").read_bytes()
                rec['bank_whole3_prediction_bytes_match_own_science']=True
            print(stamp(),'BOUNDARY_DONE',label,guard['elapsed_s'],flush=True)
        parity_command=[str(PY),'-B',str(PARITY),'analyze','--manifest',str(manifest)]
        status=run(parity_command,out/'parity_analyzer.log',30);rec['parity_returncode']=status;save();assert status==0
        equivalence=transfer(plan,helper,paired,read(out/'RESULT_v1.json'))
        ep=out/'SCIENCE_EQUIVALENCE_TRANSFER_v1.json';helper.put(ep,equivalence);rec['science_equivalence_receipt']=str(ep)
        if reused_bank is not None:
            for p,h in reused_bank['pins'].items():assert sha(p)==h
        br=rows(bank/'rows.jsonl');cr=rows(candidate/'rows.jsonl')
        rec['quiet_vs_bank']=[dict(rank=b['rank'],wall_pct=100*(c['wall_s']/b['wall_s']-1),
             backend_pct=100*(c['backend_s']/b['backend_s']-1),thread_residual_instructions_pct=100*(c['thread_residual_instructions']/b['thread_residual_instructions']-1),
             requests_delta=c['requests']-b['requests'],flops_pct=100*(c['flops']/b['flops']-1)) for b,c in zip(br,cr)]
        for p,h in gate_pins.items():assert sha(p)==h,('gate pin changed',p)
        command=['bash',str(RELEASE_MIN),v['release'],release_tag,str(q),str(LAST_ARCHIVE)]
        rec.update(release_command=command,release_tag=release_tag,qualification=str(q));save()
        status=run(command,out/'release_min.log',150);rec['release_gate_returncode']=status;save();assert status==0
        archive=O/'release/runs'/release_tag/'submission.tar.gz'
        rec.update(archive=str(archive),archive_sha256=sha(archive),status='QUIET_WHOLE3_EQUIVALENCE_AND_ORIGINAL_GATES_PASS_NO_FIRE')
        exitcode=0
    except BaseException as error:
        rec['error']=repr(error);raise
    finally:
        table=processes();resumed=[]
        for pid,before in reversed(list(paused.items())):
            now=table.get(pid)
            if now and now['started']==before['started'] and now['command']==before['command']:
                try:os.kill(pid,signal.SIGCONT);resumed.append(pid)
                except ProcessLookupError:pass
        rec.update(end_utc=stamp(),pause_elapsed_s=time.monotonic()-began,resumed=resumed,exitcode=exitcode)
        try:rec['free_pct_after']=pressure()
        except Exception as error:rec['free_pct_after_error']=repr(error)
        save();print(rec['end_utc'],'RESEARCH_RESUMED',len(resumed),'exitcode',exitcode,flush=True)
    return exitcode

if __name__=='__main__':raise SystemExit(main())
