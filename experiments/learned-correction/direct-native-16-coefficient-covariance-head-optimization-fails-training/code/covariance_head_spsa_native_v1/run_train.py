import datetime
import numpy as np
from common import HERE,PARENT,RUNTIME,TRAIN,CHECK,read,write,pin,verify,seal_check,guarded,native,qualified,score,target,aggregate,build
from algorithm import update

def as_parent(call,truth):
    return dict(rank=call['rank'],output=pin(call['output']),flops=call['flops'],weights_sha256=call['weights_sha256'],truth=truth)

def parent_call(p):
    return dict(output=p['output']['path'],flops=p['flops'],multiplier=max(.1,p['flops']/2**41))

def score_stage(stage,entries,calls,threshold,wins,worst,strict):
    qualified_rows=[]
    for p,c in zip(entries,calls):
        assert c['rank']==p['rank'] and c['includes_initialization'];qualified(c,p)
        qualified_rows.append(dict(rank=p['rank'],parent=p,call=c,prediction=pin(c['output'])))
    write(HERE/f'{stage}_predictions_sealed.json',dict(rows=qualified_rows,candidate=pin(HERE/'candidate.py'),qualified_before_this_stage_scoring=True))
    rows=[]
    for r in qualified_rows:
        truth=target(r['parent']['truth']);ps=score(parent_call(r['parent']),truth);cs=score(r['call'],truth)
        rows.append(dict(rank=r['rank'],parent=ps,candidate=cs,adjusted_ratio=cs['adjusted']/ps['adjusted'],raw_ratio=cs['raw']/ps['raw']))
    result=aggregate(rows,threshold,wins,worst,strict);write(HERE/f'{stage}_score.json',result)
    print(dict(stage=stage,**{k:result[k] for k in ['passed','pooled_adjusted_ratio','wins','worst_adjusted_ratio']}),flush=True)
    return result['passed']

def main():
    seal_check();guarded('unit',60,[RUNTIME,'-B',HERE/'unit.py']);assert read(HERE/'unit.json')['passed']
    inputs=read(HERE/'inputs.json');parents={p['rank']:p for p in inputs['train']};fresh=[]
    for rank in TRAIN:
        p=parents[rank]
        if rank in [36,38]:continue
        c=native(f'baseline_{rank}',PARENT,[rank])[0];q=qualified(c,p)
        old=np.load(p['output']['path'],allow_pickle=False)
        assert q.tobytes()==old.tobytes() and c['flops']==p['flops']
        parents[rank]=as_parent(c,p['truth']);fresh.append(c)
    write(HERE/'baselines_sealed.json',dict(rows=list(parents.values()),fresh_calls=fresh,all16_bytes_bills_exact=True,
        schedule=pin(HERE/'schedule.json'),sources=pin(HERE/'sources_sealed.json'),target_arrays_not_decoded=True))
    # Only now open the six TRAIN targets. No prior secants or accuracy values used.
    truths={r:target(parents[r]['truth']) for r in TRAIN}
    baseline_scores={r:score(parent_call(parents[r]),truths[r]) for r in TRAIN}
    write(HERE/'train_targets_opened.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_scores=baseline_scores,baselines=pin(HERE/'baselines_sealed.json')))
    schedule=read(HERE/'schedule.json');base=np.asarray(schedule['base_head'],dtype=np.float64);theta=base.copy();centers=[]
    for item in schedule['rows']:
        t=item['t'];rank=item['rank'];delta=np.asarray(item['delta'],dtype=np.float64);calls=[];losses=[];variants=[]
        for sign,label in [(1,'plus'),(-1,'minus')]:
            path=HERE/'variants'/f'iter{t:02d}_{label}.py';variant=build(path,theta+sign*item['c']*delta);variants.append(variant)
            c=native(f'iter{t:02d}_{label}',path,[rank])[0];qualified(c,parents[rank]);calls.append(c)
            losses.append(score(c,truths[rank]))
        lp,lm=[v['adjusted']/baseline_scores[rank]['adjusted'] for v in losses]
        before=theta.copy();theta,meta=update(theta,base,lp,lm,item);centers.append(theta.copy())
        write(HERE/f'iteration_{t:02d}.json',dict(schedule=item,center_before=before.tolist(),variants=variants,calls=calls,
            absolute_losses=losses,normalized_plus=lp,normalized_minus=lm,**meta))
        print(dict(iteration=t+1,of=36,rank=rank,plus_ratio=lp,minus_ratio=lm,center_distance=meta['distance_from_parent']),flush=True)
    assert len(centers)==36
    terminal=np.mean(np.stack(centers[-18:]),axis=0).astype(np.float32);candidate=build(HERE/'candidate.py',terminal)
    write(HERE/'fit.json',dict(completed=True,iterations=36,loss_evaluations=72,terminal_rule='FP32 mean of last18 post-update centers',
        candidate=candidate,source_seal=pin(HERE/'sources_sealed.json'),schedule=pin(HERE/'schedule.json'),
        terminal_distance=float(np.linalg.norm(terminal.astype(np.float64)-base)),new_samples=0,GPU_calls=0))
    calls=[native(f'terminal_train_{r}',HERE/'candidate.py',[r])[0] for r in TRAIN]
    if not score_stage('train',[parents[r] for r in TRAIN],calls,.995,4,1.05,False):
        write(HERE/'outcome.json',dict(stage='train',passed=False,study_closed=True,uploaded=False));return
    entries=[]
    for e in inputs['check']:
        c=native(f'check_parent_{e["rank"]}',PARENT,[e['rank']])[0];entries.append(as_parent(c,e['truth']))
    triple=native('check_candidate_triple',HERE/'candidate.py',[35,100,35]);cold=native('check_candidate_cold100',HERE/'candidate.py',[100])[0]
    assert triple[0]['output_sha256']==triple[2]['output_sha256']
    assert triple[1]['output_sha256']==cold['output_sha256']
    for c in triple+[cold]:qualified(c,entries[0 if c['rank']==35 else 1])
    write(HERE/'check_repeat_qualified.json',dict(exact_repeat35=True,exact_warm_cold100=True,triple=triple,cold100=cold))
    if not score_stage('check',entries,[triple[0],cold],1,0,1.05,True):
        write(HERE/'outcome.json',dict(stage='check',passed=False,study_closed=True,uploaded=False));return
    for stage,wins in [('panel',6),('confirmation',5)]:
        entries=inputs[stage];calls=[native(f'{stage}_{p["rank"]}',HERE/'candidate.py',[p['rank']])[0] for p in entries]
        if not score_stage(stage,entries,calls,1,wins,1.10,True):
            write(HERE/'outcome.json',dict(stage=stage,passed=False,study_closed=True,uploaded=False));return
    write(HERE/'outcome.json',dict(stage='confirmation',passed=True,study_closed=False,release_qualification_pending=True,uploaded=False))

if __name__=='__main__':main()
