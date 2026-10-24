"""Shape-only setup selection for the frozen prospective88 policy; no arrays."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent


def read(p): return json.loads(p.read_text())


def pin(p):
    b=p.read_bytes()
    return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def main():
    plan_path=HERE/'prospective_union_price_v1.json'
    assert pin(plan_path)['sha256']=='2e014b0fa100de3bcc622936f2def6c9f244a1ef79a72eab3a89a1a33a466ca6'
    plan=read(plan_path)
    helper_path=H2/'teammate_s32_rank15_dispatch_v1/coherent_plan_v2.py'
    assert pin(helper_path)['sha256']=='5c134be66436e835a81304547739b659fd6961bedbabd3dd7a1cee8188a9dc7c'
    spec=importlib.util.spec_from_file_location('prior_setup_arithmetic',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old=set(helper.source_keys(helper.S32));bundle=set(helper.source_keys(helper.BUNDLE))
    protected=bundle-old
    assert len(old)==600 and len(bundle)==636 and len(protected)==36
    table={tuple(r['key']):r['alternative'] for r in plan['table']}
    surviving=set();replaced=set();per_rank=[];outside_frames={}
    for row in plan['rows']:
        selected={r['root'] for r in row['proposals']}
        left=set();gone=set();outside=Counter()
        with (HERE/f"trace_{row['rank']}.jsonl").open() as stream:
            for line in stream:
                e=json.loads(line)
                if e['kind']=='a_begin' and e['root'] is None:
                    outside[tuple(e['frame'])]+=1
                if e['kind']!='product' or e['op']!='matmul' or len(e['shapes'])!=2:
                    continue
                if any(d!='float32' for d in e['dtypes']):continue
                a,b=e['shapes'];pre=a[:-2]
                if not pre or pre!=b[:-2] or any(v!=7 for v in pre):continue
                d=len(pre);m,k,n=a[-2]*2**d,a[-1]*2**d,b[-1]*2**d
                sides={(0,d,m,k),(1,d,k,n),(2,d,m,n)}
                (gone if e['root'] in selected and (d,m,k,n) in table else left).update(sides)
        # The only A-forward calls outside the recorded complete _mm owners
        # are the fixed three star-product D3 frames, whose nine keys are all
        # explicitly protected. No outside frame is silently ignored.
        assert set(outside)=={(3,1024,1024),(3,32,1024),(3,1024,32)}
        assert {k for k in protected if k[1]==3}=={
            (0,3,1024,1024),(1,3,1024,32),(2,3,1024,32),
            (0,3,32,1024),(1,3,1024,1024),(2,3,32,1024),
            (0,3,1024,32),(1,3,32,1024),(2,3,1024,1024)}
        outside_frames[str(row['rank'])]=[dict(frame=k,calls=v) for k,v in sorted(outside.items())]
        surviving.update(left);replaced.update(gone)
        per_rank.append(dict(rank=row['rank'],surviving_keys=sorted(left),
                             replaced_keys=sorted(gone)))
    new={tuple(k) for k in plan['required_actual_compact_keys']}
    assert len(new)==206
    mandatory=(surviving&bundle)|protected
    room=636-len(new)-len(mandatory)
    assert room>=0
    prior=sorted(bundle-mandatory,key=lambda k:(-helper.side_cost(k),k))
    retained=mandatory|set(prior[:room]);dropped=bundle-retained
    assert len(retained)==430 and len(dropped)==206
    assert not(dropped&surviving) and protected.issubset(retained)
    counts=Counter(k[0] for k in retained)+Counter(k[0] for k in new)
    assert all(counts[k]<=cap for k,cap in enumerate((320,352,320)))
    newbind=sum(15*helper.side_cost(k) for k in new)
    oldbind=sum(helper.side_cost(k) for k in bundle)
    removedbind=sum(helper.side_cost(k) for k in dropped)
    six=read(helper.KEYS6)
    used6={tuple(r['key']) for r in six['rows'] if r['calls']>0}
    six_dropped=[r for r in six['rows'] if tuple(r['key']) in dropped and r['calls']>0]
    output=dict(
        status='CONDITIONAL_SOURCE_ONLY_636_SETUP_PLAN_UNEXECUTED',
        numerical_jobs=0,candidate_built=False,setup_qualified=False,
        rule='Mandatory surviving original-kernel keys in observed35/100 plus36 protected keys; fill remaining old slots by descending original reshape_cost, then lexical key. No target/prediction values.',
        per_rank=per_rank,outside_owner_A_frames=outside_frames,
        surviving_observed_keys=len(surviving),surviving_existing_bundle_keys=len(surviving&bundle),
        replaced_only_existing_bundle_keys=sorted((replaced-surviving)&bundle),
        mandatory_old_count=len(mandatory),prior_fill_old_count=room,
        retained_old_keys=sorted(retained),dropped_old_keys=sorted(dropped),
        protected36=sorted(protected),new_compact_keys=sorted(new),
        total_side_bindings=636,side_counts=dict(counts),side_caps=[320,352,320],
        unchanged_KS_accumulator_bindings=24,expected_total_bindings=660,
        observed35_100_used_old_key_drop_count=0,
        prior_setup_two_reshape_flops=oldbind,dropped_old_two_reshape_flops=removedbind,
        new_HK_two_reshape_flops=newbind,
        proposed_setup_two_reshape_flops=oldbind-removedbind+newbind,
        incremental_setup_two_reshape_flops=newbind-removedbind,
        observed_original_TRAIN6_drop_key_count=len(dropped&used6),
        observed_original_TRAIN6_distinct_drop_reshape_cost=sum(helper.side_cost(k) for k in dropped&used6),
        observed_original_TRAIN6_dropped_key_rows=six_dropped,
        limitations=[
            '636 is side-key count, not proof of setup wall, handles, fusion completion or new namespace reset/eviction behavior.',
            'No prediction-time new binds are assumed only if all required206 compact views and retained430 originals are prebound and remain resident.',
            'The120 dropped keys seen in original TRAIN6 may or may not survive this prospective HK policy there; aggregate key incidence cannot resolve complete owning calls. Their distinct cost is not a changed-policy forecast.',
            'Unknown networks and post-HK floating-point routing can introduce other shapes, binds and evictions; zero observed used-key drops is only for these two unchanged traces.',
            'Setup retains24 original KS accumulator bindings; the206 count does not price additional fused/DAG descriptor construction.',
            'This prospective plan adds no kernel or score claim and does not alter the sealed fixed54 policy or component receipts.'
        ],
        pins=[pin(p) for p in (Path(__file__),plan_path,helper_path,helper.S32,helper.BUNDLE,
              helper.KEYS6,HERE/'trace_35.jsonl',HERE/'trace_100.jsonl')])
    dest=HERE/'prospective_setup636_v1.json'
    with dest.open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=pin(dest),mandatory=len(mandatory),counts=dict(counts),
                         oldbind=oldbind,removedbind=removedbind,newbind=newbind,
                         setup_delta=newbind-removedbind,known6_drops=len(dropped&used6))))


if __name__=='__main__':main()
