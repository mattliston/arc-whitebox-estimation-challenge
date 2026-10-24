"""Reconcile all accounts and preserve the measured current-source frontier."""
import datetime
import shutil
import subprocess
from common import *

verify();verify('reprice_inputs_sealed.json');verify('finalize_sources_sealed.json')
assert read(HERE/'completed.json')['completed']
result=read(HERE/'reprice_result.json');paths=read(HERE/'paths_verified.json')
assert paths['passed'] and paths['result_sha256']==sha(HERE/'reprice_result.json')
assert all(not f['passes_necessary_3e9_gate'] for f in result['families'])
controls=read(HERE/'controls.json')['rows'];censuses=[]
for control in controls:
    rank=control['rank'];folder=HERE/f'rank{rank:04d}'
    receipt=read(folder/'result.json');data=read(folder/'attribution.json');total=data['total_flops']
    assert receipt['parent_sha256']==PARENT_SHA and receipt['complete_flops']==total==control['flops']
    assert receipt['all16_prediction_bytes_exact'] and receipt['parent_bill_exact'] and receipt['charge_method_restored'] and receipt['numeric_cache_retired']
    assert receipt['original_charge_called_without_argument_changes'] and not receipt['truth_accessed']
    assert sha(folder/'prediction.npy')==control['prediction_sha256']==receipt['output_sha256']
    assert sha(folder/'attribution.json')==receipt['attribution_sha256']
    for category in ('by_function','by_site','by_op_shape','by_layer','by_stack'):
        assert sum(x['flops'] for x in data[category])==total,category
    calls=data['outermost_mm_calls']
    assert sum(c['flops'] for c in calls)==data['outermost_mm_sum']
    assert all(c['input_dtypes']==['float32','float32'] for c in calls)
    assert all(min(*c['left'],*c['right'])>=0 and c['left'][1]==c['right'][0] for c in calls)
    op_bill=sum(x['flops'] for x in data['by_op_shape'] if x['op']=='matmul')
    shape_bill=sum(x['flops'] for x in data['by_op_shape'] if x['op']=='reshape')
    funcs={x['function']:x['flops'] for x in data['by_function']}
    censuses.append(dict(rank=rank,total_flops=total,outermost_mm_calls=len(calls),
                    empty_calls=sum(min(*c['left'],*c['right'])==0 for c in calls),
                    outermost_mm_fraction=data['outermost_mm_sum']/total,matmul_fraction=op_bill/total,
                    cold_reshape_fraction=shape_bill/total,by_function=data['by_function'],
                    young_push_and_shared_plane_fraction=(funcs['_push_bank_shared_newborn']+funcs['_shared_old384_read'])/total,
                    old_reconstruction_inner_rotation_fraction=(funcs['_slr_reconstruct']+funcs['_slr_inner']+funcs['_slr_join_factors'])/total))

# Recheck the generator's whole-source inverse certificates after both calls.
for proof in read(HERE/'build.json')['proofs']:
    new=Path(proof['generated']['path']).read_text()
    for edit in reversed(proof['edits']):
        assert new.count(edit['after'])==1
        new=new.replace(edit['after'],edit['before'],1)
    assert new==Path(proof['original']['path']).read_text()

watch=[read(p) for p in sorted(HERE.glob('*_watch.json'))]
assert len(watch)==4 and all(r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes']==0 for r in watch)
groups={r['pid'] for r in watch}
ps=subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True)
assert not [line for line in ps.splitlines() if len(line.split())>2 and int(line.split()[1]) in groups]
peaks=[read(HERE/f'rank{rank:04d}/result.json')['peak_rss_bytes'] for rank in (35,100)]
peaks.extend([result['peak_rss_bytes'],paths['peak_rss_bytes']])
resources=dict(guarded_groups=len(watch),all_groups_retired=True,full_native_calls=2,
               full_native_flops=sum(c['total_flops'] for c in censuses),
               peak_group_rss_bytes=max(r['max_rss_bytes'] for r in watch),
               peak_self_reported_rss_bytes=max(peaks),max_swap_growth_bytes=0,
               targets_decoded=0,fits=0,new_samples=0,new_downloads=0,GPU_calls=0,
               artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
               disk_free_bytes=shutil.disk_usage(HERE).free)
assert resources['artifact_bytes_before_closeout']<30_000_000
write(HERE/'resources.json',resources)
need=result['required_uniform_fraction_to_cross_3e9_at_unchanged_raw']
bank=read(D/'official_best_331684.json')
assert need==1-3e-9/bank['adjusted']
summary=dict(completed=True,censuses=censuses,required_fraction=need,
             unchanged_raw_floor=bank['raw']*.1,
             families=[dict(name=f['name'],passes_necessary_gate=f['passes_necessary_3e9_gate'],
                            records=[{k:v for k,v in r.items() if k!='details'} for r in f['records']]) for f in result['families']],
             all_5_account_views_reconciled=True,all_source_inverses_rechecked=True,
             verified_schedules=paths['schedules_replayed'],verified_call_records=paths['call_records_reconciled'],
             next_kernel_selected=False)
write(HERE/'summary.json',summary)
names=['_push_bank_shared_newborn','_shared_old384_read','_slr_reconstruct','_flms_star',
       '_shared_old384_push','_flms_covariance','_slr_inner','_slr_join_factors','gnn_step','_edge_source']
by_rank=[{r['function']:r['fraction'] for r in c['by_function']} for c in censuses]
cost_table='\n'.join(f'|{name}|{100*by_rank[0][name]:.3f}%|{100*by_rank[1][name]:.3f}%|' for name in names)
envelope_table='\n'.join(
    f"|{f['name']}|{100*f['records'][0]['optimistic_fraction']:.6f}%|{100*f['records'][1]['optimistic_fraction']:.6f}%|"
    f"{100*f['records'][0]['leaf_storage_limited_fraction']:.6f}%|{100*f['records'][1]['leaf_storage_limited_fraction']:.6f}%|"
    for f in result['families'])
report=f'''# Exact331684 compute frontier: fixed catalog misses standalone3e-9 gate

Both cold predictions reproduce all16 current-parent output bytes and exact
complete bills:413448696909F on35 and416264792412F on100. All five accounting
views independently reconcile every charged operation once, including cold
initialization. The observer calls the original meter once with unchanged
arguments and restores it. Installed libraries and estimator source are
unchanged. Research observer wall times are not production runtime.

The current implementation has295 outermost _mm calls per network,
including two zero-width/zero-bill calls, rather than the older Lazy335.
All observed operands are FP32. Recursive calls are included exactly once
inside their outer bill. Direct products outside _mm remain separately paid.

|Semantic caller,nonoverlapping charge|Rank35 share|Rank100 share|
|---|---:|---:|
{cost_table}

Native matmul-only shares are{100*censuses[0]['matmul_fraction']:.4f}% and
{100*censuses[1]['matmul_fraction']:.4f}%. The broader outermost _mm shares
are{100*censuses[0]['outermost_mm_fraction']:.4f}% and
{100*censuses[1]['outermost_mm_fraction']:.4f}%, including transforms,packing
and other helper operations. Cold reshape shares are
{100*censuses[0]['cold_reshape_fraction']:.4f}%/{100*censuses[1]['cold_reshape_fraction']:.4f}%.
These categories describe actual cost; they are not independent removable
algorithms. In particular, the shared-plane caller includes young reads and
old outer contraction,while the shared-push caller includes acquisition/QR.
The old reconstruction/inner/rotation group costs about23% of the whole call.

|Fixed catalog|35 optimistic saving|100 optimistic saving|35 with leaf limit|100 with leaf limit|
|---|---:|---:|---:|---:|
{envelope_table}

The unchanged-accuracy uniform planning threshold for3e-9 is
{100*need:.9f}%. Both catalogs fail it on BOTH current development networks,
even before adding omitted implementation work. Extended optimistic totals
are352755774112F/351519644667F; the leaf-limited totals are
357048172304F/355985134441F. The necessary1168900096-byte leaf limit covers
simultaneous leaf A/B/C only,not complete workspace or streaming schedules.

These are finite-family optimistic arithmetic envelopes. They assume
complete-call replacement can remove measured initialization/copy work,
omit new transform buffers,copies,dispatch and finite-precision effects,
and include catalog entries without a qualified native implementation.
No actual saving,runtime,accuracy,official prediction or universal bound
follows. This result closes this catalog as a standalone3e-9 route under
the fixed two-case gate. Smaller improvements,other algorithms and combined
accuracy/cost work are not ruled out. No new kernel is selected automatically.

The integer dynamic program is unchanged from the previous certified search:
at most five levels,minimum leaf dimension8,ceil padding or floor-core/fringe
schedules. Three whole-source inverses prove the limited current-parent,
metadata,call-count and current-yardstick adaptations. There are
{result['checks']['classical_partition_cases']} independent scalar partition
checks and{result['checks']['winograd_depth_cases']} Winograd identities.
The separate verifier replays all{paths['schedules_replayed']} emitted
schedules across{paths['call_records_reconciled']} call records,reconciles
totals and checks catalog-superset monotonicity. No extra search was added.

Earlier Smirnov stream/global-batch runtime failures remain actual failures,
not unexecuted batching leads. The old full-Lazy cost envelope remains
preserved with its own earlier parent and21.39% threshold; neither number
should be substituted for these current measurements.

Two full native calls used{resources['full_native_flops']} FLOPs. All four
guards retired,zero added swap; sampled peak RSS
{resources['peak_group_rss_bytes']} bytes,self-reported peak
{resources['peak_self_reported_rss_bytes']} bytes. New artifacts before
closeout total{resources['artifact_bytes_before_closeout']} bytes. No targets,
training,new samples,downloads,GPU,candidate,package or upload.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures.4e-9 is banked;3/2/~1e-9 remain unmet.
At unchanged raw error the0.1 multiplier floor is1.949715441540434e-9,
so approximately1e-9 still requires accuracy improvement. The goal is active.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='closeout.json']
external=[r for r in read(HERE/'sources_sealed.json')['pins'] if not Path(r['path']).is_relative_to(HERE)]
for r in artifacts+external:assert sha(r['path'])==r['sha256']
write(HERE/'closeout.json',dict(completed=True,catalog_standalone_gate_failed=True,
      created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifacts=artifacts,
      external_pins=external,resources=resources,goal_achieved=False,official_changed=False,
      next_kernel_selected=False))
print(json.dumps(dict(completed=True,catalog_standalone_gate_failed=True,artifacts=len(artifacts),
                     external_pins=len(external),resources=resources)),flush=True)
