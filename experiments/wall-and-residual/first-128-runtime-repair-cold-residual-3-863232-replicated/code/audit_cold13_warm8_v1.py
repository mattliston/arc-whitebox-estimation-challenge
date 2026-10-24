from pathlib import Path
import hashlib,json,math,statistics,struct,subprocess

R=Path('/Users/matt/arc-white-box-estimation')
O=R/'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H=O/'lead/runs/astra_oct20_bill_v1/runtime_first128_frozen_census_v1'
PINS={};FIELDS=['wall_s','backend_s','residual_s','thread_residual_cycles','thread_residual_instructions','requests','flops']
FIXED_PAIRS=[[35,150],[35,200],[250,259],[277,287],[16,293],[16,300],[310,312],[310,316]]
def sha(p):
 p=Path(p);b=p.read_bytes();h=hashlib.sha256(b).hexdigest();PINS[str(p.resolve())]=h;return h
def read(p):return json.loads(Path(p).read_text())
def same(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)
def summary(rows):
 return dict(n=len(rows),mean_changes_pct={k+'_pct':statistics.mean(r[k+'_pct']for r in rows)for k in FIELDS},residual_pct_range=[min(r['residual_s_pct']for r in rows),max(r['residual_s_pct']for r in rows)],residual_time_delta_ms_range=[min(r['residual_time_delta_ms']for r in rows),max(r['residual_time_delta_ms']for r in rows)],actual_residual_improvements=sum(r['residual_s_pct']<0 for r in rows))

reports={};observations=[];guards=[];all_rows=[];contexts=0
for protocol,dir_name,native_count in [('cold13','root_cold13_v1',26),('warm8','root_warm8_v1',16)]:
 D=H/dir_name;plan=read(D/'COVERAGE_PLAN_v1.json');result=read(D/'COVERAGE_RESULT_v1.json');iso=read(D/'ISOLATION_v1.json');ip=read(D/'ISOLATED_PLAN_v1.json')
 assert sha(D/'COVERAGE_PLAN_v1.json')==result['plan_sha256']
 assert sha(D/'ISOLATED_PLAN_v1.json')==iso['manifest_sha256']
 for q,digest in ip['source_pins'].items():assert sha(q)==digest,(q,digest)
 assert sha(D/'run_isolated_root_v1.py')==iso['runner_sha256']
 for f in ['COVERAGE_RESULT_v1.json','ISOLATION_v1.json']:sha(D/f)
 for arm,s in plan['sources'].items():assert sha(s['path'])==s['sha256']
 assert plan['sources']['combined']['sha256']=='f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6'
 assert plan['sources']['control']['sha256']=='660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200'
 labels=[f'{arm}_part{g["number"]:02d}'for g in plan['groups']for arm in ['control','combined']]+['analyze']
 assert len(labels)==native_count+1 and list(ip['commands'])==labels
 assert [x['label']for x in iso['runs']]==labels
 assert iso['exitcode']==0 and not iso['cleanup_errors'] and 77740 in iso['resumed']
 assert iso['free_pct_before']>=60 and iso['no_release_or_fire'] is True
 for run in iso['runs']:
  assert run['returncode']==0 and run['command']==ip['commands'][run['label']]
  if run['label']!='analyze':assert run['command']==plan['commands'][run['label']]
 if protocol=='warm8':assert[g['ranks']for g in plan['groups']]==FIXED_PAIRS
 else:assert[g['ranks'][0]for g in plan['groups']]==[35,150,200,250,259,277,287,16,293,300,310,312,316]
 combined=[]
 for group in plan['groups']:
  j=group['number'];arms={};ranks=group['ranks']
  comparable=group['reference_bill_comparable'] if protocol=='warm8'else[group['reference_bill_comparable']]
  for arm in ['control','combined']:
   tag=f'{arm}_part{j:02d}';out=Path(plan['output'])/tag;g=read(out/'GUARD_v1.json');m=read(out/'manifest.json');l=read(out/'launcher.json');rs=[json.loads(x)for x in(out/'rows.jsonl').read_text().splitlines()if x.strip()]
   for n in ['GUARD_v1.json','manifest.json','launcher.json','rows.jsonl']:sha(out/n)
   proof=result['proof'][tag]
   assert PINS[str((out/'GUARD_v1.json').resolve())]==proof['guard_sha256']
   assert PINS[str((out/'rows.jsonl').resolve())]==proof['rows_sha256']
   assert g['guard_version']==3 and g['completed'] is True and g['exit_code']==0 and g['error'] is None
   assert g['whole_group_retired'] is True and not g['remaining_rows']
   assert g['footprint_trigger_bytes']==g['rss_backstop_bytes']==7500000000
   assert g['max_rss_bytes']<=7500000000 and g['elapsed_s']<=125
   assert g['failed_readings']==0 and not g['tolerated_failures']
   assert g['max_swap_growth_bytes']==0
   cmd=plan['commands'][tag]
   assert cmd[cmd.index('--seconds')+1]=='125' and cmd[cmd.index('--footprint-limit')+1]=='7500000000' and cmd[cmd.index('--rss-backstop')+1]=='7500000000'
   boundary_at=cmd.index(str(O/'lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/boundary_run_dual_v3.py'))
   assert g['command']==cmd[boundary_at-2:]
   assert m['sha256']==l['sha256']==plan['sources'][arm]['sha256']
   assert m['ranks']==ranks==[r['rank']for r in rs]
   assert m['client_flopscope']=='0.12.3' and m['short_setup'] is True and m['profiled'] is False and m['sites'] is False
   assert m['setup']['failed'] is False and m['setup']['setup_s']>0
   assert m['diagnostic_harness_sha256']==plan['pins'][str(O/'lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/boundary_client_dual_v3.py')]
   assert m['thread_helper_sha256']==plan['pins'][str(O/'lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/thread_counts_v3.py')]
   assert l['client_exit']==l['server_exit']==0 and l['handle_cap']==100000
   assert l['server']['array_count_limit']==100000
   guards.append(dict(protocol=protocol,label=tag,rss_bytes=g['max_rss_bytes'],footprint_bytes=g['max_footprint_bytes'],swap_growth_bytes=g['max_swap_growth_bytes'],min_free_pct=g['min_pressure_free_pct'],elapsed_s=g['elapsed_s'],max_sample_interval_s=g['maximum_sample_interval_s']))
   for pos,r in enumerate(rs):
    ref=group['references'][arm][pos];assert r['rank']==ref['rank']
    p=out/f'pred_{r["rank"]}.bin';raw=p.read_bytes();ph=sha(p)
    assert len(raw)==65536 and r['n_out']==16384 and r['dtype']=='float32' and r['nonfinite']==0
    assert all(math.isfinite(x[0])for x in struct.iter_unpack('<f',raw))
    assert ph==sha(ref['prediction'])==ref['prediction_sha256']
    assert r['fast_rescue_stats']['calls']==r['fast_rescue_stats']['elements']==0
    assert r['fast_rescue_final_state']['on'] is False and all(v is None for v in r['fast_rescue_control'].values())
    assert r['compile_stats']['missing']==r['compile_stats']['capacity_skipped']==0
    assert sum(v[1]for v in r['operations'].values())==r['flops']
    assert r['thread_total_cycles']-r['thread_dispatch_cycles']==r['thread_residual_cycles']
    assert r['thread_total_instructions']-r['thread_dispatch_instructions']==r['thread_residual_instructions']
    assert all(math.isfinite(r[k])and r[k]>0 for k in FIELDS)
    if comparable[pos] and arm=='control':assert r['flops']==ref['row']['flops']
    if arm=='combined':
     pr=group['own_paid_references'][pos];assert pr['rank']==r['rank']
     assert ph==sha(pr['prediction'])==pr['prediction_sha256']
     if comparable[pos]:
      assert r['flops']==pr['row']['flops'] and r['flops']>ref['row']['flops'];contexts+=1
    observations.append(dict(protocol=protocol,label=tag,position=pos+1,rank=r['rank'],prediction_sha256=ph,flops=r['flops'],invoice_operation_sum_exact=True,old_bill_comparable=comparable[pos],residual_s=r['residual_s'],thread_residual_cycles=r['thread_residual_cycles'],thread_residual_instructions=r['thread_residual_instructions']))
   arms[arm]=rs
  for pos,(a,b)in enumerate(zip(arms['control'],arms['combined'])):
   row=dict(protocol=protocol,group=j,rank=a['rank'],cold=protocol=='cold13'or pos==0,warm_target=protocol=='warm8'and pos==1,**{k+'_pct':100*(b[k]/a[k]-1)for k in FIELDS},residual_time_delta_ms=1000*(b['residual_s']-a['residual_s']),control={k:a[k]for k in FIELDS},candidate={k:b[k]for k in FIELDS},old_bill_comparable=comparable[pos])
   rr=next(x for x in result['rows']if x['group']==j and x['rank']==a['rank'])
   for k in FIELDS:same(row[k+'_pct'],rr[k+'_pct']);assert rr['control'][k]==a[k]and rr['candidate'][k]==b[k]
   same(row['residual_time_delta_ms'],rr['residual_time_delta_ms'])
   if comparable[pos]:
    expected=b['flops']-group['references']['combined'][pos]['row']['flops']
    field='paid_final_flop_delta_same_context_physics'if protocol=='warm8'else'paid_final_flop_delta_same_cold_physics'
    assert rr[field]==expected
   else:assert not any(x.startswith('paid_final_')for x in rr)
   combined.append(row)
  
 if protocol=='cold13':
  s=summary(combined)
  for k,v in s['mean_changes_pct'].items():same(v,result['mean_changes_pct'][k])
  assert s['residual_pct_range']==result['residual_pct_range'];reports[protocol]=s
 else:
  target=[r for r in combined if r['warm_target']];prefix=[r for r in combined if not r['warm_target']]
  for name,rows in [('primary_warm8',target),('conditioning_cold_prefix8',prefix)]:
   s=summary(rows)
   for k,v in s['mean_changes_pct'].items():same(v,result[name]['mean_changes_pct'][k])
   for k in ['n','residual_pct_range','residual_time_delta_ms_range','actual_residual_improvements']:assert s[k]==result[name][k]
   reports[name]=s
 all_rows.extend(combined)

assert len(guards)==42 and len(observations)==58 and len(all_rows)==29 and contexts==18
outliers=[r for r in all_rows if r['residual_s_pct']>=0];assert len(outliers)==1 and outliers[0]['rank']==310 and outliers[0]['group']==7
report={'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
 'status':'INDEPENDENT_COMPLETE_COLD13_WARM8_BYTES_GUARDS_BILLS_AND_RUNTIME_RECOMPUTATION_PASS',
 'source_release_sha256':'f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6',
 'independent_method':'Standalone standard-library audit; did not invoke either supplied analyzer or root result review.',
 'guards_verified':42,'prediction_observations_verified':58,'actual_invoices_operation_sums_verified':58,'exact_same_context_own_paid_invoices_verified':18,
 'all_original_commands_and_isolation_order_verified':True,'all_source_and_harness_pins_rehashed':True,'all_process_groups_retired_and_77740_restored':True,
 'original_125s_7_5GB_guards_unchanged':True,'maximum_rss_bytes':max(g['rss_bytes']for g in guards),'minimum_free_pct':min(g['min_free_pct']for g in guards),'maximum_swap_growth_bytes':max(g['swap_growth_bytes']for g in guards),
 'summaries':reports,'all_paired_rows':all_rows,'all_observations':observations,'guard_receipts':guards,'preserved_nonimproving_observations':outliers,
 'candid_lane_judgment':{'measured_progress':'Real final-stack actual residual reduction is verified: all13 singleton-cold cases and all8 warm targets improved, with complete bills and matching predictions.',
  'robustness':'Not yet comfortable treating the available margin as robust enough to spend the scarce repair attempt. Cold minimum improvement is only4.343059ms/1.883641%, warm minimum6.702634ms/2.919115%; the retained cold prefix310 regressed22.238922ms/9.396064% under clean guards despite lower instructions. That outlier is larger than much of the measured benefit.',
  'recommended_next':'One preregistered counterbalanced replication of the entire same fixedwarm8 panel: candidate first then fresh337466 control for each pair, same sources/guards/group order. Retain all eight targets and eight conditioning prefixes; jointly report both panels with equal weights, ranges and every outlier, no best-of selection or original-data replacement.',
  'not_an_absolute_guarantee':'Replication can clarify order/variance; it cannot guarantee hosted timing or erase the observed regression.'},
 'limitations':['Original three-prediction reference failed the7.5GB RSS backstop at7.510163456GB, not an observed OS OOM. No failed partials reused or thresholds raised.',
  'Warm targets200,300,316 are now second predictions after original first prefixes; original third-position retained depth remains untested by this quiet protocol.',
  'All prefix observations remain explicit and are repeated resource-conditioning contexts, not additional independent models.',
  '28of29 paired observations improve, with repeated ranks/protocols; do not present this as29 independent networks or all observations improving.',
  'Cross-host public timing and noncontemporaneous all32 descriptive timings are not qualification substitutes. No instruction/count-only inference.'],
 'source_changed':False,'native_launched':False,'root_verdict_inferred':False,'release_or_fire_authorized':False,'pins':PINS}
dest=H/'INDEPENDENT_COLD13_WARM8_AUDIT_v1.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'pins':len(PINS),'summaries':reports,'guards':42,'observations':58}))
