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

reports={};panel_reports={};observations=[];guards=[];all_rows=[];contexts=0
for protocol,dir_name,native_count in [('cold13','root_cold13_v1',26),('warm8','root_warm8_v1',16),('warm8_counterbalanced','root_warm8_counterbalanced_v1',16)]:
 D=H/dir_name;plan=read(D/'COVERAGE_PLAN_v1.json');result=read(D/'COVERAGE_RESULT_v1.json');iso=read(D/'ISOLATION_v1.json');ip=read(D/'ISOLATED_PLAN_v1.json')
 assert sha(D/'COVERAGE_PLAN_v1.json')==result['plan_sha256']
 assert sha(D/'ISOLATED_PLAN_v1.json')==iso['manifest_sha256']
 for q,digest in ip['source_pins'].items():assert sha(q)==digest,(q,digest)
 assert sha(D/'run_isolated_root_v1.py')==iso['runner_sha256']
 for f in ['COVERAGE_RESULT_v1.json','ISOLATION_v1.json']:sha(D/f)
 for arm,s in plan['sources'].items():assert sha(s['path'])==s['sha256']
 assert plan['sources']['combined']['sha256']=='f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6'
 assert plan['sources']['control']['sha256']=='660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200'
 execution_arms=['combined','control']if protocol=='warm8_counterbalanced'else['control','combined']
 labels=[f'{arm}_part{g["number"]:02d}'for g in plan['groups']for arm in execution_arms]+['analyze']
 assert len(labels)==native_count+1 and list(ip['commands'])==labels
 assert [x['label']for x in iso['runs']]==labels
 assert iso['exitcode']==0 and not iso['cleanup_errors'] and 77740 in iso['resumed']
 assert iso['free_pct_before']>=60 and iso['no_release_or_fire'] is True
 for run in iso['runs']:
  assert run['returncode']==0 and run['command']==ip['commands'][run['label']]
  if run['label']!='analyze':assert run['command']==plan['commands'][run['label']]
 if protocol!='cold13':assert[g['ranks']for g in plan['groups']]==FIXED_PAIRS
 else:assert[g['ranks'][0]for g in plan['groups']]==[35,150,200,250,259,277,287,16,293,300,310,312,316]
 combined=[]
 for group in plan['groups']:
  j=group['number'];arms={};ranks=group['ranks']
  comparable=group['reference_bill_comparable'] if protocol!='cold13'else[group['reference_bill_comparable']]
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
   assert g['max_swap_growth_bytes']<=1024**3 and g['min_pressure_free_pct']>=50
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
   row=dict(protocol=protocol,group=j,rank=a['rank'],cold=protocol=='cold13'or pos==0,warm_target=protocol!='cold13'and pos==1,**{k+'_pct':100*(b[k]/a[k]-1)for k in FIELDS},residual_time_delta_ms=1000*(b['residual_s']-a['residual_s']),control={k:a[k]for k in FIELDS},candidate={k:b[k]for k in FIELDS},old_bill_comparable=comparable[pos])
   rr=next(x for x in result['rows']if x['group']==j and x['rank']==a['rank'])
   for k in FIELDS:same(row[k+'_pct'],rr[k+'_pct']);assert rr['control'][k]==a[k]and rr['candidate'][k]==b[k]
   same(row['residual_time_delta_ms'],rr['residual_time_delta_ms'])
   if comparable[pos]:
    expected=b['flops']-group['references']['combined'][pos]['row']['flops']
    field='paid_final_flop_delta_same_context_physics'if protocol!='cold13'else'paid_final_flop_delta_same_cold_physics'
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
   reports[protocol+'_'+name]=s
 all_rows.extend(combined)
 panel_reports[protocol]=result


assert len(guards)==58 and len(observations)==90 and len(all_rows)==45 and contexts==31
newdir=H/'root_warm8_counterbalanced_v1';newplan=read(newdir/'COVERAGE_PLAN_v1.json');new=panel_reports['warm8_counterbalanced']
assert sha(newdir/'COVERAGE_PLAN_v1.json')=='9fbe07f1b6790fd199ec6b9ba0d3a2024d534118e5944c1aede8430b4fa421f1'
assert new['status']=='COMPLETE_COUNTERBALANCED_FIXED_WARM8_REQUIRES_ROOT_JUDGMENT'
assert newplan['execution_order']=='combined_then_control'
fixed={'new_warm8_mean_residual_pct_max':-2.0,'both_panels_warm8_mean_residual_pct_max':-3.0,'each_pooled_target_mean_residual_pct_max':-1.0,'rank310_five_cold_context_mean_residual_pct_max':0.0,'all16_new_observation_residual_pct_upper_exclusive':5.0}
assert all(newplan['decision_criteria'][k]==v for k,v in fixed.items())
assert new['preregistered_criteria']==newplan['decision_criteria']
assert new['original_panel_immutable_reverified']['rows']==panel_reports['warm8']['rows']
assert new['original_panel_immutable_reverified']['proof']==panel_reports['warm8']['proof']
old_rows=[r for r in all_rows if r['protocol']=='warm8'];new_rows=[r for r in all_rows if r['protocol']=='warm8_counterbalanced']
joint_targets=[];joint_prefix=[]
for warm,destination,supplied in [(True,joint_targets,new['joint_targets']),(False,joint_prefix,new['joint_conditioning_prefixes'])]:
 for j in range(1,9):
  a=next(r for r in old_rows if r['group']==j and r['warm_target']==warm)
  b=next(r for r in new_rows if r['group']==j and r['warm_target']==warm)
  assert a['rank']==b['rank']
  paired=dict(group=j,rank=a['rank'],warm_target=warm,original=a,replication=b,mean_changes_pct={k+'_pct':statistics.mean([a[k+'_pct'],b[k+'_pct']])for k in FIELDS},residual_pct_range=[min(a['residual_s_pct'],b['residual_s_pct']),max(a['residual_s_pct'],b['residual_s_pct'])],residual_time_delta_ms_range=[min(a['residual_time_delta_ms'],b['residual_time_delta_ms']),max(a['residual_time_delta_ms'],b['residual_time_delta_ms'])])
  check=next(r for r in supplied if r['group']==j)
  assert check['rank']==paired['rank']and check['warm_target']==warm
  for k,v in paired['mean_changes_pct'].items():same(v,check['mean_changes_pct'][k])
  assert paired['residual_pct_range']==check['residual_pct_range']and paired['residual_time_delta_ms_range']==check['residual_time_delta_ms_range']
  assert len(check['observations'])==2
  for label,source in [('original_control_first',a),('replication_candidate_first',b)]:
   cr=next(r for r in check['observations']if r['panel']==label)
   for k in FIELDS:
    same(cr[k+'_pct'],source[k+'_pct'])
    assert cr['control'][k]==source['control'][k]and cr['candidate'][k]==source['candidate'][k]
  destination.append(paired)
joint_mean={k+'_pct':statistics.mean(r['mean_changes_pct'][k+'_pct']for r in joint_targets)for k in FIELDS}
joint_prefix_mean={k+'_pct':statistics.mean(r['mean_changes_pct'][k+'_pct']for r in joint_prefix)for k in FIELDS}
for k in joint_mean:same(joint_mean[k],new['joint_warm8_mean_changes_pct'][k]);same(joint_prefix_mean[k],new['joint_conditioning_mean_changes_pct'][k])
cold310=[r for r in all_rows if r['rank']==310 and r['cold']]
assert len(cold310)==5 and sum(r['protocol']=='cold13'for r in cold310)==1 and sum(r['protocol']=='warm8'for r in cold310)==2 and sum(r['protocol']=='warm8_counterbalanced'for r in cold310)==2
cold310_mean=statistics.mean(r['residual_s_pct']for r in cold310)
same(cold310_mean,new['rank310_five_cold_context_mean_residual_pct'])
newtargetmean=reports['warm8_counterbalanced_primary_warm8']['mean_changes_pct']['residual_s_pct']
criteria=dict(all_original_guards_bytes_and_invoices_clean=True,new_warm8_mean=newtargetmean<=-2.,both_panel_warm8_mean=joint_mean['residual_s_pct']<=-3.,all_eight_pooled_target_means=all(r['mean_changes_pct']['residual_s_pct']<= -1.0 for r in joint_targets),rank310_all_five_cold_context_mean=cold310_mean<=0.,no_new_large_regression_in_any_observation=all(r['residual_s_pct']< 5.0 for r in new_rows))
assert criteria==new['criteria_results'] and all(criteria.values())==new['all_local_operational_criteria_met']
old_nonimproving=[r for r in old_rows if r['residual_s_pct']>=0];new_nonimproving=[r for r in new_rows if r['residual_s_pct']>=0]
assert len(old_nonimproving)==1 and old_nonimproving[0]['rank']==310 and old_nonimproving[0]['group']==7
assert len(new_nonimproving)==len(new['new_nonimproving_observations'])
for r in new_nonimproving:
 q=next(x for x in new['new_nonimproving_observations']if(x['group'],x['rank'])==(r['group'],r['rank']));same(q['residual_s_pct'],r['residual_s_pct'])
assert len(new['all_nonimproving_observations_preserved'])==len(old_nonimproving)+len(new_nonimproving)
for label,rs in [('original_control_first',old_nonimproving),('replication_candidate_first',new_nonimproving)]:
 for r in rs:
  q=next(x for x in new['all_nonimproving_observations_preserved']if(x['panel'],x['group'],x['rank'])==(label,r['group'],r['rank']));same(q['residual_s_pct'],r['residual_s_pct'])
report=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status='INDEPENDENT_COMPLETE_COLD13_AND_BOTH_WARM8_PANELS_AUDIT_PASS',source_release_sha256='f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6',independent_method='Standalone standard-library audit of raw receipts; did not invoke supplied analyzers or root result review. All ratios, pooled means and operational gates independently recomputed.',guards_verified=58,prediction_observations_verified=90,actual_invoices_operation_sums_verified=90,exact_same_context_own_paid_invoices_verified=31,all_original_commands_and_isolation_order_verified=True,all_source_and_harness_pins_rehashed=True,all_process_groups_retired_and_77740_restored=True,original_125s_7_5GB_guards_unchanged=True,maximum_rss_bytes=max(g['rss_bytes']for g in guards),minimum_free_pct=min(g['min_free_pct']for g in guards),maximum_swap_growth_bytes=max(g['swap_growth_bytes']for g in guards),summaries=reports,joint_warm8_mean_changes_pct=joint_mean,joint_conditioning_mean_changes_pct=joint_prefix_mean,joint_targets=joint_targets,joint_conditioning_prefixes=joint_prefix,cold310_five_contexts=cold310,cold310_five_context_mean_residual_pct=cold310_mean,preregistered_operational_criteria=fixed,criteria_results=criteria,all_local_operational_criteria_met=all(criteria.values()),all_paired_rows=all_rows,all_observations=observations,guard_receipts=guards,preserved_original_nonimproving_observations=old_nonimproving,preserved_new_nonimproving_observations=new_nonimproving,no_best_of_or_trimming=True,no_new_panel_requested=True,no_automatic_further_retry=True,limitations=['Operational local material-margin decision criteria, not statistical significance or a hosted failure probability bound.','Original3-prediction reference was retired at7.510163456GB by the7.5GB RSS guard, not an observed OS OOM; no failed partial used and no guard raised.','Formerthird targets200/300/316 now follow only their originalfirst prefix, so this quiet evidence does not qualify originalthird-position retained depth. OriginalMac32 retains own prediction/bill coverage, but its timing comparison is noncontemporaneous.','Conditioning-prefix ranks repeat; rank310 five cold contexts are not5independent networks. The original+22.238922ms/+9.396064%prefix regression remains in every joint analysis.','AllactualFLOP debits and own-source prediction bytes are retained. Runtime arithmetic does not turn a localscore projection into a publicbank.'],source_changed=False,native_launched=False,root_verdict_inferred=False,release_or_fire_authorized=False,pins=PINS)
destination=H/'INDEPENDENT_COUNTERBALANCED_RUNTIME_AUDIT_v2.json';assert not destination.exists();destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),pins=len(PINS),new_warm8=reports['warm8_counterbalanced_primary_warm8'],joint_warm8_mean_changes_pct=joint_mean,cold310_mean=cold310_mean,criteria=criteria,all_pass=all(criteria.values()),new_nonimproving=[dict(group=r['group'],rank=r['rank'],warm_target=r['warm_target'],residual_s_pct=r['residual_s_pct'],residual_time_delta_ms=r['residual_time_delta_ms'])for r in new_nonimproving])))
