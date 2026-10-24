"""Fixed warm8 after one original first-rank prefix; preserve every actual invoice and byte."""
import hashlib,importlib.util,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());plan=read(HERE/'COVERAGE_PLAN_v1.json')
for p,v in plan['pins'].items():assert sha(p)==v,('pin changed',p)
hp=Path(plan['proof_helper']);spec=importlib.util.spec_from_file_location('runtime_coverage_proof',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
rows=[];proof={};fields=['wall_s','backend_s','residual_s','thread_residual_cycles','thread_residual_instructions','requests','flops']
fixed_pairs=[[35,150],[35,200],[250,259],[277,287],[16,293],[16,300],[310,312],[310,316]]
assert plan['protocol']=='fixed8_one_original_cold_prefix_then_warm_target_v1'
assert [g['ranks']for g in plan['groups']]==fixed_pairs
for group in plan['groups']:
 result={};j=group['number'];ranks=group['ranks'];assert len(ranks)==2 and group['original_positions'][0]==1 and group['original_positions'][1]in(2,3)
 assert group['reference_bill_comparable']==[True,group['original_positions'][1]==2]
 for key in ('control','combined'):
  directory=Path(plan['output'])/f'{key}_part{j:02d}';guard=h.valid_guard(directory/'GUARD_v1.json',original=True);manifest=read(directory/'manifest.json');launcher=read(directory/'launcher.json')
  assert manifest['sha256']==launcher['sha256']==plan['sources'][key]['sha256']
  assert manifest['ranks']==ranks and manifest['client_flopscope']=='0.12.3'
  assert manifest['diagnostic_harness_sha256']==plan['pins'][str(h.B/'boundary_client_dual_v3.py')]
  assert manifest['thread_helper_sha256']==plan['pins'][str(h.B/'thread_counts_v3.py')]
  assert launcher['client_exit']==launcher['server_exit']==0 and launcher['handle_cap']==100000
  got=h.rows(directory/'rows.jsonl');assert [x['rank']for x in got]==ranks
  for pos,(row,ref)in enumerate(zip(got,group['references'][key])):
   assert row['rank']==ref['rank'];assert sha(ref['prediction'])==ref['prediction_sha256']
   assert h.valid_row(row,directory/f"pred_{row['rank']}.bin")==Path(ref['prediction']).read_bytes()
   assert row['compile_stats']['missing']==row['compile_stats']['capacity_skipped']==0
   if group['reference_bill_comparable'][pos]:
    if key=='control':assert row['flops']==ref['row']['flops']
    else:assert row['flops']>ref['row']['flops']
   if key=='combined':
    paidref=group['own_paid_references'][pos];assert paidref['rank']==row['rank'];assert sha(paidref['prediction'])==paidref['prediction_sha256']
    assert(directory/f"pred_{row['rank']}.bin").read_bytes()==Path(paidref['prediction']).read_bytes()
    if group['reference_bill_comparable'][pos]:assert row['flops']==paidref['row']['flops'],('metadata-only edit changed same-context bill',row['rank'])
  result[key]=got;proof[f'{key}_part{j:02d}']=dict(rows_sha256=sha(directory/'rows.jsonl'),guard_sha256=sha(directory/'GUARD_v1.json'),source_sha256=manifest['sha256'],manifest_sha256=sha(directory/'manifest.json'),launcher_sha256=sha(directory/'launcher.json'),ranks=ranks,bytes_exact=True,old_reference_bill_comparable=group['reference_bill_comparable'])
 for pos,(control,candidate)in enumerate(zip(result['control'],result['combined'])):
  physics=group['references']['combined'][pos];comparable=group['reference_bill_comparable'][pos]
  item=dict(group=j,rank=control['rank'],protocol_position=pos+1,cold_prefix=pos==0,warm_target=pos==1,prefix_rank=ranks[0],original_group=group['original_group'],original_position=group['original_positions'][pos],reduced_retained_depth=pos==1 and group['reduced_retained_depth'],old_reference_bill_comparable=comparable,**{k+'_pct':100*(candidate[k]/control[k]-1)for k in fields},residual_time_delta_ms=1000*(candidate['residual_s']-control['residual_s']),total_flop_delta_vs337466=candidate['flops']-control['flops'],control={k:control[k]for k in fields},candidate={k:candidate[k]for k in fields})
  if comparable:
   item['paid_final_flop_delta_same_context_physics']=candidate['flops']-physics['row']['flops'];item['paid_final_bill_pct_same_context_physics']=100*(candidate['flops']/physics['row']['flops']-1)
  rows.append(item)
targets=[r for r in rows if r['warm_target']];prefixes=[r for r in rows if r['cold_prefix']]
assert len(rows)==16 and len(targets)==len(prefixes)==8
assert[r['rank']for r in targets]==[150,200,259,287,293,300,312,316]
assert sum(r['old_reference_bill_comparable']for r in targets)==5 and all(r['old_reference_bill_comparable']for r in prefixes)
def summary(rs):
 return dict(n=len(rs),mean_changes_pct={k+'_pct':statistics.mean(row[k+'_pct']for row in rs)for k in fields},residual_pct_range=[min(x['residual_s_pct']for x in rs),max(x['residual_s_pct']for x in rs)],residual_time_delta_ms_range=[min(x['residual_time_delta_ms']for x in rs),max(x['residual_time_delta_ms']for x in rs)],actual_residual_improvements=sum(x['residual_s_pct']<0 for x in rs))
report=dict(utc=h.utc(),status='FIXED_WARM8_TWO_PREDICTION_PAIRED_RUNTIME_AND_BYTES_VERIFIED_REQUIRES_ROOT_JUDGMENT',plan_sha256=sha(HERE/'COVERAGE_PLAN_v1.json'),protocol=plan['protocol'],proof=proof,rows=rows,primary_warm8=summary(targets),conditioning_cold_prefix8=summary(prefixes),fresh_controls=True,all8_original_warm_targets_preserved=True,reduced_retained_depth_ranks=[200,300,316],no_three_prediction_runtime_qualification=True,no_failed_partial_reuse=True,no_time_selected_cases=True,no_old_third_position_invoice_transfer=True,no_hosted_guarantee=True,no_release_or_fire=True,limitations=plan['limitations'])
with(HERE/'COVERAGE_RESULT_v1.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(status=report['status'],primary_warm8=report['primary_warm8'],conditioning_cold_prefix8=report['conditioning_cold_prefix8'])))
