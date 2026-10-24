"""Fixed all13 singleton-cold runtime coverage; all actual invoices, no warm transfer."""
import hashlib,importlib.util,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());plan=read(HERE/'COVERAGE_PLAN_v1.json')
for p,v in plan['pins'].items():assert sha(p)==v,('pin changed',p)
hp=Path(plan['proof_helper']);spec=importlib.util.spec_from_file_location('runtime_coverage_proof',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
rows=[];proof={};fields=['wall_s','backend_s','residual_s','thread_residual_cycles','thread_residual_instructions','requests','flops']
assert plan['protocol']=='fixed13_all_singleton_cold_v1'
for group in plan['groups']:
 result={};j=group['number'];ranks=group['ranks'];assert len(ranks)==1 and group['cold'] is True
 for key in ('control','combined'):
  directory=Path(plan['output'])/f'{key}_part{j:02d}';guard=h.valid_guard(directory/'GUARD_v1.json',original=True);manifest=read(directory/'manifest.json');launcher=read(directory/'launcher.json')
  assert manifest['sha256']==launcher['sha256']==plan['sources'][key]['sha256']
  assert manifest['ranks']==ranks and manifest['client_flopscope']=='0.12.3'
  assert manifest['diagnostic_harness_sha256']==plan['pins'][str(h.B/'boundary_client_dual_v3.py')]
  assert manifest['thread_helper_sha256']==plan['pins'][str(h.B/'thread_counts_v3.py')]
  assert launcher['client_exit']==launcher['server_exit']==0 and launcher['handle_cap']==100000
  got=h.rows(directory/'rows.jsonl');assert [x['rank']for x in got]==ranks
  for row,ref in zip(got,group['references'][key]):
   assert sha(ref['prediction'])==ref['prediction_sha256']
   assert h.valid_row(row,directory/f"pred_{row['rank']}.bin")==Path(ref['prediction']).read_bytes()
   assert row['compile_stats']['missing']==row['compile_stats']['capacity_skipped']==0
   if group['reference_bill_comparable']:
    if key=='control':assert row['flops']==ref['row']['flops']
    else:assert row['flops']>ref['row']['flops']
   if key=='combined':
    paidref=group['own_paid_references'][0];assert paidref['rank']==row['rank']
    assert sha(paidref['prediction'])==paidref['prediction_sha256']
    assert (directory/f"pred_{row['rank']}.bin").read_bytes()==Path(paidref['prediction']).read_bytes()
    if group['reference_bill_comparable']:assert row['flops']==paidref['row']['flops'],('cold metadata-only edit changed bill',row['rank'])
  result[key]=got;proof[f'{key}_part{j:02d}']=dict(rows_sha256=sha(directory/'rows.jsonl'),guard_sha256=sha(directory/'GUARD_v1.json'),source_sha256=manifest['sha256'],ranks=ranks,bytes_exact=True,cold=True,old_reference_bill_comparable=group['reference_bill_comparable'])
 control=result['control'][0];candidate=result['combined'][0];physics=group['references']['combined'][0]
 item=dict(group=j,rank=control['rank'],group_first=True,cold=True,original_group=group['original_group'],original_position=group['original_position'],old_reference_bill_comparable=group['reference_bill_comparable'],**{k+'_pct':100*(candidate[k]/control[k]-1)for k in fields},residual_time_delta_ms=1000*(candidate['residual_s']-control['residual_s']),total_flop_delta_vs337466=candidate['flops']-control['flops'],control={k:control[k]for k in fields},candidate={k:candidate[k]for k in fields})
 if group['reference_bill_comparable']:
  item['paid_final_flop_delta_same_cold_physics']=candidate['flops']-physics['row']['flops'];item['paid_final_bill_pct_same_cold_physics']=100*(candidate['flops']/physics['row']['flops']-1)
 rows.append(item)
assert len(rows)==13 and [r['rank']for r in rows]==[r for g in plan['groups']for r in g['ranks']]
assert sum(r['old_reference_bill_comparable']for r in rows)==5
mean={k+'_pct':statistics.mean(row[k+'_pct']for row in rows)for k in fields}
report=dict(utc=h.utc(),status='FIXED13_ALL_COLD_PAIRED_RUNTIME_AND_BYTES_VERIFIED_REQUIRES_ROOT_JUDGMENT',plan_sha256=sha(HERE/'COVERAGE_PLAN_v1.json'),protocol=plan['protocol'],proof=proof,rows=rows,mean_changes_pct=mean,residual_pct_range=[min(x['residual_s_pct']for x in rows),max(x['residual_s_pct']for x in rows)],fresh_controls=True,all13_cold=True,no_warm_runtime_qualification=True,no_failed_partial_reuse=True,no_time_selected_cases=True,no_full32_invoice_transfer=True,no_hosted_guarantee=True,no_release_or_fire=True,limitations=plan['limitations'])
with(HERE/'COVERAGE_RESULT_v1.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(status=report['status'],mean=mean,residual_pct_range=report['residual_pct_range'])))
