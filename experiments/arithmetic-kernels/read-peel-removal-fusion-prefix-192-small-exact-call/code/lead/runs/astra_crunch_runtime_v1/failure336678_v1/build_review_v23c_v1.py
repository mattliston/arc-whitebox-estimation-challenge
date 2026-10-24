from pathlib import Path
import json,hashlib,statistics
P=Path(__file__).resolve().parent;O=P.parents[3]
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=P/'candidate_gen2_read_off_fusion192_v23c.py'
age=load(P/'AGE20_RESULT_v1.json');rs=age['rows'];guard=age['guard'];assert len(rs)==20
assert age['summary']['all_finite'] and age['summary']['all_enabled'] and age['summary']['all_full_gc_only_at_return']
assert all(r['full_gc_count']==1 for r in rs)
assert guard['completed'] and guard['whole_group_retired'] and guard['exit_code']==0 and not guard['error']
assert guard['max_rss_bytes']<7500000000 and guard['max_footprint_bytes']<7500000000
stress=age['end_stress'];assert stress['automatic_gc_enabled'] and stress['threshold']==[700,10,10]
raw=list(map(json.loads,(P/'integrated_age20_v1/rows.jsonl').read_text().splitlines()))
assert all(r['alias_hold_lifetime']['live']==0 for r in raw)
repeat=list(map(json.loads,(P/'integrated_35_100_v1/rows.jsonl').read_text().splitlines()))
for r,b in zip(raw[:2],repeat):
 assert r['rank']==b['rank'] and r['flops']==b['flops'] and r['requests']==b['requests']
 assert (P/'integrated_age20_v1'/f"pred_{r['rank']}.bin").read_bytes()==(P/'integrated_35_100_v1'/f"pred_{r['rank']}.bin").read_bytes()
startup=load(P/'CONCURRENT_V23C_v1_SUMMARY.json');assert startup['passed']
package=load(P/'PACKAGE_REVIEW_v23c_v1.json');assert sha(source)==package['source_sha256']==age['source_sha256']
archive=Path(package['archive']);assert sha(archive)==package['archive_sha256']
assert load(Path(package['stage_a']))['passed']
paired=[r for r in rs if 'parent_residual_instructions'in r]
ordinary=0.364317867
nonfull=[(r['residual_instructions']-r['full_gc_instructions'])/r['parent_residual_instructions']for r in paired]
scale=ordinary/min(r['parent_residual_instructions']for r in paired)
projection=dict(reference='336678 completed ordinary warm maximum, excluding the isolated failed43',ordinary_max_s=ordinary,paired_cases=len(paired),max_non_fullscan_instruction_ratio=max(nonfull),mean_non_fullscan_instruction_ratio=statistics.mean(nonfull),min_parent_residual_instructions=min(r['parent_residual_instructions']for r in paired),worst_charged_full_scan_instructions=max(r['full_gc_instructions']for r in rs),worst_charged_full_scan_wall_s=max(r['full_gc_wall_s']for r in rs),instruction_scaled_ordinary_plus_worst_full_scan_s=ordinary*max(nonfull)+scale*max(r['full_gc_instructions']for r in rs),local_scan_wall_added_to_instruction_scaled_ordinary_s=ordinary*max(nonfull)+max(r['full_gc_wall_s']for r in rs),no_double_count='The explicit full-scan instructions are subtracted from each paired candidate total before the worst aged scan is added once.',limitation='Conditional sensitivity calculations, not platform guarantees. CPU scaling, memory stalls and distinct unseen geometry may differ. Remote failure causality remains unproven.')
review=dict(status='REVIEW_REQUIRED_NOT_CONFIRMED',uploaded=False,source=str(source),source_sha256=sha(source),archive=str(archive),archive_sha256=sha(archive),stage_a=package['stage_a'],changes=['Preserve automatic young-generation GC; defer only generation2 during prediction and explicitly collect2 before return, restoring original thresholds/enabled state in nested finally.','Retain original peel admission/cap-debit accounting but route every peel product to its existing parent; keep P40/W3/ZB4.','Spend removed peel setup work on fixed historical generic fusion/finish prefix192, without new numeric owners or packed/boxed geometry changes.'],startup=startup,age20=age,paired_integrated=load(P/'INTEGRATED_RESULT_v1.json'),conditional_residual_projection=projection,source_peer=str(O/'lead/runs/astra_crunch_physics_v1/gc_repair336657_v1/PEER_V23C_v1.json'),limitations=['No remote failed fixture is locally available; actual outside-dispatch GC was not reproduced in baseline10.','This does not deliver8–10%steady residual reduction. One controlled full scan replaces unpredictable placement only if that mechanism caused the isolated tail.','Twenty distinct fixtures cover average worker age, not a proven maximum platform assignment.','The harness accumulates GC event/metric objects outside predictions; aged scans conservatively include that instrumentation heap.','Read fallback changes floating-point recombination, so numerical comparison uses deltas rather than a false byte-parity claim.','Root must decide whether evidence justifies release; no upload or automatic CONFIRMED.'])
P.joinpath('REVIEW_v23c_v1.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'status':review['status'],'source_sha256':review['source_sha256'],'archive_sha256':review['archive_sha256'],'summary':age['summary'],'projection':projection},indent=2))
