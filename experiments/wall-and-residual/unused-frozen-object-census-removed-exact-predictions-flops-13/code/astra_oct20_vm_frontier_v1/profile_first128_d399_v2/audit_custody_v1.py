"""Read-only audit and hotspot inventory after the one frozen v2 diagnostic."""
import hashlib
import json
from pathlib import Path
import subprocess

H=Path(__file__).resolve().parent
R=next(p for p in H.parents if (p/'HANDOFF_TO_ASTRA.md').exists())
O=R/'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
C=H/'custody_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
recipe=json.loads((H/'RECIPE_v1.json').read_text())
for r in recipe['files']:assert sha(R/r['path'])==r['sha256']
manifest=json.loads((C/'REMOTE_HASHES_v1.json').read_text())
for r in manifest['files']:assert sha(C/r['path'])==r['sha256'],r['path']
guards={m:json.loads((C/(m+'_GUARD_v1.json')).read_text()) for m in ('control','observe')}
for m,g in guards.items():
 assert g['reason'] is None and g['child_exit']==0 and g['whole_group_retired'] and not g['cleanup_errors'],m
 assert not g['max_owned_swap_bytes'] and not any(g['cgroup_events_final'].values()),m
rows={m:json.loads((C/m/'diagnostic_rows.jsonl').read_text()) for m in guards}
profiles={m:json.loads((C/m/'candidate_cprofile_v1.json').read_text()) for m in guards}
anchor=O/'lead/runs/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native32_vm_custody_v1/part01/pred_35.bin'
pred={m:C/m/'diagnostic_pred_35.bin' for m in guards}
assert pred['control'].read_bytes()==pred['observe'].read_bytes()==anchor.read_bytes()
equality={k:rows['control'][k]==rows['observe'][k] for k in ('flops','operations','requests','free_trips','freed_handles')}
for row in rows.values():assert row['fast_rescue_stats'].get('calls',0)==0
for p in profiles.values():assert p['started'] and p['stack_empty'] and p['sdk_unmodified'] and p['gc_policy_unmodified'] and p['profiler_restored']
assert profiles['observe']['enabled'] and profiles['observe']['rows'] and not profiles['control']['enabled'] and not profiles['control']['rows']
complete_path=C/'COMPLETE_v1.json'
qualified=all(equality.values()) and complete_path.exists()
if qualified:
 complete=json.loads(complete_path.read_text());assert complete['successful'] and complete['identical_requests_and_frees'] and complete['byte_exact']
ranked=sorted(profiles['observe']['rows'],key=lambda r:r['self_s'],reverse=True)
callbacks=[r for r in ranked if r['function']=='__del__' or 'callback' in r['function'] or 'finaliz' in r['function']]
ordinary=[r for r in ranked if r not in callbacks]
x=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status=('QUALIFIED_DIAGNOSTIC_ONLY' if qualified else 'REJECTED_DIAGNOSTIC_PRESERVED_NO_HOTSPOT_QUALIFICATION'),recipe_sha256=sha(H/'RECIPE_v1.json'),source_sha256=recipe['candidate']['sha256'],rehashed_files=len(manifest['files']),same_prediction_as_fresh_control_and_own4da=True,prediction_sha256=sha(pred['control']),prediction_bytes=pred['control'].stat().st_size,exact_gates=equality,observed_values={m:{k:rows[m][k] for k in ('flops','requests','free_trips','freed_handles')} for m in rows},guards={m:{k:g.get(k) for k in ('utc','group','whole_group_retired','child_exit','peak_group_rss_bytes','max_owned_swap_bytes','cgroup_events_final')} for m,g in guards.items()},profile_sha256=sha(C/'observe/candidate_cprofile_v1.json'),callback_rows=callbacks,top_python_self_time_inventory=ordinary[:40] if qualified else [],profile_totals={m:{k:p[k] for k in ('root_wall_ns','root_dispatch_ns','root_outside_ns','candidate_python_self_s','candidate_direct_c_self_s')} for m,p in profiles.items()},limitations=['Instrumented timing is diagnostic, never a release/runtime qualification.','Per-function self-times exclude callee time; direct C child rows are not all outside-dispatch.','Any source callback/finalizer potentially executing inside SDK spans must be excluded from outside-SDK conclusions.','SDK self-time inventory mixes counted and uncounted work; no aggregate runtime credit.','Do not retry a failed strict gate unchanged.'])
p=H/'INDEPENDENT_AUDIT_v1.json'
with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),status=x['status'],gates=equality,profile_totals=x['profile_totals'],top=[{k:r[k] for k in ('function','line','calls','self_s','direct_c_self_s')} for r in x['top_python_self_time_inventory'][:16]])))
