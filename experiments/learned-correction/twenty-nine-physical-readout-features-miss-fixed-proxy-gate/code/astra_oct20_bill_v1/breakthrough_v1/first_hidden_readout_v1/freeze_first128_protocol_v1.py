import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
physical=json.loads((D/'physical_input29_v1/PREREG_v1.json').read_text())
price=HERE/'remote_receipts_v1/PRICE_v1.json';diag=HERE/'remote_receipts_v1/DIAGNOSTIC_v1.json'
assert sha(price)=='df17fb223ef8232d7e3992e9fc8994f706dd1107186bbe66803746ae13ba20d0'
assert sha(diag)=='138e09eb0be38df9e13c705e07d0adf45bfd8b4cd4cc222e96e956802c516dc5'
p=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
 kind='FIXED_FIRST_HIDDEN128_INCREMENT_AFTER_INDEPENDENT_FOLD_HIDDEN_COMPLEMENT90',
 train=physical['train'],design_sha256=physical['design_sha256'],
 source_sha256='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510',
 source_ancestry='Authentic777a TRAIN90; this is not an on-policy337466 capture. No adjusted-score claim from proxy.',
 architecture='Per-layer scalar readout from first128 SiLU activation, residualized against state64+constant+hidden2 fixed enc4-nullspace64. Preserve encoder, state, messages and existing second-hidden complement.',
 baseline=physical['baseline'],projection='Normalized uncentered129-dimensional linear projection with ridge1e-10; fit only fold-TRAIN. Raw321=[state64,constant,hidden2_128,first_hidden1_128].',
 fit=physical['fit'],validation=physical['validation'],gate=physical['gate'],
 reconstruction=dict(script='diagnose_hc_inverse_v1.py',script_sha256=sha(HERE/'diagnose_hc_inverse_v1.py'),
  protocol_sha256=sha(HERE/'INVERSE_PROTOCOL_v1.json'),anchor_report_sha256=sha(diag),
  method='Exact frozen inverse(base,target,U,W,b), zero start,20 iterations, step infinity limit2,damping.8,zclip+-8,relative normal ridge1e-8. No parameter, branch, start or iteration sweep.',
  shards=[physical['train'][i:i+15] for i in range(0,90,15)],
  gates=dict(hidden2_scaled_max_lte=128,hidden2_scaled_rms_lte=16,quantized_first_hidden_replay_scaled_max_lte=128,quantized_first_hidden_replay_scaled_rms_lte=16,min_jacobian_normal_eigen_ratio_gte=1e-8,finite=True,z_abs_max_lte=8),
  reference_limitation='All90 have direct captured hidden2 but no observed first-hidden arrays. First-hidden reconstruction is independently validated only on the60 sampled TRAIN4 records using exact dynamic bias; all90 require hidden2 replay and conditioning before labels.',
  acceptance_order='ALL90 H/source/guard/original prediction-byte custody and every reconstruction gate must pass before any truth-array load. No fallback if a gate fails.'),
 price_reference=dict(sha256=sha(price),added_flops=3932160,projected_added_bill_pct=.0016601922139882182,guard_sha256=sha(HERE/'remote_receipts_v1/first_hidden128_readout_price_GUARD_v1.json')),
 pinned_dependencies=dict(physical_custody_module_sha256=sha(D/'physical_input29_v1/fit_physical29_v1.py'),baseline_module_sha256=sha(D/'hidden_complement90_v1/fit_hidden_complement_v1.py')),
 deployment='Compute one first_hidden128@coefficient128 scalar before existing h variable is overwritten. Keep only scalar1024, zero full-feature copies or hidden recomputation; fold residualization into existing state head/bias/hidden2 head.',
 root_authorized=True,no_labels_read_at_freeze=True,no_native_auto_launch=True,no_candidate_export=True,no_held_or_fresh_truth_fit=True,no_grid=True)
with (HERE/'PREREG_FIRST128_v1.json').open('x') as f:json.dump(p,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(utc=p['utc'],prereg_sha256=sha(HERE/'PREREG_FIRST128_v1.json'),dependency=p['pinned_dependencies'])))
