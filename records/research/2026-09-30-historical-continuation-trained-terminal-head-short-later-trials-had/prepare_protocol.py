from pathlib import Path
import ast, datetime, hashlib, json
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(n,x):
 with (D/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
review=json.loads((D/'data_review.json').read_text())
ranks=review['training_ranks']
parts=json.loads((P/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
assert ranks==parts['train'][:160] and len(ranks)==160
valid=parts['valid'];assert len(valid)==40 and not set(ranks)&set(valid)
data=[{k:x[k] for k in ('rank','weights','truth')} for x in review['data']]
write('data_roster.json',dict(training_ranks=ranks,data=data,authority=pin(D/'data_review.json'),original_valid40=valid,screen2=valid[:2],additional6=valid[2:8]))
old=json.loads((H/'hill_climb_rrpi_extra32_warm_v1/protocol.json').read_text())
pins=[p for p in old['pins'] if not str(p['path']).startswith(str(H/'hill_climb_rrpi_extra32_warm_v1'))]
pins += [pin(D/n) for n in ['train_segment.py','native_parameters.py','controller.py','data_review.json','data_roster.json']]
pins += [pin(P/'fable_physicist_v1/refit6b/refit_mps_v3j2.py'),pin(P/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json')]
# Deduplicate exact file bindings and verify current sources.
unique={}
for r in pins:
 assert pin(r['path'])==r
 unique[r['path']]=r
protocol=dict(schema='rrpi-fixed-full80-segmented-v1',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_admitted=True,
 baseline=pin(H/'hill_climb_pair_rr_inverse_v1/candidate.py'),official=pin(H/'hill_climb_prefix128_fusion_v1/candidate.py'),
 training_ranks=ranks,data=data,updates=80,batch=4,schedule_steps=120,
 shuffle='torch.manual_seed(0), np.random.default_rng(0), continuous RNG permutations for two complete epochs; no per-segment reset of order',
 trainable='3437 main +44144 trunk scalars and10 physical dials; active head15 trained as in original Opus driver; NO explicit head/column gradient masks.',
 frozen='centers,scales,rms,gain,first; dialsX3,c1. Structurallyunused entries retainzero gradients naturally.',
 optimizer=old['optimizer'],torch_runtime='/Users/matt/anaconda3/bin/python',light="/Users/matt/arc-white-box-estimation/.venv/bin/python",guard=pin("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py"),
 segment_limits=dict(updates=4,seconds=600,rss_bytes=7500000000,added_swap_bytes=0,threads=1,serial_only=True),
 checkpoint_policy='Twenty immutable resumption checkpoints preserve params+allAdamstate+RNG+globalstep; only80 is an accuracy candidate. No early model selection, validation, or learning-rate/roster changes.',
 card='Existing tested loop5js+current32rankstar+j3cut, CPUsinglethread. Not identical to original MPSloop4js.',
 remaining_structural_gaps=old.get('remaining_structural_gaps',json.loads((H/'s33_two_update_warm_proposal_v1/protocol.json').read_text())['remaining_structural_gaps']),
 reason='Successful Opus R7s/R8s epoch0≈flat and epoch1improved;2–8update failures didnotexecute80-update fullTRAIN160 trajectory. Priorwarmalsofroze active terminalhead15, unlikeoriginaldriver. One fixedpropercontinuation, not restartfromscratch.',
 prior_failures_preserved=['s33_two_update_warm_proposal_v1','hill_climb_s34_anchored_small4_v1','hill_climb_rrpi_extra32_warm_v1','hill_climb_rrpi_tail_warm2_v1'],
 selection_bias_review='No concrete successful bankwithboth35/100worse was found. This doesnotreclassify recentfailedcandidates. TRAIN160includesall8usualscreenranks; theycannotvalidate this refit.',
 validation=dict(roster_source=pin(P/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json'),historically_reused_not_pristine=True,disjoint_from_this_fit=True,
 screen2=valid[:2],additional6=valid[2:8],fixed40=valid,
 screen2_gates=dict(raw_pct_max=-.1,adjusted_pct_max=-.1,wins_min=1,max_regression_pct=1.,all16_finite=True,layer0_exact=True),
 screen8_gates=dict(raw_pct_max=-.1,adjusted_pct_max=-.1,wins_min=6,ci_upper_max=0.,additional6_gain_required=True,max_regression_pct=1.),
 bank40_gates=dict(adjusted_pct_max=-1.,wins_min=28,ci_upper_max=0.,max_regression_pct=10.,own_boundary=True,own_serving20=True,own_setup3=True,contract=True),
 no_promotions_from_training_loss=True,screen_receipts_before_candidate_run=True),
 pins=list(unique.values()),native_gradient_certified=False,endpoint_export='Only five learned literal assignments +six finalphysicaldial assignments; source inverse mustrestoreRRPI. Exportafter80 only.')
write('protocol.json',protocol)
for p in ('train_segment.py','native_parameters.py','controller.py'):ast.parse((D/p).read_text())
print(json.dumps(dict(protocol=pin(D/'protocol.json'),ranks=len(ranks),valid2=valid[:2],additional6=valid[2:8],pins=len(unique))))

