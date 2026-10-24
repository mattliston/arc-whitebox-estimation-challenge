"""Preserve the fixed complete-tensor sketch screen and lifecycle evidence."""
from datetime import datetime,timezone
import hashlib,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
ROOT=Path('/Users/matt/arc-white-box-estimation')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')

seal=read(HERE/'source_sealed.json')
for name,digest in seal['files_sha256'].items():assert sha(HERE/name)==digest,name
review=read(HERE/'source_review.json')
for name,digest in review['sources_sha256'].items():assert sha(A/name)==digest,name
assert sha(D/'pair384_metric_v1/closeout.json')==review['existing_pair384_negative_sha256']
assert sha(D/'lazy_rotation_objective_v1/candidate.py')==seal['parent_sha256']
unit=read(HERE/'unit.json')
pair=read(HERE/'pair_gram_certificate_v2.json')
assert unit['passed'] and pair['passed'] and pair['exact_cross_pair_cancellation']
assert sha(HERE/'pair_gram_certificate_v2.py')==pair['source_sha256']
capture=read(HERE/'capture/result.json')
assert capture['all16_parent_bytes_exact'] and capture['parent_bill_exact'] and capture['cache_retired']
assert not capture['truth_accessed']
assert sha(HERE/'capture/join.npz')==capture['packet_sha256']
assert sha(HERE/'capture/prediction.npy')==capture['output_sha256']
info=read(HERE/'information/result.json')
assert not info['information_gate_passed']
assert not info['truth_accessed'] and not info['actual_accuracy_score_measured'] and not info['production_cost_measured']
assert info['packet_sha256']==capture['packet_sha256']
assert info['basis_sha256']==sha(HERE/'information/fixed_basis.npz')
assert info['source_sha256']==seal['files_sha256']['information.py']
watches=[read(HERE/n) for n in ['unit_watch.json','pair_gram_watch_v2.json','capture_watch.json','information_watch.json']]
for w in watches:
    assert w['completed'] and w['exit_code']==0 and w['error'] is None
    assert w['whole_group_retired'] and w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<8_000_000_000
peak=max([w['max_rss_bytes'] for w in watches]+[x['max_rss_bytes'] for x in [unit,pair,capture,info]])
resources=dict(full_parent_prediction_calls=1,guarded_stages=4,
               charged_parent_prediction_flops=capture['complete_charged_flops'],
               peak_rss_bytes=peak,additional_swap_bytes=0,all_owned_groups_retired=True,
               total_guarded_wall_s=sum(w['elapsed_s'] for w in watches),
               capture_packet_bytes=capture['packet_bytes'])
write_json(HERE/'resources.json',resources)
bank=read(D/'official_best_331599.json')
assert sha(D/'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz')==bank['archive_sha256']
text=f'''# Complete signed-tensor sketch: fixed information failure

The fixed rank256 /256-construction-probe QR basis failed at the declared
current Lazy rank35 layer8 join. Its512 independent verification probes give
residual squared-energy ratio **{info['probe_residual_energy_ratio']:.12g}** versus
the actual rank384 parent, with{info['probe_wins']}/512 query wins. The complete
next repeated-index plane gives ratio **{info['plane_residual_energy_ratio']:.12g}**.
Both ratios had to be<=.75. This decisive failure stops the construction.

| Same current join tensor | Existing rank384 | FP64 sketch rank256 |
| --- | ---: | ---: |
| Relative query residual squared energy | {info['parent_relative_probe_residual_squared']:.12g} | {info['candidate_relative_probe_residual_squared']:.12g} |
| Relative full-plane residual squared energy | {info['parent_relative_plane_residual_squared']:.12g} | {info['candidate_relative_plane_residual_squared']:.12g} |

Marginal diagonal residual-energy ratio is{info['marginal_diagonal_residual_energy_ratio']:.12g},
reported diagnostically and not used to replace either failed criterion.
Mean paired per-probe error-energy increment is
{info['paired_probe_energy_difference_mean']:.12g}, with empirical standard error
{info['paired_probe_energy_difference_se']:.12g} over these512 fixed random probes.
That sampling diagnostic is not uncertainty over networks or labels.

The target is the complete signed CP tensor supplied by the current shared
prefix Qp F plus its dense entering cohort J, pushed through the actual next
weight rows. It is not a sampled physical truth. Construction includes all
cross-pair terms and refreshes both contracted legs. The trial uses optimistic
FP64 algebra; the comparison uses the parent's actual rounded output factors.
No full-estimator error, score, production compute or runtime gain was measured.
This is one join/network, not a universal tensor-rank or sketching bound.

The capture has896 active rows,6144 prefix columns and1984 entering columns,
8128 total. All16 Lazy output bytes and its exact398080917697F cold bill match
the original receipt. Research copies are explicitly outside a deployable
implementation claim. The original cache retires; the54,920,372-byte packet
is sealed before analysis. No accuracy labels or new samples were opened.

The original three tiny cases certify contractions, full Gram and three-leg
projection with maximum scaled error7.673381875059282e-14. The v1 sum-individual
cube diagnostic was not a complete-pair comparison; an additive v2 exact
witness supplies the missing distinction: two complete pairs cancel to zero,
while their summed individual pair Grams equal diag(2,2,0). Both records and
all five originally sealed source/protocol files remain unchanged. QR relative
orthogonality error is{info['qr_orthogonality_relative']:.12g}.

All four guarded stages exit0 and retire. Peak RSS including self-reports is
{peak} bytes, zero added swap. Analysis takes{info['seconds']:.6f}s and peaks at
{info['max_rss_bytes']} bytes. No rank100, other join, extra probes, seed/rank/
precision/blending search, fit, production port, accuracy panel or upload
follows this failure. Earlier pair384 and whole-pair512 negatives stay closed.

Official best remains Lazy #331599 adjusted{bank['adjusted']:.16g}. The4e-9
milestone is achieved;3e-9,2e-9 and approximately1e-9 remain active/unmet.
Future qualified Phase2 uploads have the user's explicit preauthorization.
'''
with (HERE/'NEGATIVE_RESULT.md').open('x') as stream:stream.write(text)
artifacts={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
size=sum((HERE/n).stat().st_size for n in artifacts)
assert size<200_000_000
closeout=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
              status='closed negative at fixed full-tensor sketch information gate',
              goal_status='active',goal_achieved=False,official_bank=bank,
              parent_sha256=seal['parent_sha256'],unit_passed=True,cross_pair_witness_passed=True,
              original_source_seal_unchanged=True,capture_output_and_bill_parity_passed=True,
              information_gate_passed=False,probe_residual_energy_ratio=info['probe_residual_energy_ratio'],
              plane_residual_energy_ratio=info['plane_residual_energy_ratio'],probe_wins=info['probe_wins'],
              verification_probes=512,fit_probes=256,trial_rank=256,parent_rank=384,layer=8,network_rank=35,
              truth_arrays_opened=False,actual_accuracy_score_measured=False,production_cost_measured=False,
              resources=resources,artifact_bytes_before_closeout=size,artifacts_sha256=artifacts,
              remaining_disk_GiB=shutil.disk_usage(ROOT).free/2**30)
write_json(HERE/'closeout.json',closeout)
print(json.dumps({k:v for k,v in closeout.items() if k not in ['artifacts_sha256','official_bank']}))
