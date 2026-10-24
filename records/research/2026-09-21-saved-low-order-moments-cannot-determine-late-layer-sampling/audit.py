"""Preserve two bounded historical findings; standard library only."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
D = HERE.parent
A = D.parent / 'astra_phase2_reexamination_v1'
J = D / 'joint_incoming_moment_head_v1'
OLD = D / 'incoming_third_pair_head_v1'


def read(p):
    return json.loads(p.read_text())


def pin(p):
    p = p.resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


screen = A / 'current_terminal80_magnitude_screen_v1'
requal = A / 'current_p1_terminal80_requalification_v1'
s = read(screen / 'screen_v1/result.json')
r = read(requal / 'inference_v1/result.json')
assert s['passed'] and s['all_predictions_verified_before_truth']
assert not s['native_stage_reconsideration_gate_passed']
assert not s['native_score_measured'] and not s['adjusted_score_measured']
assert not s['estimator_executed'] and not s['checkpoint_selection']
assert s['comparison'] == 'terminal80 MPS versus P1 native; same fixed1e9 means'
assert s['raw_ratio'] == 0.9980865177902755 and s['raw_wins'] == 23
assert r['passed'] and r['inference_integrity_passed'] and r['all40_outputs_sealed']
assert r['historical_training_graphs'] == 320
assert r['historical_updates'] == 80 and r['historical_terminal_epoch'] == 2
assert r['old_R2_remains_failed'] and r['old_builder_not_admitted']
assert r['historical_external_terminal_binding_absent']
assert r['historical_replay_count'] == 13 and r['repeat37_byte_exact']
assert pin(requal / 'inference_v1/result.json')['sha256'] == '9cfee94622c238a8fc0dc6cf7454e28ec54b73fb4998697d30182343fc405bcf'
assert pin(requal / 'inference_watch_v1/result.json')['sha256'] == '5f1d0e310107cefe089d1be4509807492408cbdf0010f29aa28582afa5217701'
terminal = dict(
    additional_training_graphs=320, additional_epochs=2, terminal_update=80,
    sole_terminal_requalified_for_inference=True, original_training_watch_still_failed=True,
    comparison=s['comparison'], raw_ratio=s['raw_ratio'], raw_wins=s['raw_wins'], networks=40,
    p1_native_raw=s['mean_p1_native_raw'], terminal_surrogate_raw=s['mean_terminal80_surrogate_raw'],
    magnitude_gate=dict(ratio_at_most=.90, wins_at_least=24, passed=False),
    native_adjusted_candidate_result=False, current_host_convergence_bound=False,
    pins=[pin(screen / x) for x in ['PROTOCOL.md', 'RESULT.md', 'measure.py', 'screen_v1/result.json', 'screen_v1/completion.json']]
         + [pin(requal / x) for x in ['PROTOCOL.md', 'inference_v1/result.json', 'inference_watch_v1/result.json']])

j = read(J / 'summary.json')
info = read(J / 'information_result.json')
assert j['completed'] and not j['information_gate_passed']
assert not j['full_candidate_created'] and not info['actual_native_candidate_tested']
assert not info['passed'] and not info['public1B_labels_decoded']
assert [(x['ordinal'], x['ratios'][-1]) for x in j['validation']] == [(14, .9173587466627752), (71, .9311696194214288)]
expected_names = {
    'pre_mean.npy', 'pre_m2.npy', 'pre_m3.npy', 'pre_m4.npy',
    'mean.npy', 'post_m2.npy', 'post_m3.npy', 'post_m4.npy',
    'pre_M11.npy', 'pre_M21.npy', 'pre_M31.npy', 'pre_M22.npy', 'M11.npy',
    'gt_mean.npy', 'global_index.npy', 'mlp_name.npy', 'n_samples.npy', 'sample_seed.npy'}
payloads = []
for ordinal in (57, 68, 45, 59, 3, 12, 14, 71):
    download_path = OLD / f'download_mini{ordinal:04d}.json'
    target_path = J / f'targets_mini{ordinal:04d}.json'
    download = read(download_path)
    target = read(target_path)
    assert download['passed'] and target['passed']
    assert not target['late_MC_noise_removed'] and not target['public1B_labels_decoded']
    p = Path(download['payload']['path'])
    assert target['source'] == download['payload'] == pin(p)
    with zipfile.ZipFile(p) as z:
        entries = z.infolist()
        assert len(entries) == len(expected_names)
        assert {x.filename for x in entries} == expected_names
        members = [dict(name=x.filename, bytes=x.file_size, compressed_bytes=x.compress_size, crc32=x.CRC) for x in entries]
    payloads.append(dict(ordinal=ordinal, payload=download['payload'], download=pin(download_path),
                         target_receipt=pin(target_path), members=members,
                         payload_hash_reverified=True, member_headers_decoded=False,
                         array_values_decoded=False, member_streams_opened=False))

# Two symmetric scalar laws are represented by Y=X^2 and its probabilities.
# They have the same first four X moments, but different fourth-cumulant
# influence variance. This proves that these low-order aggregate statistics
# do not generally identify even the scalar sampling variance.
laws = [((0, Fraction(2, 3)), (3, Fraction(1, 3))),
        ((0, Fraction(1, 2)), (1, Fraction(1, 3)), (4, Fraction(1, 6)))]
counterexamples = []
for law in laws:
    moments = {2*k: sum(prob * y**k for y, prob in law) for k in range(1, 5)}
    influence_mean = sum(prob * (y*y - 6*y + 3) for y, prob in law)
    influence_variance = sum(prob * (y*y - 6*y + 3)**2 for y, prob in law)
    assert moments[2] == 1 and moments[4] == 3 and influence_mean == 0
    counterexamples.append(dict(squared_support=[dict(y=y, probability=str(prob)) for y, prob in law],
                                symmetry='each positive squared support splits equally between plus and minus sqrt(y)',
                                even_moments={str(k): str(v) for k, v in moments.items()},
                                odd_moments_through_three=0,
                                fourth_cumulant_influence='X^4 - 6 X^2 + 3',
                                influence_variance=str(influence_variance)))
assert [x['influence_variance'] for x in counterexamples] == ['18', '10']

report = '''# Historical training and moment-noise evidence

The older terminal80 run adds two 160-network passes to its old P1 host.
Its original training qualification remains failed. A separate inference
qualification recovered all40 outputs plus repeat from the sole fixed
terminal payload, reproducing the13 historical validation outputs exactly.
The saved-output magnitude screen compared terminal MPS predictions with
P1 native predictions on the same1B means: raw ratio0.998086518 and23/40
wins. It failed its fixed ratio<=0.90 and>=24-wins gate. This is a mixed
backend screen, not a negative native adjusted score or a convergence bound
for the current host. No current extra-epoch experiment is selected here.

All eight incoming-moment payload hashes were rechecked, and their ZIP
directories were inspected without opening member streams or decoding
arrays. They contain pooled moments through fourth order and a single
sample count/seed; they contain no sample blocks, independent replicate
moments, or sixth/eighth-order moments. The previous joint-head validation
next-mean ratios0.917359 and0.931170 remain failures of its fixed0.90 gate.
Its Gaussian first-layer and FP32 rounding checks do not quantify late-layer
Monte Carlo noise. Its own target receipts explicitly say that this noise
was not removed.

An exact rational counterexample illustrates the missing information. Two
symmetric scalar laws can both have mean0, variance1, third moment0 and
fourth moment3, yet have different eighth moments and different sampling
variances for the fourth cumulant. Here the cumulant influence function is
X^4-6X^2+3; its variance is18 for one law and10 for the other. Thus even a
scalar fourth-cumulant variance is not identified by these aggregates.
The actual transported joint consumer additionally needs covariances across
its moment entries. A late-layer Gaussian approximation would be an extra
model assumption, not measured noise correction.

This audit neither estimates that missing noise nor changes either failed
gate. Establishing it would require additional appropriate sample statistics
or independently acquired moments under a separately specified study. No
samples, numerical backend, estimator, fit, candidate, or target array ran
or changed here. The currently running downstream validation is unaffected.
'''
with (HERE / 'RESULT.md').open('x') as f:
    f.write(report)
imports = sorted({'numpy', 'torch', 'flopscope'} & set(sys.modules))
assert not imports
result = dict(utc=datetime.now(timezone.utc).isoformat(), completed=True,
              scope='historical receipts, payload hashes/ZIP directories, exact rational illustration',
              terminal80=terminal,
              moment_noise=dict(validation=j['validation'], fixed_information_gate_passed=False,
                                payloads=payloads, late_MC_noise_identified=False,
                                counterexamples=counterexamples,
                                pins=[pin(J / x) for x in ['PROTOCOL.md', 'summary.json', 'information_result.json', 'moments.py', 'targets.py', 'data.py', 'math_core.py', 'replay_result.json']]),
              new_model_selected=False, new_study_selected=False, candidate_changed=False,
              failed_gates_changed=False, estimator_calls=0, array_values_decoded=False,
              numerical_modules_imported=imports, report=pin(HERE / 'RESULT.md'), source=pin(HERE / 'audit.py'))
write(HERE / 'audit.json', result)
print(json.dumps(dict(completed=True, audit=pin(HERE / 'audit.json'), estimator_calls=0,
                      array_values_decoded=False, payloads_reverified=len(payloads))))
