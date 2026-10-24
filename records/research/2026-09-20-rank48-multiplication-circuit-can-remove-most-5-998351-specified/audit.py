"""Exact scalar source accounting; no numerical package, array or target access."""
from fractions import Fraction
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time

HERE = Path(__file__).resolve().parent
D = HERE.parent
A = D.parent / 'astra_phase2_reexamination_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
started = time.monotonic()
seal = read(HERE / 'source_sealed.json')
for path, digest in seal['files_sha256'].items():
    assert sha(HERE / path) == digest, path
for path, digest in seal['external_sha256'].items():
    assert sha(path) == digest, path

previous_path = D / 'rank256_objective_v1/closeout.json'
previous = read(previous_path)
for path, digest in previous['artifacts_sha256'].items():
    assert sha(previous_path.parent / path) == digest, path
for path, digest in previous['external_sha256'].items():
    assert sha(path) == digest, path

def item(f):
    f = Fraction(f)
    return dict(exact=str(f), value=float(f))

records = []
for q in (1, 2, 3):
    # Coefficients multiply common leaf-block areas, not parent block areas.
    old_sum = sum(49**i * 16**(q-1-i) for i in range(q))
    new_sum = sum(48**i * 16**(q-1-i) for i in range(q))
    cob_sum = sum(47**i * 16**(q-1-i) for i in range(q))
    assert old_sum == (49**q - 16**q) // 33
    assert new_sum == (48**q - 16**q) // 32
    assert cob_sum == (47**q - 16**q) // 31
    old_linear = 77 * old_sum
    direct = [a * new_sum for a in (84, 76, 124)]
    alt = [c * cob_sum + 2 * (48**q - 47**q) for c in (83, 75, 123)]
    leaf_ratio = Fraction(48, 49)**q
    direct_ratios = [Fraction(x, old_linear) for x in direct]
    alt_ratios = [Fraction(x, old_linear) for x in alt]
    assert all(x >= leaf_ratio for x in direct_ratios + alt_ratios)
    records.append(dict(
        paired_stages=q, parent_binary_depth=2*q,
        parent_leaf_products=49**q, new_leaf_products=48**q,
        alternative_intermediate_dimension=47**q,
        parent_linear_coefficients=[old_linear]*3,
        direct_linear_coefficients=direct,
        alternative_linear_coefficients=alt,
        direct_area_ratios=[item(x) for x in direct_ratios],
        alternative_area_ratios=[item(x) for x in alt_ratios],
        componentwise_floor=item(leaf_ratio),
        maximum_saving_in_specified_arithmetic=item(1-leaf_ratio)))

dev = read(D / 'lazy_rotation_v1/dev/result.json')
assert dev['candidate_sha256'] == seal['parent_sha256']
cold, warm100, repeat = dev['records']
assert [cold['rank'], warm100['rank'], repeat['rank']] == [35, 100, 35]
assert cold['output_sha256'] == repeat['output_sha256']
assert cold['flops'] == 398080917697
assert repeat['flops'] == 377556934779
assert warm100['flops'] == 388883126034
assert cold['weights_sha256'] == repeat['weights_sha256']
# Source accounting does not re-decode any prediction or numerical truth.
floor = Fraction(48, 49)**3
hypothetical = floor * repeat['flops']
scenario = dict(
    scope='Illustration only: apply maximum component saving to ALL warm work, forgive all observed cold overhead; NOT a certified changed-estimator bound or score.',
    rank=35, cold_flops=cold['flops'], repeat_flops=repeat['flops'],
    cold_minus_repeat_flops=cold['flops']-repeat['flops'],
    identical_saved_prediction_hash=cold['output_sha256'],
    hypothetical_flops=item(hypothetical),
    hypothetical_ratio_to_cold=item(hypothetical / cold['flops']),
    hypothetical_saving_to_cold=item(1-hypothetical / cold['flops']))

result = dict(
    created_at_utc=datetime.now(timezone.utc).isoformat(),
    status='Source screen complete; no kernel or feature experiment selected',
    goal_status='active', goal_achieved=False,
    parent_sha256=seal['parent_sha256'],
    source_seal_sha256=sha(HERE / 'source_sealed.json'),
    coefficient_comparison_passed=True,
    records=records, most_favorable_arithmetic_ratio=item(floor),
    rank35_hypothetical=scenario,
    previous_closeout_sha256=sha(previous_path),
    previous_artifacts_verified=len(previous['artifacts_sha256']),
    previous_external_pins_verified=len(previous['external_sha256']),
    no_new_kernel=True, no_model_calls=True, no_array_decode=True,
    no_target_access=True, no_fit=True, no_archive=True, no_upload=True,
    scope='Specified paired-stage replacements at unchanged shapes/depth only; no universal FMM, final-mean or complete-candidate bound.',
    external_sources=['https://arxiv.org/html/2603.18699v3#S3',
                      'https://arxiv.org/html/2603.18699v3#S4'],
    elapsed_s=time.monotonic()-started,
    remaining_disk_GiB=shutil.disk_usage(D).free / 2**30)
with (HERE / 'result.json').open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))
