# Phase 2 receipts — actual September 4, 2026, 19:55 PDT

Dates checked against the UTC clock and filesystem modification times.
Phase2 only; no deletion, frozen-file overwrite, commit, promotion, or
upload. Goal approximately4e-9 adjusted remains unmet. Upload approval is
still outstanding after the earlier automatic review rejection.

## Best completely validated and locally packaged result

`candidate_fullpair_scalar_final_batched_d6_v1.py`, SHA256
`b95face842cee345eeee5393c49eee7c718c482fa9930ffeb22d1974da7d5120`.

Independent4M noise-corrected14-net panel:

| Group | Mean raw MSE | Mean utilization | Mean adjusted |
| --- | ---: | ---: | ---: |
| Public8 | 2.45023949397e-8 | .21895 | 5.37165714446e-9 |
| Virgin6 | 2.00427699535e-8 | .217366667 | 4.37040032467e-9 |
| All14 | 2.25911270885e-8 | .218271429 | 4.94254707884e-9 |

These are LOCAL scores, not official leaderboard results. Official best
remains329790, adjusted1.43699341254e-8.

The preceding source/full-pair learned model, before scalar-final savings,
scored public5.530e-9 /virgin4.532e-9 on identical panels. Thus this step
saves about2.9% /3.6% adjusted. The earlier compact scalar-final model was
public5.8500e-9 /virgin5.0310e-9. The fixed source/full-pair training had
heldout one-step5.703532e-9, and full-loop development raw2.04444475e-8 /
2.00406279e-8 against stored organizer1e9-sample means onr35/r100.

The matched final-only fit uses its own frozen prefix,200nets/fixed160/40,
40epochs. Heldout raw2.24756231e-8 versus the original final head2.23118555e-8
(+0.73%). Trace-aware scalar reference includes
`k3 = CPdiag + stardiag + 3*diag(Bpre)*bpre`, and separates active covariance
from cold diagonal variance. Full-loop development raw2.07256816e-8 /
2.05830925e-8. Frozen full-matrix host means match within9.91e-8 and final
inputs within8.35e-6. No current-layer target feedback enters any input.

Native official-version35/100/35 INCLUDING both heads passes; repeated
outputs byte-identical; max mean1.632e-6, max final-delta6.0e-8, peak7.324GB.
Actual numpy-free remote gate: utilization.218223239, wall66.27s,
residual.23575527s, peak6.654GB. Whest.16 tiny plumbing validation passes.
Local archive`fullpair_scalar_final_batched_d6_v1_whest016.tar.gz` contains
onlyestimator.py+manifest.json. Its estimator is byte-identical to tested
source. Bound evidence:`verified_local_fullpair_scalar_final_v1.json`.

## Exact pair batching: completed and composed above

The three pair fields share a leading batch dimension. Two small algebra
tests are byte-identical. Actual full-size remote output hash matches the
unbatched model byte-for-byte. Residual measured.2952128s versus.3812354s
unbatched; FLOPs499109452279 versus499812555255. Different concurrent loads
mean wall-time differences are not a controlled benchmark.

## Third-tensor directional features: bounded branch closed

Nine exact contractions along two covariance directions, appended to the
old30features. Dense small-tensor certificate passed. All200 honest row
host predictions were byte-identical to their original compact host.
Fixed160/40,C64,40epoch fit: heldout6.315394e-9 versus old6.393391e-9.
BUT full-loop development raw2.39536514e-8 /2.33634216e-8 versus old compact
2.29298372e-8 /2.34139424e-8: +4.5% /-0.2%, not a clear gain. No learned
port, panel, or further training of this form. Files`compact_tensorfeat_*`,
`payload_compact_tensorfeat_v1.npz`, `rows_compact_tensorfeat_v1_receipt.json`.

## Quadratic third-cumulant density completion: matched round running

The new independent map omitted the familiar`L3**2/2`term already owned by
historicalD11. This is NOT a new connected sixth cumulant or a reopening of
the old D11 sentinel. Independent polynomial convolution verifies all seven
degree-six allocations and marginal-product subtraction. At zero third
cumulant the first-order parent is reproduced byte-for-byte. The batched
deployment formula agrees within3.5e-18 in the small tests.

Current source/full-pair unlearned development raw2.46413140e-8 /
2.69883751e-8; completed map2.40829242e-8 /2.41372115e-8 (-2.3% /-10.6%).
Metered r35 implementation raw2.41035531e-8, utilization.227317580 versus
parentunlearned.224338957: added cost~1.3%. No unmatched learned-head gain
is claimed.

`chain_compact_secondorder_training_v1.py` live session94433. Three nice10
row workers,200rows,thenONE MPS fixed160/40,C64,40epochs. Latest poll190/200
rows. Expected`payload_compact_secondorder_v1.npz`,
`chain_compact_secondorder_training_v1.json`, G22 tag
`ASTRA_COMPACT_SECONDORDER_V1_C64_E40`. Before training a profile guard
checks late one-step errors do not collapse spuriously.

Then use`screen_compact_secondorder_closedloop_v1.py --rank35/100
--payload payload_compact_secondorder_v1.npz --tag UNIQUE`. If successful,
attach matched heads to`candidate_compact_secondorder_unlearned_d6_v1.py`
with`build_compact_learned_v1.py`. The scalar-final map for this family MUST
also include the new univariate second-order terms and require a new
matched final-only fit; do not reuse the first-order final head.

## Exact square schedule audit: cost gain, margin work in progress

`price_direct_kernel_schedules_v1.py` identifies the fixed schedule's
kernel-only cost polynomial from four small shapes and verifies it on a
fifth, for depths3..7. Float32 relative arithmetic errors were recorded.
The current D3 square schedule leaves~13–22% kernel cost available on
representative512..1024shapes. D6 rectangles are already close to their
measured minimum; D7 was not blindly deployed.

`candidate_fullpair_scalar_final_squareopt_v1.py` chooses square schedules
from those costs, reusing the SAME447469568-element arena. No mask,
moment-map, or learned-parameter change. Full-size r35 raw2.07072177e-8
(effectively unchanged), utilization.208840234 versus.21814. Native
35/100/35 passed: max mean4.249e-6, final delta6.494e-8, byte-identical
repeated output, peak7.197GB. Actual remote utilization.208921821,
wall92.83s,residual.386981586,peak6.726GB: technically passes but too near
the residual limit to call comfortable.

Setup/dispatch-only follow-up:
`candidate_fullpair_scalar_final_squarecache_v1.py` lazily binds square
views and uses shape-only lookup tables for common shapes. Its choices
match the original dispatcher for ALL32768 /32 shapes through1024.
Live actual remote gate session67461, tag
`fullpair_scalar_final_squarecache_remote_v1`. Numerical output should be
byte-identical to squareopt remote hash
`2619439d29c631181f07a4163ea4e807685bd4d88297775a88dda15eb731b1a7`;
check rather than assume. Cached version still needs its own repeated-call
gate and full14-net panel if resources are satisfactory.

## Newborn-only pruning: bounded branch closed

Prior stronger pruning changed carried-state support. The isolated test
`radial_compact_birthonly_v1.py` leaves active/carried support at.01 and
changes ONLY newborn selection to.05, restoring omitted current repeated
entries to the star. No threshold sweep. Development raw3.33227876e-8 /
3.41314838e-8 versus2.46413140e-8 /2.69883751e-8 (+35% /+26%). No port or
coadaptation justified. The selected beta clipping count was zero for every
layer on both nets in THIS test; that observation is not a universal proof
about every candidate trajectory.

Last checked swap6458.44MiB, stable. No MPS job was active at that poll;
the second-order chain owns the next MPS slot after its rows complete.
No upload permission has arrived. Continue safe local research; do not
mark the goal complete or blocked while meaningful local progress remains.
