# Gaussian four-cycle: an existing, narrowly matched mean witness

September 7, 2026 PDT. Phase 2 only. Source/metadata audit; no numerical
execution, array decoding, truth-file access, candidate change or fit.

## Correction to the earlier provenance conclusion

The broad statement "no matching error reference exists" was too strong.
There is an already-sealed **old official-bank rank35 direct mean13**
reference. It is not a matched covariance-innovation, complete-feedback,
retained-response-parent, rank100 or final-score witness.

The source capture ran the unchanged official P+Q bank and saved all16
served outputs. Its receipt checks those bytes against the independent
bank panel capture. The existing 1e9 truthpost file contains all16 true
post means, including layer13. Those are unconditional network targets,
not predictions from the previously truth-forced oracle trajectory.

No saved scalar result has evaluated the new four-cycle direction against
this mean error. This note identifies available information, not positive
accuracy evidence or numerical admission. The closed two-quadratic/
two-linear Gaussian R1 test remains closed; its target-free coordinate
selection can be identified without re-running or tuning that test.

## Exact provenance and access scope

All paths below are relative to this note's directory unless prefixed by
`../`. The following SOURCE/JSON files were read or hashed in this audit:

| Source or metadata file | SHA256 |
|---|---|
| `check_gaussian_fourth_r1_source_v1.py` | `e88da777239fee2b03a0c74d84ef772eaf51d3d76fce55a5f026f6c878ce40f9` |
| `check_response_rows_adjusted_v1.py` | `8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304` |
| `gaussian_fourth_r1_source_test_v1/result.json` | `f08efe2a6367085ce0bf7db6a964d38d61d56fb1638e8ca15e379c14af29db0f` |
| `gaussian_fourth_r1_source_test_v1/capture_sealed.json` | `d079f5cd686e10d2183aaf1f43607370741847f934adf0cb14adf53084df8769` |
| `gaussian_fourth_r1_source_test_v1/bank_bill.json` | `b05fa3d564c1ae23e6d7e1b150212e251c3f9a7fd9cf3359f6bd943b22c75548` |
| `response_rows_adjusted_reevaluation_v1/score/result.json` | `a5e4a8aa64968218931d3b11aa80141580371ec17931f7a1532b7381ab5437d0` |

The following hashes are COPIED FROM THOSE RECEIPTS, not recomputed here.
No array, weight or truth bytes were opened or hashed for this note:

| Recorded artifact | Recorded SHA256 |
|---|---|
| `candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py` | `3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757` |
| `gaussian_fourth_r1_source_test_v1/causal_inputs.npz` | `2143d18e35f7bdf06101b718d16df13ed3d5c0ce68d6d3f3dff04929870265a0` |
| `gaussian_fourth_r1_source_test_v1/proxy_values.npz` | `fb1d82abb8433e862abb66c979eedfadad6820bd284332e83c971a15c0c070c7` |
| `gaussian_fourth_r1_source_test_v1/bank_output.npy` | `43f66a6fbb1cd0a362a8d2b1251235322a53994650be2da3c0262c53ad26a5b5` |
| `response_rows_panel_v1/bank/call_00_rank35_output.npy` | `43f66a6fbb1cd0a362a8d2b1251235322a53994650be2da3c0262c53ad26a5b5` |
| `../public_g22_operator_features_v1/truthpost/truthpost_rank0035.npz` | `ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61` |

The capture receipt binds rank35, MLP seed2758000291903877946 and weight
digest `14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2`.
The capture source explicitly requires all16 served byte parity, restores
its wrappers and verifies the unchanged P/Q literals and causal call order.

## One specified direct marginal direction, not a full carrier

Use the already-fixed unit direction `u` in `proxy_values.npz`, its
Rayleigh value `proxy_scalars.theta` from the result JSON, and the original
captured support/response values. No new power iterations, directions,
selection criterion, source-strength fit or rank choice are needed.

For the selected covariance component C1 = theta*u*u^T, define

    dA  = alive12 * pdf12 / sigma12
    f   = theta * dA * u^2
    a   = W13^T f
    rho = 9 * (sum_i f_i^2)^2 / bank_Q

Here `bank_Q` is the receipt's literal Q(M) = trace(M)^2 + 2*sum(M*M),
where M is the actual post12 P-corrected/masked covariance entering the
bank's covariance call13. The positive source is exactly 3*f^tensor4 for
the selected rank-one Gaussian four-quadratic cycle. The scalar rho is its
natural trace subtraction under the BANK CONVENTION, not a claim that
the slightly asymmetric M defines an exact symmetric Gaussian tensor.

For the prospective approximate radial-reset completion, the direct
pre13 marginal fourth change and the bank's literal marginal response are

    delta_k4 = 3*a^4 - rho*pre_v13^2
    omega    = (alpha13^2 - 1)*pdf13 / (24*pre_v13*sigma13)
    delta_m13 = omega * delta_k4

`omega` is also already sealed in `proxy_values.npz`. Its denominator uses
the ORIGINAL pre variance, not sigma^2 if the sigma floor were active.
All remaining vector inputs and W13 are in `causal_inputs.npz`.

This choice uses the bank's marginal radial reader `pre_v13^2`. It is NOT
the exact transport of a full negative countermetric, which would involve
the quadratic form q_j = w_j^T M w_j. The earlier source test explicitly
recorded that q differs from the rounded compact pre variance. This note
does not test or select between the two conventions.

Let e be the sealed served bank mean13, promoted to FP64, minus the
existing true postmean13. The quantities e dot delta_m13 and
mean((e+delta_m13)^2) give a precisely identified DIRECT mean-direction
diagnostic in that existing reference frame. They have not been computed.
This does not recover the bank's unsaved unclipped FP64 post-head value,
so it is not an exact finite-strength, rounded/clamped output replay.
The existing Monte Carlo reference uncertainty also remains applicable.

## Complete acquisition scope and unresolved information

For only this diagnostic there is no new network or Monte Carlo
acquisition. Reusing the recorded u, theta and bank_Q leaves one FP64
1024-by-1024 matrix-vector product plus O(1024) arithmetic: approximately
4,194,304 metered FLOPs under the current FP64 charge, plus those vector
operations. This is a static subtotal, not a newly measured bill. File
verification, IO and reference comparisons must also be accounted for.

The direction omits changed P features/outputs, K31/K22 covariance reads,
third-order descendants, radial evolution and later layers. It therefore
cannot establish a final raw-times-compute improvement or exclude an
entire multi-layer cycle carrier. A pass would not automatically authorize
a carrier, coefficient/head fit, another layer/rank, or an adjusted-score
claim.

No equivalent exact native rank100 source capture, retained-response-P
source capture, or complete same-state covariance/feedback witness has
been established by this audit. Those need additional observation-only
state capture from the actually selected parent, with all provenance and
parity qualifications. As an ACQUISITION REFERENCE ONLY, the already
completed old-bank rank35 capture charged490,466,018,197 FLOPs, took
21.0457798336s in its native budget context and peaked at6,117,900,288B.
Those measured figures are not a new run or a full-cycle implementation
cost forecast.

No numerical check is admitted by this note. In particular, no truth
file may be opened during the currently active kernel acquisitions; their
outputs and complete bills must seal under their own protocol first.
Any later direct-direction check requires separate reviewed authority and
a source-only prediction seal before truth access. Existing closeouts,
frozen artifacts, and the prohibition on parameter/source sweeps remain.
