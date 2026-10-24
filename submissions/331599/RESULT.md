# Fixed lazy rotation: qualified local improvement

The unchanged lazy-rotation candidate improves adjusted error on all14 panel
networks and all8 additional confirmation networks. Native/client output and
bill parity, real-client runtime, contract and package checks pass. It is now
the strongest qualified local candidate at **3.816239868909105e-9** on the
existing noise-corrected14 panel, **1.7245% below Weight5**. No upload has
occurred; the official best is still #331584 at4.015535762022727e-9. The
approximately1e-9 objective remains active and substantially unfinished.

## Accuracy and complete cost

| Fixed evaluation | Weight5 adjusted | Candidate adjusted | Candidate/Weight5 | Individual wins |
| --- | ---: | ---: | ---: | ---: |
| Existing public8, noise-corrected4M | 4.288924275919752e-9 | 4.216982029664846e-9 | .9832260395319108 | 8/8 |
| Existing fresh6, noise-corrected4M | 3.3422511334334315e-9 | 3.281916987901452e-9 | .981948051441006 | 6/6 |
| Combined existing14, noise-corrected4M | 3.883207214854186e-9 | 3.816239868909105e-9 | .9827546298098864 | 14/14 |
| Additional fixed8, ordinary1B | 3.9288648520349014e-9 | 3.8588427268781516e-9 | .9821775174779853 | 8/8 |

The additional ranks are37,39,42,48,59,63,66,67, chosen before their new runs
as the first8 entries of the old joint-prefix validation manifest. They are
unused by this component study, but not globally untouched networks. Both
estimators were rerun cold on every confirmation network. Ordinary1B and
noise-corrected4M scores are separate evaluations and are not pooled together.

The candidate's mean complete cold cost on the14 panel is398551812880.5714F,
versus405993892525.8571F for Weight5. Mean rawNC error increases0.1175%; the
compute saving more than offsets that increase. All errors were recomputed
from the actual saved predictions. No parent raw errors were reused after
changed output bytes. The initial saved35/100 development audit passed at
adjusted ratios .9812459537/.9747264525, pooled .9779885778.

The direct paired ratio to graded Cost150 projects an official score of
3.8497199235363885e-9. That is an estimate, not a graded score or a banked
sub4e-9 milestone. See OBJECTIVE_GAP.md: the same predictions would still
score about2.10e-9 on this panel at the minimum compute multiplier.

## Numerical scope and the preserved earlier failure

The exact candidate is the previous port, with no subsequent code, coefficient,
schedule, rank, precision or training change. It delays four full coefficient
rotations, retains rank384 and every QR/core update, and reconstructs the same
real-arithmetic factor expression with a different FP32 association. All rows,
signed coefficients, physical/learned consumers, kernels and Weight5 ownership
checks remain. Dense rank384 algebra and ten rejection cases passed previously.

The earlier maxabs<=1e-5 compatibility gate FAILED and remains recorded in
lazy_rotation_v1. It was an assistant-imposed numerical-identity bound, not the
user's adjusted-error objective. This separate study explicitly scored the
frozen candidate and required new broader score and runtime evidence; it did
not retroactively declare that original gate passed. The original closeout,
failed scorer and negative result remain byte-for-byte intact.

All22 networks retain exact first8 parent-layer bytes; all16 outputs are
finite nonnegative FP32. Maximum all-layer parent difference on the14 panel is
2.7768313884735107e-5 and on the8 confirmation networks3.159046173095703e-5.
All-layer equivalence to Weight5 is not claimed. Own-repeat35 and warm/cold100
remain byte exact. No threshold, candidate or checkpoint was selected from the
confirmation results.

## Actual client, contract, resources and package

The NumPy-free client35/100/35 takes40.3462281665/36.8841500822/37.9341387497s.
Each is below45s; pooled wall ratio .9990264512 meets the fixed<=1.10 gate.
This is a local saved-run comparison, not a platform timing guarantee. Exact
native/client bytes and bills match all three calls. Request counts are
60558/50624/45699; actual increases1131/680/1073 are retained in the full bills
and timings. No source-only request-count forecast substitutes for these runs.

Standard whestbench0.14.0 contract validation passes the2x4 smoke case. That
small contract check is separate from the full-width1024/depth16 native and
real-client evidence; it is not a claim of an exhaustive domain test. The
779370-byte archive validates and contains only estimator.py and manifest.json,
with exact source bytes verified against the manifest.

This objective study adds31 full predictions:12 panel,16 paired confirmation,
and3 client. Their complete billed work totals12346690237947F. It also reuses
four lazy-native calls from the previous port, for35 full evidence calls and
13912104279929F including those reused calls. These totals do not include unit,
contract or scorer setup and are not a count of all historical research work.

All35 new guarded stages exit0 and retire. Peak observed owned-group RSS is
7102414848B, with zero additional swap. Work stayed serial, nice10 and BLAS1.
Approximately6.4MB of local artifacts were added, with over459GiB free. The
earlier failed scorer remains separate and is not counted as a passed stage.

Candidate SHA256:
`b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c`.
Archive: `submission-lazy-objective-20260919.tar.gz`,779370bytes.
Archive SHA256:
`594aa6e9debf9f0e370c29afea9ccb33b475af6d27f7454bbd85abfeb8a5c8f4`.

The pinned submit.py is prepared but has not run. The prior automatic review
rejected Cost75 for missing approval for its exact private archive and AIcrowd
destination. This archive needs its own explicit approval; neither a pending
Weight5/Read5 question nor an automatic continuation grants it. All earlier
qualified archives remain unchanged, and approval for one must not silently
substitute another. approval_question.json records delivery once requested.
