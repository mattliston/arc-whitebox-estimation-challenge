# Return to the demonstrated hill-climbing workflow

User steering: study and follow the successful Fable/Opus incremental improvement process. Derivative/JVP V11 remains unfinished and is not the next task. No new numerical experiment or upload ran during this review.

The submission ledger and Fable TEAM_LOG show small complementary changes tested on the working card, combined, measured on native adjusted score, and submitted after qualification. S28 accepted +0.839% raw error because -2.608% compute yielded -1.788% adjusted. The S30 planning log retained components of about0.15% each. These are empirical decisions, not an argument that every proposed change must work.

A specific mismatch in our recent process is team_score_screen_v1.py: an individual arm must improve TRAIN2 by1% to advance; TRAIN8 must improve2%. The user's couple-percent requirement applies to a submitted bank. It does not require every useful component to clear that threshold individually. Preserve historical failed-gate records; do not rewrite them as successes. For new work, retain credible small components for combination and validate their actual combined result. Do not add overlapping improvements or loosen final submission qualification.

Concrete existing evidence from metadata only:
- Thin star route: TRAIN8 adjusted -0.57346357%,8/8wins,95%CI[-0.73317%,-0.41389%]. Existing candidate_thin_star_route_v1.py.
- late_common256: TRAIN2 adjusted -0.75183519%,2/2wins,CI[-1.11482%,-0.36428%]; closed by the1% advancement gate. Existing teammate_recovery_20260927_v1/late_common256.py. Its manifest shows it ALREADY INCLUDES Thin. These gains must not be added. It has no six-network follow-up or validation in that result record.
- Several other sub1% records have mixed wins or intervals crossing zero, or share a mechanism. They are not established components.
- Fable's deferred4096-wide depth7 packets save projected runtime/request count at unchanged FLOPs and add~331MB memory. They are not evidence of adjusted-score improvement; do not prioritize them as a bank.

Next concrete step: inspect the unchanged late_common256 patch and its existing guards, then, if resource-safe, precommit one existing six-network TRAIN follow-up against the matched S31 baseline to decide component retention. No coefficient/direction sweep, new gradient infrastructure, automatic retries, VALID access or upload follows from this audit. Keep the final >=couple-percent bank requirement. Any next experiment should have a short fixed execution budget and stop condition.

Last confirmed bank remains S31#332517 adjusted2.1736465056908115e-9. This review is a strategy correction, not a bank or a score gain.

## First bounded follow-up results

- Retain late_common256 as the hill-climb parent: TRAIN8 adjusted -0.996111% versus S31,8/8wins,paired bootstrap95%CI[-1.32661%,-0.67454%]. Incremental versus Thin is -0.425085%,7/8wins,CI[-0.82760%,-0.05768%]. Full receipt: hill_climb_late_common256_v1/result.json.
- Reject adjacent unchanged late_common192 after its six-network extension: TRAIN8 -0.552965% versus S31,5/8wins,CIcrosseszero; +0.447604% versus retained256,3/8wins. Full receipt: hill_climb_late_common192_v1/result.json. No further rank sweep follows.
- Explicit covariance symmetry was actually +0.0796% worse than Thin on TRAIN2, and training convention improved only0.0227% with mixed wins. Neither is an evidenced complementary gain; do not add their S31-relative scores to the new parent.
- Both new guards retired with zero added swap; peak RSS5.870/5.905GB and elapsed147.46/144.29s. No VALID,worker,archive,submission or new bank. Final couple-percent release requirement stays unchanged.
- DerivativeV11 stays parked. The improvement above is a retained training-screen component, not a held-out or leaderboard claim.

Bounded next-lead lookup (source only; not admitted or run): no direct current-S31/Thin native receipt found for low-rank-star K32/R32/Q1 or K32/R=None/Q1. Old S18 K40/R32/Q1 Gaussian sketch was approximately adjusted-flat (raw+0.41%,bill-0.36%); old K32 used no power iteration. These do not establish current deterministic-column K32/Q1 performance. If explored later, use one fixed candidate on retained common256 with native scoring; current _TSR_SHAPES lacks (32,1024,1024), so blindly setting K32 would lose the Thin route for Q.T@A. Source fit checks and a measured complete bill would be needed. No claim that this lead yields a gain.
All teammate tasks now finished/interrupted; no numerical jobs live.
