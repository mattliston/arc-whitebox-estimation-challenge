# One fixed Gaussian R4 component

Prospective component study under `hill_climb_policy_20260928_v2.json`. Parent is
the retained precision/symmetry candidate1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3.
Its original TRAIN8 bank gate failed at -0.8191817203% adjusted versus S32 despite
8/8 wins. That receipt and its documented near-threshold extension remain
unchanged. This study applies the new prospective component policy, not a
retroactive pass. The core-carry composition is not included.

Exactly three numerical source replacements shorten only the Gaussian pair
series in `_nonlinear_pair_batch`, used at layers1–14, from r=1..6 to r=1..4:
`ans[c,i,j] = sum_r (rho[i,j]^r/r!) F[r,c,i] G[r,c,j]`.
The cached Fr16/Gr16/P16/P6 views become four-term views, DIV6 holds divisors1..4,
and X1 becomes shape(4,3,n,n). Names remain unchanged to minimize the patch.
One comment records the changed recurrence. Divide/cumprod actually operate on
four terms; rho5/6 are not computed. The seven-plane PB allocation remains, but
its final two planes are unused. The first Gaussian layer's separate series and
terminal scalar function are unchanged. Require L0 bytes equal the precision
parent; later predictions may change causally.

All derivative rows0..6 remain because local fourth-moment terms require them.
Plane corrections retain RR=(1,rho,rho²/2), bond order2. Local t3/t4 corrections
retain P12=(rho,rho²/2), local bond order2. Their X2/X3/T/U buffers, index maps,
FT/GT tables, marginal formulae, GN correction and all dtype choices remain
unchanged. GN's P16[0] rho consumer is still valid. X1 overlaps the same idle
arena region as before and is fully consumed before X2/X3 overwrite it. No new
model parameter, operation family, host numerical work, mask or layer choice is
introduced. Reversing the four literal edits recovers the exact parent bytes;
only `_e2_pair_state` and `_e2_pair_arena` function ASTs change.

Prior evidence is modest and from an older parent. The S4 learned-on torch audit
`fable_analyst_v1/audit_s4_v1.jsonl` tested the same6:2:2→4:2:2 series scope:
raw final MSE changes on ranks35/100/37/39 were +0.048636%, -0.062934%,
+0.001277%, -0.014377%, pooled -0.006548%. Its estimated bill saving was0.07%.
`fable_analyst_v1/RESULT.md` section36 explicitly says lower-series flags were
not rerun on S23's loop3 card. Local-order1 instead regressed about0.32% pooled;
plane bond-order0 was strongly harmful. Neither is proposed. No current native
positive accuracy result is claimed. Gaussian-tail removal can alter learned
compensation and downstream selection despite small old-parent effects.

At width1024, Gaussian products/reduction fall from51 to33 F32 n² operations
per recurrent pair call. Across14 calls this is264241152 fewer FLOPs; the unused
two power/divisor rows add roughly58720256, for a rough322961408-FLOP ceiling
(about0.1168% of the old rank35 S32 bill), plus a small one-time X1 binding change.
Complete metered bills and trajectory changes, not this estimate, determine
component value. This saving alone is not predicted to close the parent's gap
to the1% bank threshold.

Root alone runs numerical work. Native TRAIN2 ranks35,100, in that order, compare
against the matching frozen precision-parent predictions and complete bills.
Advance only for strictly positive pooled adjusted improvement versus that
parent, with all16 finite1024-element predictions, L0 exact, pinned source,
roster, full-bill integrity, and completed resource guard. Raw MSE and individual
wins are reported; there is no independent raw-error veto or pilot1% threshold.
Guard150seconds, RSS strictly below7.5GB, zero added swap, BLAS1, pinned native
NumPy2.4.6, per-network2**41 budget, entire process group retired.

Only if TRAIN2 advances, run unchanged source on fixed TRAIN6
36,45,52,58,64,70 under300seconds with the same resource/integrity rules.
Combined TRAIN8 must improve pooled adjusted score at least0.05% versus the
precision parent, win at least6/8 networks, and have bootstrap95% upper bound
belowzero (50000 resamples, seed20260926). Require L0 equality for all8. Also
report total adjusted/raw/bill changes versus S32 without conflating the two
comparators. These panels have been historically reused.

A passing component is eligible for later explicit composition, not automatic
submission. Banking still requires unchanged full VALID40 orders and S32
controls, at least1% pooled adjusted gain, at least28/40 wins, CI upper belowzero,
maximum per-network adjusted regression10%, and complete serving qualification.
The new policy reports raw error without a separate0.25% veto. All prior studies'
old gates and failures remain preserved.

No R3/R5, correction-order reduction, scope change, coefficient fit, blend, grid
or post-outcome rescue is admitted. Any failed integrity/resource/component gate
closes this fixed source. Preparation executes stdlib source/AST/JSON only and
never imports the candidate or numerical arrays.
