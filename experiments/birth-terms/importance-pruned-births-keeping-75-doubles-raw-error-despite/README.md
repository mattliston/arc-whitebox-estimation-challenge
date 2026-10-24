# Importance-pruned births: keeping 75% doubles raw error despite 16% less work

2026-09-22T20:59Z · experiment · rejected · measured

Births were ranked by the born tensor's Frobenius norm and the lowest-importance fraction removed before the push. The bill scales with the columns removed but the raw error doubles at a 25% cut, and a static check shows the pruned quarter still carries 13-16% of the consumed plane and diagonal at every layer, so the loss is real information, not only a broken closure.

Recorded measurement:

    keep 1.0: util .1883 / 1.648e-8 | keep .75: util .1589 / 3.25e-8 (2.0x) | keep .5: util .1236 / 1.23e-7 (7.5x); v1 importance birth*|b|^2 at .75 -> 3.76e-8; zeroing the lowest-importance 25% changes the consumed plane 13-16% and the diagonal 13-15% at every layer (50%: 37-40%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_oldtier_economics_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/birth_prune.py](code/birth_prune.py)
- [code/prune_info_loss.py](code/prune_info_loss.py)

## Evidence

- [NOTES.md#L34](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L34)
- [NOTES.md#L52](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L52)

<details>
<summary>All result files and records (4)</summary>

- [birth_prune_r35_imp.log](birth_prune_r35_imp.log) · log
- [birth_prune_r35.log](birth_prune_r35.log) · log
- [prune_info_loss_r35.log](prune_info_loss_r35.log) · log
- [NOTES.md](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-importance-pruned-births-keeping-75-doubles-raw-error-despite) · [Copy provenance](../../../PROVENANCE.json)

