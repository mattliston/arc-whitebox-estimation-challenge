# Fourth birth-direction leg removal saves 0.091% work with approximately neutral pooled error

2026-09-25T08:52Z · experiment · adopted · measured

Removing the gate-derivative birth leg reduces the direction stack from four to three. Native per-network differences require pooled validation because rounding obscures the small physical effect.

Recorded measurement:

    - 2026-09-25 01:52 [engineer2] J3CUT READY FOR S25 (lead; analyst's ablation survivor, RESULT.md sec. 36): drop the birth update's gate-jet leg j = 3 (learned c3 * orth(dir3) + X3 raw add) with the full native saving.
      - Change: the direction stack goes (4,n,n) -> (3,n,n) (_e2_bd_state3), the gate-jet chain is skipped, coefficient rows 0..2 are used (the four-column node head is unchanged, so rows 0..2 are unchanged), and the S19 raw adds cover j = 0..2.
      - Directions 0..2 are computed exactly as before. The diff is 5 lines (fable_engineer2_v1/j3cut/build_j3cut.py; asserted replacements on the base's live _bd_update source).
      - Files: fragment j3cut/S23_j3cut_mem_fragment.py (4874216a); S23 + fragment = j3cut/S23_j3cut_mem.py (c81a68a8). For a new base, rebuild with build_j3cut.py <base> <out>: it fails loudly if _bd_update changed.
      - Einsum gate PASS.
      - Native parity (setup first, flopscope 0.12, one candidate per process), nets 35/100/37/39:
        - bill -283.5 M/net = -0.0913 / -0.0914 / -0.0923 / -0.0862%;
        - final-layer MSE ratio 0.99481 / 1.00378 / 1.00435 / 0.99816 (mean +0.03%);
        - max |diff| 1.0e-5..1.5e-5.
      - About -12 requests per layer.
      - CAUTION: the native per-net MSE spread (SD ~0.45%) is ~4x what the analyst's torch VALID16 interval implies (+0.002%, 90% [-0.05, +0.05]). The S25 VALID40 must confirm raw neutrality natively; expected adjusted ~-0.09% if raw stays ~0.
    - 2026-09-25 02:43 [engineer2] J3CUT per-net spread EXPLAINED (analyst side-by-side on the torch card, s6bcard_loop3 loopR1):
      | net | torch leg3 removed | native j3cut |
      | 35 | -0.074% | -0.519% |
      | 100 | +0.055% | +0.378% |
      | 37 | -0.034% | +0.435% |
      | 39 | -0.186% | -0.184% |
      Torch per-net stays within +-0.2% (VALID16 sd 0.117%). Native is larger and random-signed: the known native per-net precision noise of +-0.3-0.5% under tiny perturbations (judge pooled). Both means are ~0. My earlier CAUTION stands only as "judge it by S25's pooled VALID40", not as j3 mattering more natively.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/j3cut`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_j3cut.py](code/build_j3cut.py)
- [code/S23_j3cut_mem_fragment.py](code/S23_j3cut_mem_fragment.py)
- [code/S23_j3cut_mem.py](code/S23_j3cut_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L915](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L915)
- [TEAM_LOG_2026-09-22_to_09-26.md#L916](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L916)
- [TEAM_LOG_2026-09-22_to_09-26.md#L917](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L917)
- [TEAM_LOG_2026-09-22_to_09-26.md#L918](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L918)
- [TEAM_LOG_2026-09-22_to_09-26.md#L919](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L919)
- [TEAM_LOG_2026-09-22_to_09-26.md#L920](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L920)
- [TEAM_LOG_2026-09-22_to_09-26.md#L921](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L921)
- [TEAM_LOG_2026-09-22_to_09-26.md#L922](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L922)
- [TEAM_LOG_2026-09-22_to_09-26.md#L923](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L923)
- [TEAM_LOG_2026-09-22_to_09-26.md#L924](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L924)
- [TEAM_LOG_2026-09-22_to_09-26.md#L925](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L925)
- [TEAM_LOG_2026-09-22_to_09-26.md#L959](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L959)
- [TEAM_LOG_2026-09-22_to_09-26.md#L960](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L960)
- [TEAM_LOG_2026-09-22_to_09-26.md#L962](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L962)
- [TEAM_LOG_2026-09-22_to_09-26.md#L963](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L963)
- [TEAM_LOG_2026-09-22_to_09-26.md#L964](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L964)
- [TEAM_LOG_2026-09-22_to_09-26.md#L965](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L965)
- [TEAM_LOG_2026-09-22_to_09-26.md#L966](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L966)
- [SUBMISSION_LEDGER.md#L885](../../../records/SUBMISSION_LEDGER.md#L885)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [SUBMISSION_LEDGER.md](../../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- shipped in: [Submission #332295: third birth-term row selection, split products, frame reuse, and learned refit](../../../submissions/332295/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-fourth-birth-direction-leg-removal-saves-0-091-work) · [Copy provenance](../../../PROVENANCE.json)

