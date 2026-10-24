# Doubling learned width adds only about 0.08%; extra training improves forty-network error 0.377%

2026-09-25T15:31Z · experiment · adopted · measured

Matched narrow and wide training arms showed negligible width benefit. Continued training of the original model transferred to native validation.

Recorded measurement:

      CAPACITY TEST DONE (covhunter, MPS, from loopR3 on the S24 card): arm A (2 learned-only epochs) VALID16 -0.32% -> -0.463%; arm B (wide2x) -0.40% -> -0.538%. Width adds only ~0.08% (closed), but LONGER TRAINING PAYS: 2 learned-only epochs beat loopR4's single joint epoch (-0.18..-0.26%).
    2026-09-25 08:41 covhunter: ARM-A (S27 literals) torch VALID40 pre-read vs S24 torch (S24-structure card, CPU): -0.377% [CI -0.505,-0.245] 35/40 (VALID16 -0.468% 14/16 = MPS -0.463%; other24 -0.318% 21/24). Rows capA_ep1_vs_S24.json (porter parity; note S27 native adds S26's j3cut).
    2026-09-25 09:22 porter: S27 PASSES the VALID40 bar vs S24: adjusted (pooled): -1.235%  95% CI [-1.366%, -1.103%]  wins 40/40; per-net min -2.23% median -1.23% max -0.31% | raw raw (pooled MSE ratio): -0.238%  95% CI [-0.369%, -0.105%]  wins 28/40 (ties 0) | vs S26 rows: adjusted (pooled): -0.295%  95% CI [-0.422%, -0.167%]  wins 31/40; per-net min -1.34% median -0.31% ; residual gate next.
    - 2026-09-25 09:23 lead: S27 VALID40 PASS (porter; vs the S24 rows): adjusted -1.235% [CI -1.366, -1.103], 40/40; raw -0.238% [CI -0.369, -0.105] (28/40). Arm A's own effect (S27 vs the S26 rows): raw -0.296% [CI -0.421, -0.168], 31/40 (torch predicted -0.377%, per-net corr .80). Residual gate running (lock 09:22); staging ~09:30; then FIRE as today's last UTC fire. Expected ~2.343e-9 (-1.2% vs S24). Podium (2.299e-9) then needs -1.9%: S28 (loopR5 + cache caps + v5w + g5) + S29 (router).
    2026-09-25 09:23 porter: S27 VALID40 vs S24 rows PASS: adjusted -1.235% [-1.366, -1.103] 40/40, raw -0.238% [-0.369, -0.105] 28/40 (bar -1.0%); arm A alone (S27 vs S26 rows) raw -0.296% [-0.421, -0.168] 31/40 vs covhunter torch -0.377% (per-net corr .80). S27 residual gate running (took the lock first, 09:22). R320 ROUTER PREP: build_r320_router.py identity controls PASS (router all-384 == S26 bit-identical on 35/100 incl. flops; router all-320 == S26 + build_tier_rank 320 bit-identical on 35); mixed-rule probe (native resp_L10 vs the analyst's torch table, per-join picks) queued after S27's residual gate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_r320_router.py](code/build_r320_router.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1004](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1004)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1024](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1024)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1048](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1048)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1049](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1049)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1050](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1050)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [capA_ep1_vs_S24.json](capA_ep1_vs_S24.json) · result

</details>

## Related work

- shipped in: [Submission #332295: third birth-term row selection, split products, frame reuse, and learned refit](../../../submissions/332295/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-doubling-learned-width-adds-only-about-0-08-extra) · [Copy provenance](../../../PROVENANCE.json)

