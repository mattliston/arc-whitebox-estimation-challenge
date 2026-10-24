# Rank-320 package fails forty-network gate: adjusted -0.343% with confidence interval crossing zero

2026-09-25T09:27Z · experiment · rejected · measured

Matching prebind keys recovered the native rank-cut saving, but raw error rose by essentially the same amount. Attribution preserved only the small exact-reuse and birth-leg savings.

Recorded measurement:

      R320 native census (porter): bill -4.29% (-4.18/-4.40), requests +1,193/+373 because S23's static prebind list holds 384-wide keys and 320-wide products bind lazily. Measuring whether a matched key list clears -4.5%.
      S25 (S24 + treuse + j3, ~-0.18%) is on HOLD until it carries >= ~0.5% (R320 with a matched list; the weight-average probe avg(loopR3, loopR4), physicist ~20 min).
    - 2026-09-25 02:29 lead: R320 native (porter): bill -4.37..-4.47% in process (the static prebind still saves 4.9 of its 5.8 GF; a matched list adds only ~-0.1..-0.2%); native 4-net raw +0.05/+2.79/+4.66/+4.69% (mean +3.05%; torch VALID16 +4.04%). Adjusted between -0.4% and -1.4%: only a VALID40 settles it. OVERRIDE the -4.5% trigger: GO S25 = S24 + treuse + j3cut + R320 (matched prebind list), VALID40 vs the S24 rows, bar adj <= -0.5%, CI excl 0. If it fails, S25 stays held.
    2026-09-25 02:31 porter: R320 census (S23, old-tier rank 384 -> 320, build_tier_rank.py; R384 control = S23 exactly): bill -4.18% (r35) / -4.40% (r100), pooled -4.29% with S23's static prebind list (requests +1,193 / +373: the 320-wide keys bind lazily); in process without setup -4.37/-4.41/-4.47/-4.39% (35/100/37/39), so a matched key list adds only ~-0.1..-0.2%. Native MSE +0.05/+2.79/+4.66/+4.69% (mean +3.05%; torch VALID16 +4.04%). Trigger (bill <= -4.5%) NOT met -> no VALID40, R320 out of S25. S25 READY: fire_chain_S25.sh [BAR] [DIALS|none] = S24 + treuse_fragment_v2 (75aa9233; S24+it = the engineer's 32babb65 exactly) + j3cut (4874216a; disjoint names; base _bd_update body = S23's) -> treuse identity with setup() -> einsum S16/S23/S24 -> smoke -> VALID40 vs the S24 rows -> residual vs S24 -> fire_candS25. Waiting for the physicist's c0 result.
    2026-09-25 02:37 porter: S25 (lead GO: S24 + treuse + j3cut + R320 with a MATCHED prebind key list) in build. Key-list method: S24's live _PB_SINGLE is engineer2's d6x re-derivation (not v2b); its rule = single keys used by >= 2 of the 8 selection nets (300..580), ordered (-nets, -calls, key), no cap needed (<=160/side); reproduced EXACTLY (359 keys, same order) from fable_engineer2_v1/d6x/ipt_key_census_S18d6x_sel8test4.json (v2b's rule, -nets then key, cap 121, also reproduced exactly). R320 key census (ipt_key_census_porter.py = the engineer's census, output in my lane) running on S24 + j3 + R320 over 300..580 + 35/100/37/39. j3cut rebuilt on S24 with build_j3cut.py = engineer2's fragment byte-identical (4874216a). Order per the engineer: j3 + R320 + key list first, treuse LAST; treuse identity by fable_engineer_v1/e73_parity.py (setup, 35/100/37/39). Chain: fire_chain_S25r.sh (bar -0.5%; on FAIL an attribution VALID40 of S24 + j3 + treuse).
    2026-09-25 02:49 porter: S25 (163b51fc) built = S24 + j3cut + R320 (3 edits) + matched key list + treuse last; treuse identity PASS (35/100/37/39); einsum/validate/literals PASS; smoke next.
    2026-09-25 02:50 porter: S25 smoke OK (wall 1.608 s at load { 2.72 4.35 4.87 }; bound 353 entries); VALID40 vs S24 rows STARTED (bar adj <= -0.5%, CI excl 0).
    2026-09-25 02:57 porter: S25 FAILS the VALID40 bar (adj <= -0.5% vs S24, CI excl 0): adjusted (pooled): -0.343%  95% CI [-1.248%, +0.528%]  wins 21/40; per-net min -5.76% median -0.08% max +5.63%; running the attribution VALID40 (S24 + j3 + treuse, no R320).
    2026-09-25 03:04 porter: S25 attribution vs S24 rows: j3+treuse adjusted (pooled): -0.136%  95% CI [-0.235%, -0.033%]  wins 31/40; per-net min -1.01% median -0.09% max +0.95% | R320 part (package vs j3+treuse) adjusted (pooled): -0.207%  95% CI [-1.095%, +0.653%]  wins 19/40; per-net min -5.38% median +0.36% max +5.80%. S25 NOT staged (held).
    - 2026-09-25 03:08 lead: S25 FAILS its bar and is HELD (porter; VALID40 vs the S24 rows). S24 + j3 + R320 + treuse: adjusted -0.343% [CI -1.248, +0.528], 21/40.
      Attribution:
      - WITHOUT R320 (j3 + treuse): adjusted -0.136% [CI -0.235, -0.033], 31/40; raw +0.043%.
      - WITH R320: raw +4.41% [CI +3.48, +5.30] vs util -4.41% = adjusted neutral.
      R320 is CLOSED (the raw cost equals the bill cut, as on S11). S26 = S24 + j3 + k-split v2 + (group 5 optional) + treuse last; expected ~-1.1% vs S24. It is the last fire of UTC 09-25 (or later).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/build_tier_rank.py](code/fable_porter_v1/build_tier_rank.py)
- [code/fable_porter_v1/fire_chain_S25.sh](code/fable_porter_v1/fire_chain_S25.sh)
- [code/fable_porter_v1/ipt_key_census_porter.py](code/fable_porter_v1/ipt_key_census_porter.py)
- [code/fable_engineer2_v1/j3cut/build_j3cut.py](code/fable_engineer2_v1/j3cut/build_j3cut.py)
- [code/fable_engineer_v1/e73_parity.py](code/fable_engineer_v1/e73_parity.py)
- [code/fable_porter_v1/fire_chain_S25r.sh](code/fable_porter_v1/fire_chain_S25r.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L949](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L949)
- [TEAM_LOG_2026-09-22_to_09-26.md#L950](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L950)
- [TEAM_LOG_2026-09-22_to_09-26.md#L951](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L951)
- [TEAM_LOG_2026-09-22_to_09-26.md#L952](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L952)
- [TEAM_LOG_2026-09-22_to_09-26.md#L955](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L955)
- [TEAM_LOG_2026-09-22_to_09-26.md#L972](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L972)
- [TEAM_LOG_2026-09-22_to_09-26.md#L973](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L973)
- [TEAM_LOG_2026-09-22_to_09-26.md#L974](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L974)
- [TEAM_LOG_2026-09-22_to_09-26.md#L975](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L975)
- [TEAM_LOG_2026-09-22_to_09-26.md#L979](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L979)
- [TEAM_LOG_2026-09-22_to_09-26.md#L980](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L980)
- [TEAM_LOG_2026-09-22_to_09-26.md#L981](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L981)
- [TEAM_LOG_2026-09-22_to_09-26.md#L982](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L982)
- [TEAM_LOG_2026-09-22_to_09-26.md#L983](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L983)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [ipt_key_census_S18d6x_sel8test4.json](ipt_key_census_S18d6x_sel8test4.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-rank-320-package-fails-forty-network-gate-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

