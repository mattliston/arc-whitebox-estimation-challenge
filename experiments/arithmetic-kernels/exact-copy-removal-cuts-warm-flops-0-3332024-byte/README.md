# Exact copy removal cuts warm FLOPs 0.3332024% with byte-identical predictions on network 35

2026-10-20T19:22:39Z · experiment · adopted · measured

Expanded exact copy removal reduced cold and warm FLOPs while preserving prediction bytes. It increased requests, so full-stack runtime qualification remained necessary.

Recorded measurement:

    COPY_ALL_EXACT native rank35 PASS: D/candidate_copy_all_exact_sci_v1.py acda0d7766f7b04327af4c3814b3f1edf3c49fcd8b660000152e5b16d7ec1104, releaseaf622a45. Same predictionSHA cold/warm asv4a; adjusted/bill cold-.3220965%,warm-.3332024%, setup unchanged27.687379GF; warmrequests+485, concurrent backend+.258s/wall-.182s (not controlled timing). GuardPASS7.016GBRSS/4.064GBfoot/free>=54,swapflat. D/COPY_ALL_EXACT_NATIVE_COMPARE_v1.json. Tail includes materialize+extra-flatload+EXACT NP copying, no arithmetic reassociation. Replace prior materialize tail entirely when stacking; never append both.
    EVIDENCE: Gram fresh4 paired adjusted-.33155% SE.1701; full32 currently~19/32 (first7-.1468%). Joint earlyGSK/GK4 both-10% (.81/.936 atL1–7) fresh4-.189301% SE.2671,3/4lower; pair all32 started~19:04, source458107d9, one cache-safe wrapper. Other live early dials all positive, edge10+.0796%, rejected. KC late_m verifiedall32-.0843%,held25-.0962%. Exact copy removal D/candidate_materialize_sci_v1.py3e011d1d measuredrank35 same predictionSHA cold/warm asv4a, coldadjusted-.241514%, warm-.252389%, setupbill unchanged27.687379GF, requests+267/+243; guardPASS. D/MATERIALIZE_NATIVE_COMPARE_v1.json. Fullstack must independentlyclear <=-.35% all32 plusheldnegative, then original release_min gates/fire_once. No governors/safe twins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_copy_all_exact_sci_v1.py](code/candidate_copy_all_exact_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6246](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6246)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6236](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6236)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [COPY_ALL_EXACT_NATIVE_COMPARE_v1.json](COPY_ALL_EXACT_NATIVE_COMPARE_v1.json) · result
- [MATERIALIZE_NATIVE_COMPARE_v1.json](MATERIALIZE_NATIVE_COMPARE_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337396: Mean and carrier precision with copy removal; two residual failures](../../../submissions/337396/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-exact-copy-removal-cuts-warm-flops-0-3332024-byte) · [Copy provenance](../../../PROVENANCE.json)

