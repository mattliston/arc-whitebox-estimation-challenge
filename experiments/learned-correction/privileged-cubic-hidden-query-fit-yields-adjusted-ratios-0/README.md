# Privileged cubic hidden-query fit yields adjusted ratios 0.824954528953 and 0.813778888455, above 0.80 gate

2026-09-21 · experiment · rejected · measured

The full retained third tensor was queried in 65 fixed hidden-state directions and evaluated with network-specific oracle coefficients.

Recorded measurement:

    The complete retained formal pre15 third tensor was queried along all 65
    fixed normalized directions from [1,H15]/sigma15. The parent trajectory
    |35|0.826468677698|0.824954528953|0.899360491484|0.899594901341|
    |100|0.816024210058|0.813778888455|0.882134002925|0.881797627102|
    Every value had to be <=0.80 on each network. Both networks fail both gates.
    Even these privileged per-network fits yield only about 10–12% extra raw
    error reduction beyond the affine hidden-state oracle. All 65 affine and
    130 extended directions survive the fixed numerical cutoff. There is no
    ordinary existing 1B reference means. They are not held-out validation or
    1MF readout allowance; the affine control receives the same 1MF allowance.
    The rank100 warm capture matches its independent cold control at all 16
    permutations, preservation and repeated bytes/bills. Passive35/100/35
    captures reproduce all 16 original outputs and complete cold/warm bills;
    all repeated capture arrays are exact. The formal-diagonal/native-k3
    relative differences are about 7.02e-7, below the declared 1e-3 gate.
    The formal FP64 tensor does not claim exact packed FP32 association.
    The helper costs 14,413,775,998 and 14,345,339,518 FLOPs on ranks35/100,
    under the 20GF limit. Independent wide matrix reconstruction agrees to
    9.14e-16 relative or better. All features and bills were sealed before
    coefficients within 7.74314e-15 relative and all clamped FP32 payloads
    2.83401e-16 relative. No new fit choice was made.
    Three full native calls used 1213331094586 FLOPs and
    three feature calls used 43172891514 FLOPs.
    6615334912 bytes; maximum self-reported process
    RSS was 6615367680 bytes. The short oracle

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_cubic_hidden_queries_v1/audit.py](code/terminal_cubic_hidden_queries_v1/audit.py)
- [code/terminal_cubic_hidden_queries_v1/capture.py](code/terminal_cubic_hidden_queries_v1/capture.py)
- [code/terminal_cubic_hidden_queries_v1/common.py](code/terminal_cubic_hidden_queries_v1/common.py)
- [code/terminal_cubic_hidden_queries_v1/features.py](code/terminal_cubic_hidden_queries_v1/features.py)
- [code/terminal_cubic_hidden_queries_v1/finalize.py](code/terminal_cubic_hidden_queries_v1/finalize.py)
- [code/terminal_cubic_hidden_queries_v1/helper.py](code/terminal_cubic_hidden_queries_v1/helper.py)
- [code/terminal_cubic_hidden_queries_v1/oracle.py](code/terminal_cubic_hidden_queries_v1/oracle.py)
- [code/terminal_cubic_hidden_queries_v1/prepare.py](code/terminal_cubic_hidden_queries_v1/prepare.py)
- [code/terminal_cubic_hidden_queries_v1/run.py](code/terminal_cubic_hidden_queries_v1/run.py)
- [code/terminal_cubic_hidden_queries_v1/unit.py](code/terminal_cubic_hidden_queries_v1/unit.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)
- [REEXAMINATION_INDEX_20260919.md#L740](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L740)
- [REEXAMINATION_INDEX_20260919.md#L743](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L743)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-privileged-cubic-hidden-query-fit-yields-adjusted-ratios-0) · [Copy provenance](../../../PROVENANCE.json)

