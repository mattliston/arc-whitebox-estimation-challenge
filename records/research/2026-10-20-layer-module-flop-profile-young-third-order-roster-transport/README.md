# Layer/module FLOP profile: young third-order roster transport costs 32.6%, old-tier reconstruction 10.4%

2026-10-20T14:35:03Z · finding · informational · measured

The profile located about half of all FLOPs before layer 8 despite only about 20% of final mean error arising there. The young roster and old-tier reconstruction dominated the individually wrapped modules.

Recorded measurement:

    - [fable-lead] 2026-10-20T14:35:03Z PER-LAYER / PER-MODULE BILL PROFILE of #337259 (probe_layer_bill_v1.py, fresh 7100005, in process; lead/runs/fable_oct19_kc_v1/layer_bill_7100005_v1.log): layers 3-13 cost 6.9-8.5% of the bill EACH (L1 2.6%, L2 4.1%, L14 4.3% + terminal), i.e. L1-8 = ~50% of the bill while the post-mean oracle puts only ~20% of the final error before L8. Modules: young roster pushes (_push_bank_shared_newborn) 32.6% (1.0 GF at L1 rising to 9.3 GF at L6, 3-6 GF at L7-14), old-tier reconstruction (_slr_reconstruct) 10.4% (2.0-3.9 GF per layer L7-14), GNN 1.0%, message edge 0.7% (L8-14), NP 0.3%, K4L 0.2%, GN dcov 0.1%, F4C 0.02%; the covariance/K21 chain, reads and squares are the unwrapped remainder (~55%). JOIN FRAME is complete: at the L10 join the rank-384 frame holds 99.995% of the joined carrier energy (JDIAG debug), so the join's MSE cost is orientation (the P G (I-P) cross term), not energy. COMMON-MODE DEFICIT (dcov_structure_v2.py, cmode_fit_v1.py on nets 35/100, L8-14): the top eigenvector of dC = Cpre_card - Cpre_true carries 18-27% of |dC|^2 at every late layer, is ALWAYS negative (card too small) and has |cos| 0.72-0.85 with the pre-activation mean mu (52-68% inside the card's top Cpre eigenvector); but a rank-1 correction along mu itself explains only 5-9% (net 35) / 0.3-3.5% (net 100) of |dC|^2 with a coefficient c_dim = c|mu|^2/mean(v) of -0.05..-0.24 (not universal): a zero-bill arm (_CMODE: Cpre += c mean(v)/|mu|^2 mu mu^T at L8-14) is worth an in-process test when the lanes free the box, expectation <= -1% MSE. Per-net offset/scale of the final residual: 0.5% / 1% of the MSE, unpredictable (pernet_offset_v1.py, 129 store nets) = DEAD.

## Programs

- [probe_layer_bill_v1.py](probe_layer_bill_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6185](../../TEAM_LOG_2026-10-01_to_10-22.md#L6185)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [layer_bill_7100005_v1.log](layer_bill_7100005_v1.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-layer-module-flop-profile-young-third-order-roster-transport) · [Copy provenance](../../../PROVENANCE.json)

