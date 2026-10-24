# Candidate parity bug: exported variance head was active although the instrument had disabled it

2026-09-04 · incident · failed · measured

FIX VALIDATED: rebuilt with BUILD_VHEAD_MIX=0, cand_gnn_lean2_v2 and cand_gnn_hcap256_v2 reproduce the instrument on r35 EXACTLY (3.006e-8 / 2.424e-8, ratio 1.000). cand_gnn_hnq_v2 4.44e-8 vs 3.32e-8 (ratio 1.34): the HNQ instrument run had another flag difference (checking). The lean family is now packageable; CAP256 = -6..-10% adjusted vs the bank (not a meaningful fire on its own).

Recorded measurement:

    - T-130 PACKAGING BUG localized (LEAN-2 candidate vs instrument, r35): with the NULL closure the candidate and the instrument agree to 1e-16 at every layer (card physics under CP4_CAP=1 + no-quartic is identical; both 3.11e-7 at util 0.441). With the closure ON: per-layer deltas agree (corr 1.0000, rms ratio 0.999-1.000) but the step INPUTS differ by up to 5e-3 relative from L2 (cols 3-15 = raw20 moments, 20-50 = mech) and the STATES diverge (cand vs instrument post-mean mse 1.1e-8 at L4, 1.1e-7 at L14; final 1.58e-7 vs the instrument's ~4e-8). So the bug is in the closure's INPUT features or the feedback under the lean flags, not the card. The organizers' K=3 AUGMENT and K=4 f64 runs died silently (likely OOM); skipped.
    
    - T-130 PACKAGING BUG FOUND: every lean payload (payload_gnn_hlean2 / hcap256 / hnq, HV lineage) carries a VARIANCE HEAD (msgv_w, vrms). The instrument applies the head only under RETRO_CL_VHEAD=1 (else W_next=None -> head off) and the lean runs never set it; the builder baked _VHEAD_MIX = 0.5 by default and the candidate always sets W_next -> the candidates applied a half-strength variance head the instrument never used. Proof chain: null closure identical to 1e-16; with closure: post mean/var/k3/k4/gate/covariance after L1 identical (covariance to 2.6e-8), L2 pre_mean and pre_k3 identical, L2 pre_variance differs 4.7e-4 = the v-head's diagonal write. Fix: BUILD_VHEAD_MIX default 0.0 (build_gnn_candidate.py). The cap256 r100 5.1e-8 and lean2 1.33e-7 "divergences" were this.
    
    - T-130b cheap-card economics (instrument, 1e9 frame, r35/r100, adjusted = raw x util): full HF1 loop 2.41/2.58e-8 x 0.622 = 1.50/1.60e-8; CAP256 2.42/2.70e-8 x 0.554 = 1.34/1.50e-8 (-10%/-6%); LEAN-2 3.01/3.38e-8 x 0.462 = 1.39/1.56e-8 (-7%/-3%); NQ 3.32/3.41e-8 x 0.463 = 1.54/1.58e-8 (0%). Even with the packaging fixed, the cheap-card family is a <=10% bank, not the 2x "competitive" needs (rank 10 = 9.4e-9 adjusted; bank 1.78e-8). NEXT LEVER TO TEST (zero billed): PRECISION - the organizers' K=3 lost 10x in float32; our CP3 bank is fp32 ("full-birth fp32 CP3 state"); test the bank in fp64 (RETRO_CP3_F64) on r35: if the host drops well below 7.85e-8, precision is a real raw lever and the cheapest place to spend fp64 is the question.
    
    - T-130c FIX VALIDATED: rebuilt with BUILD_VHEAD_MIX=0, cand_gnn_lean2_v2 and cand_gnn_hcap256_v2 reproduce the instrument on r35 EXACTLY (3.006e-8 / 2.424e-8, ratio 1.000). cand_gnn_hnq_v2 4.44e-8 vs 3.32e-8 (ratio 1.34): the HNQ instrument run had another flag difference (checking). The lean family is now packageable; CAP256 = -6..-10% adjusted vs the bank (not a meaningful fire on its own).

## Programs

- [build_gnn_candidate.py](build_gnn_candidate.py)
- [../../../experiments/arithmetic-kernels/third-cumulant-rank-cuts-sharply-worsened-raw-error-while/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py](../../../experiments/arithmetic-kernels/third-cumulant-rank-cuts-sharply-worsened-raw-error-while/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L935](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L935)
- [LANE_g3_tracker.md#L936](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L936)
- [LANE_g3_tracker.md#L937](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L937)
- [LANE_g3_tracker.md#L938](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L938)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-candidate-parity-bug-exported-variance-head-active-although-instrument) · [Copy provenance](../../../PROVENANCE.json)

