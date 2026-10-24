# Complete order-three two-edge path birth, same signed-cube bank

September5,2026. Phase2 only. No change to goal or submission conditions.

Prior `radial_compact_cubicvertex_v1.py` inserted only the near-hub21
third-cumulant leg. It failed on the older compact consumer; that particular
term/round stays closed. `projected_plane_feed_v1.py` projected that same
one-bond term onto the old column and also failed. The new round is NOT a
retry of either: include ALL the primary augmented two-edge path terms at
the same power-counting order, including the opposite21 orientation and
the doubled covariance /22 leg. This follows directly from factor_k3.py's
light/heavy leg enumeration, inspected read-only.

For hub a and output i, with g=Phi(mu/s), d=pdf/s, e=-mu*d/v,

    P_ia = g_i C_ia,
    D_ia = (e_a/(2*d_a))*g_i*T_ai
           + .5*d_i*T_ia
           + (e_a/d_a)*d_i*(K22_ia/4 + C_ia^2/2).

Here T=preK21 and all entries used are off diagonal. The untraced two-edge
birth is sum_hub d_a P_ia P_ja (plus cyclic hubs). Replacing its covariance
column by P+D adds the full light-heavy path at first order, but also D^2;
that second-order remainder is an explicit approximation, not exact
factorization of the entire augmented algorithm. Our current radial bridge
already changes d to d*(1+radial/3), so divide D by that factor to preserve
the new first-order coefficient. No new factor columns. Repeated entries
continue to be restored by the existing exact current-plane star.

## Fixed gates

1. Small synthetic8/16 certificate against ALL primary-code connected
   hypertree111 diagrams of power-counting cost3 with exactly two distinct
   pair-support edges, excluding single4 and all-distinct3-edge products.
   Compare central derivative of the completed-square source, not its full
   finite-amplitude square, to the dense term sum. Require <1e-9 relative.
2. Only after certificate: natural coefficient1, current radial-feed/full
   pair host on ranks35/100. Same weights/masks/cov32 and existing1e9 means.
   Compare unlearned host and, diagnostically, frozen matched mean head.
   Require >=10% raw final-MSE improvement on BOTH unlearned hosts and
   maximum relative column shift<.1 before generating matched training rows.
   If it fails, no damping/scale/term subset/learning-rate sweep or refit.
3. Any fit would require honest pre-correction features, matched on-policy
   parity and eventual14-net adjusted/runtime gates. No gain is presumed.

This is a cheap raw-physics lever, not a compression proposal or revival
of the cost-rejected full fourth-trace carrier. Preserve all artifacts.
