# NCAR: a vector carrier of the network-specific norm coupling, used as an extra K31 plane (lead, 2026-10-02)

Reference implementation (torch, hook level): `O/ncar.py` (`install(...)`, law 2), driven by arm `nc` of `O/e16b_k4_closed_loop.py`.
Open-loop checks: `O/ncar_compare.py`, `O/ncar_mode1.py`. Monte Carlo behind it: `O/e31*`, `O/e32*`, `O/e35*`; physicist's T19 series.
Status of the design: K, the bank pass and the isotropic transport terms are being screened (E37, E39); this file is updated when they are fixed.

## What it is
For iid Gaussian weights the fourth cumulant of a layer's pre-activations has a second-chaos part carried by one matrix,
`Ntilde_ij = sum_k kappa4(z_i, z_j, z_k, z_k)` (how the covariance of a pair moves with the layer's squared norm). Its network-specific part is
mostly "top covariance eigenvector x per-unit partner vector": `sum_m (u_m q_m' + q_m u_m')`, `u_m` = top-K eigenvectors of the pre-activation
covariance `C` of the layer, `q_m` orthogonal to all `u`. The true matrix as an extra K31 plane `(3/n) Ntilde_ij` is worth -6% (network 35) and
-15% (held-out rank 651) of final-layer error (physicist's oracle). NCAR carries the `q_m` from layer to layer with matrix-vector products.

## State kept from layer l-1 (written at the end of step l-1, after the pair map and the births)
Per-unit arrays of layer l-1 (all from the pair map's `aux` and `host`): `g` gate, `m` host mean of the post-activation, `sigma`, `alpha`, `pdf`,
`v = sigma^2`, `d = pdf / sigma`, and
```
r2 = v ((1 + a^2) g + a pdf);  r3 = v sigma ((a^3 + 3a) g + (a^2 + 2) pdf);  r4 = v^2 ((a^4 + 6 a^2 + 3) g + (a^3 + 5a) pdf)        (a = alpha)
gamma = g - m d;  gbar = mean(gamma);  beta = 2 m (1 - g)
S2 = 2 (g - g^2 - m d);  S3 = 3 (1 - g)(r2 - 2 m^2);  S4 = r4 - 4 m r3 + 6 m^2 r2 - 3 m^4 - 3 (r2 - m^2)^2
```
Matrices and vectors of layer l-1: `C` (its pre-activation covariance, n x n, kept one layer longer), `Co` = C with zero diagonal, `U` (n x K) and
`Q` (n x K) of that layer (Q absent at the first layer), scalars `trC`, `a = ca * radial / 3`, `b = cb`.
Bank quantities of layer l-1 (`pp` = the pushed columns of layer l-1, `lam` their weights after that layer's scaling):
```
c_r = lam_r (pp_r . beta)                                     one number per column           (conversion weights)
tc  = Co beta + pp (lam o ((pp o pp)' gamma))                 per unit                        (norm coupling of the means)
Nd0 = a trC v + b sum_k C_ik^2 + 2 sum_m U_im Q_im            per unit                        (diagonal of the carried matrix)
Pc  = 2 (Co o Co) gamma + gbar Nd0 + (pp o pp) c              per unit                        (norm coupling of the variances)
r   = S3 + d tc - (d alpha / (2 sigma)) Pc                    per unit                        (row factor: own (3,1) + mean tilt + variance tilt)
Nd  = S4 + beta (Co beta) + 2 gamma ((Co o Co) gamma) - 2 g^2 ((Co o Co) g^2)                 (diagonal of the post matrix)
```

## Step at layer l (before the pair map; w = weights of layer l, pre = w' post)
```
U'  = top-K eigenvectors of the layer's pre-activation covariance C'            (reference: eigh; native: block power iteration, warm start w' (g o U))
y   = w U';   yg = g o y                                                         (n x K, post space of layer l-1)
Ny  = a trC (C yg) + b C (C yg) + U (Q' yg) + Q (U' yg)                          (carried matrix applied to yg)
T   = ct 2 gbar w' (g o Ny)                                                      (transport)
Cy  = Co yg
src = 2 w' [ g o (Co (S2 o Cy)) + r o Cy + g o (Co (r o y)) + Nd o y ]           (Gaussian births, row forms, diagonal)
    + conv 2 pp_l (c o (pp_l' U'))                                               (conversion; pp_l = the pushed columns of layer l that existed at l-1)
q   = T + cs src;      Q' = q - U' (U'' q)                                        (partner vectors, orthogonal to U')
plane (layers lo..hi = 2..14):  R_ij = sum_{m in sel} (U'_im Q'_jm + Q'_im U'_jm),  remainder31_ij = k31 (3/n) R_ij / (sigma'_i^3 sigma'_j),  diagonal 0
```
`sel` = all K modes when mode 1 is included, modes 2..K otherwise. The plane is ADDED to the pair map's (3,1) remainder input (the torch card's
`pair_in` value[5]; the physicist's `ntor.py` builds the same standardisation). Only the rows and columns the pair map reads are needed.
Defaults in the screens: `ct = 0.9, cs = 1, ca = 0.5, cb = 0.25, conv = 1, k31 = 1`.

## Evidence (torch, no refit, final-layer MSE against F4 0.6 + K4L 0.75)
- Open loop, network 35, cosine of the carried partner vectors of modes 2-8 with the Monte Carlo truth: 0.86-0.91 (layer 2), 0.61-0.79 (5),
  0.36-0.76 (9), 0.46-0.74 (13), 0.41-0.71 (14); share of their energy held 0.79 / 0.48 / 0.38 / 0.40 / 0.36.
- Closed loop: network 35: K = 8 without mode 1 -0.75%, with mode 1 -1.05%; held-out rank 651: -3.55% / -3.99%.
- First three held-out ranks (651, 378, 424), pooled: K = 8 -2.1% (modes 2-8), -2.9% (modes 1-8); K = 16 -3.6% (modes 2-16), -4.2% (modes 1-16).
  The full 12-rank table and the design variants (E37, E39) replace these lines when they are in.

## Native cost per layer (my count, K modes)
- eigenvectors: block power iteration, 2 K n^2 flops per sweep plus an n x K orthogonalisation; warm-started, 2-4 sweeps;
- products with w: 2 (y and the final w'), 4 K n^2 flops; products with C / Co: C yg, C (C yg) [only if b is kept], Co (S2 o Cy), Co (r o y): 6-8 K n^2 flops;
- bank pass [only if conv is kept]: pp' U' and pp (.), 4 K R n flops with R = young columns (about 5,900): 24 K Mflop: the dearest item;
- per-unit vectors (tc, Pc, Nd, c): two passes over the bank (R n) and three n^2 passes; the plane: 2 K flops per element read.
- K = 16 without the bank pass: about 0.35 Gflop per layer (about 1.7% of the bill); K = 8: half of that.

## SUPERSEDED 2026-10-02T12:54Z
The vector carrier of this file (and ncar3.py, ncar4.py, ncar5.py) is superseded by the pair-carried law: `O/ncar6.py`, spec `O/NCAR6_SPEC.md` (the whole lean law as a 2 x 2 weight per bank column pair, folded into the K21 bank read; -14.4 to -14.9% on held-out 651 against -4% here). Kept for the record.
