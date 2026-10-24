# NCAR6: the whole lean law of the norm-coupling matrix on the bank's own column pairs (lead, 2026-10-02 13:05Z)

Torch reference: `O/ncar6.py` (install(...), level 'read' = the native form; level 'r31' = research reference through a separate plane).
Harness arm: `O/e16b_k4_closed_loop.py <rank> 4 <tag> "name:f4=0.6,age=2,c=0.75,nc=1.0,nclaw=6,nclo=1,nca=0.75,ncar=0,nclev=1,ncbf=1,ncdf=2"`.
First numbers (held-out 651, base F4 0.6 + K4L 0.75 = 16.008e-9, torch, no refit, strength 1; `e45_651/rank0651_e48.log`, `rank0651_e49.log`):
bank channel only -4.64%; + row forms -12.44%; + diagonal -12.36%; + births (the full law) -14.43% (-14.90% without the isotropic-rest sandwich);
full-matrix complete law (physicist's ncarry, about 12 n^3 products a layer) -15.1%. The fold inside the bank read: four forms removed exactly -13.84%,
only the multiple of C removed exactly -12.63%, strength 1.25 -14.65%. Seven-network table: `e51_pairlaw/` (running).

## Why it is free to carry
The card's newborn bank columns of hub h are p+- = s_h + t+- e_h (s_h = `stored[:, h]` = g o Cpre[:, h] with zero own entry, plus the birth-direction
changes; e_h = unit vector of h; t+ = plus, t- = minus, gap = t+ - t-). The law's fresh sources in the post space of layer l are
births sum_h S2_h c_h c_h', row forms sum_h rv_h (e_h c_h' + c_h e_h'), diagonal sum_h D_h e_h e_h', with c_h = g o Co[:, h] = s_h (to the birth
changes). So each hub's contribution is a 2 x 2 symmetric form on its own pair, [s e] M_h [s e]', M_h = [[S2_h, rv_h], [rv_h, D_h]],
s = (t+ p- - t- p+) / gap, e = (p+ - p-) / gap, and the bank's transport of the columns (push, gates, alive mask, joins) transports it.

## State added to the bank
`w_r`: one float per bank column (all cohorts, old tier included).  `x_h`: one float per pair (cross weight).
Matrix of layer l on the columns pushed into l:  B = sum_r w_r pp_r pp_r' + sum_h x_h (pp+ pp-' + pp- pp+').

## Per layer l = 0..14, AFTER the pair map (all per-unit vectors unless said)
Inputs: mu, C = Cpre(l) (v = its diagonal, Co = C with zero diagonal), m = the layer's post mean (host + learned correction), K21 = the plane the pair map
received (Ko = K21 with zero diagonal; used for l >= 1), radial of the layer, pp = columns pushed into l, lam_in = their weights as given to the push,
diagB and aB of this layer (from the read step below; zero at l = 0), the births of this layer: selected hubs, t+, t-.
```
s = sqrt(v); al = mu / s; ph = pdf(al); g = cdf(al); d = ph / s
r2 = v ((1 + al^2) g + al ph); r3 = v s ((al^3 + 3 al) g + (al^2 + 2) ph); r4 = v^2 ((al^4 + 6 al^2 + 3) g + (al^3 + 5 al) ph)
gam = g - m d; V = 2 m (1 - g); S2 = 2 (gam - g^2); S3 = 3 (1 - g)(r2 - 2 m^2); S4 = r4 - 4 m r3 + 6 m^2 r2 - 3 m^4 - 3 (r2 - m^2)^2; e3 = - al ph / v
gb = mean(gam)                                                    (scalar)
c_r = lam_in_r (V . pp_r)                                         (one pass over the pushed columns, one vector)
K3Vd = (pp o pp) c                                                (one pass)
a_rest = 0.75 (radial / 3) tr C - aB;   Ntd = diagB + a_rest v
Co2 = Co o Co;   Pv = gb Ntd + 2 Co2 gam
rv = d (Co V) + (e3 / 2) Pv + S3 + d (Ko' gam) + (e3 / 2) (Ko V)
dgn = S4 + V (Co V) + 2 gam (Co2 gam) - 2 g^2 (Co2 g^2) + gam gb Ntd + V (Ko' gam) + gam (Ko V) - 2 g d ((Ko o Co) g^2) - 2 g^2 ((Ko' o Co)(g d))
D = dgn - g^2 (Co2 S2 + gb diagB + K3Vd)
carried columns:  w_r <- 2 (gb w_r + c_r);   x_h <- 2 gb x_h
newborn pair of hub h:  w+ = 2 (S2 t-^2 - 2 rv t- + D) / gap^2;  w- = 2 (S2 t+^2 - 2 rv t+ + D) / gap^2;  x = 2 (- S2 t+ t- + rv (t+ + t-) - D) / gap^2
```
(S2, rv, D taken at the hub.) Matrix-vector products: Co V, Co2 with gam, g^2, S2 (Co2 is one elementwise n^2 pass), Ko' gam, Ko V, (Ko o Co) g^2, (Ko' o Co)(g d):
eight n^2 products and two elementwise n^2 passes; two passes over the pushed columns with one vector each.

## At the bank read of layer l = 1..14
```
diagB_i = sum_r (w_r pp_ir^2 + x_r pp_ir pp_i,partner(r))        (one elementwise pass over rows x columns, row sums)
co = least squares of diagB on (tr C v_i, mu_i^2, 2 mu_i (C mu)_i) over the units;   aB = RHO1[l] co[0] tr C       (RHO1: 14 literals in ncar6.py)
f_i = - strength mu_i / (n v_i)
left factor of the read:   pp_ir^2 lam_r  ->  pp_ir^2 lam_r + f_i (w_r pp_ir + x_r pp_i,partner(r))      (young product and old tier's inner product)
after the read, off the diagonal, on the read rows and columns:   plane_ij -= f_i aB C_ij
```
The plane's diagonal is overwritten by the card afterwards as before (the bank's own cubic sum), so nothing of the fold stays on it.
Literals: strength (1; the trainer will line-search it), RHO1 (per layer), 0.75, first and last layer, strengths of S2 / rv / D (1, 1, 1).

## Open (being measured)
- whether D can be dropped (it does not help the final layer on 651; that would remove six of the eight n^2 products);
- the forms: only the multiple of C is removed natively; the exact four-form fit is 1.2 points better on 651;
- pooled value on the 50 held-out ranks (trainer) and the per-layer numbers the engineer needs for parity (dump option of ncar6.py).

## ADDENDUM 2026-10-02 15:10Z: K31L4, the fourth-chaos part of the young pairs' local quartic in the same read (candidate for S41)

Torch reference: `O/k31l.py` (install(cd, torch, st, k31=1.0, age=2, forms=0, chaos4=True, level='read')); harness options `k3l=1,k3lc=1,k3lf=0,k3lr=1`.
Numbers (torch, no refit, base F4 0.6 + K4L 0.75, ON TOP OF the pair-carried law with the constant-regressor three-form removal; `e62_k31l4/`,
`python e51_summary.py e62_k31l4`): seven networks 35 287 328 378 424 641 651: law alone -6.13% pooled, law + K31L4 -7.60% (per network
-6.2 -3.4 -7.7 -8.3 -12.3 -1.5 -14.7 against -6.2 -1.8 -7.1 -5.8 -9.0 -0.9 -13.0); all layers -7.99% against -6.29%. On 651: through
remainder31 -1.96% against the law alone, folded into the read -2.09%; age 1 only +0.96%, ages <= 3 / 4 -1.23 / -1.51%, strength 0.5 / 1.5
-1.38 / -1.52%; the full local form on top of the law (its second chaos counted twice) -0.4%; without the law the full local form -5.4%.

What it is. K4L's local quartic of a young pair h (cohorts of age 1 and 2 at the layer being read), on its two pushed vectors
Y = (t+ pp- - t- pp+) / gap and E = (pp+ - pp-) / gap with the birth coefficients (S1, S2, S3, S4) = (e3, S2, S3, S4) of the hub:
```
K31L_ij = sum_h [ AY_ih Y_jh + AE_ih E_jh ]
AY = 3 S1 Y^2 E + 3 S2 Y E^2 + S3 E^3                         (all per element i, h)
AE = S1 Y^3 + 3 S2 Y^2 E + 3 S3 Y E^2 + S4 E^3
second-chaos projection (already carried by the pair law), with the column means mY2 = mean_i Y^2, mYE = mean_i Y E, mE2 = mean_i E^2 of the pair:
AY -= 3 [ S1 (mY2 E + 2 mYE Y) + S2 (mE2 Y + 2 mYE E) + S3 mE2 E ]
AE -= 3 [ S1 mY2 Y + S2 (mY2 E + 2 mYE Y) + S3 (mE2 Y + 2 mYE E) + S4 mE2 E ]
```
Use: K21_eff[i, j] = K21[i, j] - (mu_i / (3 v_i)) K31L_ij on the read's rows and columns; nothing removed (the fourth-chaos part has no forms).
In the read's left factor, for the two columns of the pair: + f3_i (AE - t- AY) / gap on pp+,  + f3_i (t+ AY - AE) / gap on pp-,  f3_i = - mu_i / (3 v_i).
Elementwise work on the response rows x the columns of the two youngest cohorts (about 3,600 columns); the column means need all rows of those columns
(the means over the response rows may do: to be checked). K4L already forms Y, E and their powers for these cohorts natively.
Column means (15:30Z, `e61_k31l/rank0651_d.log`, `rank0378_d.log`; against the law alone): over all units -2.09% / -3.04% (651 / 378); over the read's
response rows only -1.97% / -2.88%; WITHOUT the cross term mYE (only mY2 and mE2, over all units) -2.11% / -3.07%: drop the cross term.
