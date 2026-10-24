# K4L: non-isotropic per-unit fourth cumulant from the exact hub coefficients (lead, 2026-10-02)

Reference implementation (torch, hook level): `O/e16b_k4_closed_loop.py` (arm options `c`, `age`, `final`, `pair`, `f4`).
Open-loop evidence: `O/e16_local_k4.py`, `O/e16_rank0035.log`. Closed-loop screens: `O/e16b_rank*_s*.log`.

## What the card does today
The per-unit fourth cumulant of the pre-activations is one number per layer: `k4_i = radial * v_i^2` (times the band dial
k4a/k4b/k4c). It feeds (a) the mean formula of every layer through the pair map's `host`, (b) the pair map's fourth-order rows
(`T31 = t4 * rowfactor * rho`), (c) the final layer's mean (`k4_15`).
Against 32M-sample truth (network 35) that isotropic value is wrong by 14% (layer 5) to 26% (layer 13) relative rms per unit; the
best single coefficient leaves 13.5% to 25%.

## What K4L adds
Every column pair of the bank was born at a hub h (layer j) as `p+- = s_h + t+- e_h`. At birth the ReLU at h makes a fourth cumulant
of the post-activations with exactly known coefficients (m legs on the hub, 4-m on the covariance column):

    v = sigma^2, d = pdf/sigma, a = alpha, g = gate, m = host mean of the hub (the pair map's mean, before the learned correction)
    r2 = v ((1+a^2) g + a pdf);  r3 = v sigma ((a^3+3a) g + (a^2+2) pdf);  r4 = v^2 ((a^4+6a^2+3) g + (a^3+5a) pdf)
    S1 = -a pdf / v                                   (= E f''')
    S2 = 2 (g - g^2 - m d)                            (= d^2 Var(x) / d mu^2)
    S3 = 3 (1 - g)(r2 - 2 m^2)                        (= d kappa3(x) / d mu)
    S4 = r4 - 4 m r3 + 6 m^2 r2 - 3 m^4 - 3 (r2 - m^2)^2     (= kappa4(x))

At a later layer l the pair's pushed columns pp+, pp- (pre-activation space) give the two transported legs

    Wt = (pp+ - pp-) / (t+ - t-)            (the hub's unit vector, transported)
    Y  = (t+ pp- - t- pp+) / (t+ - t-)      (the covariance column, transported)

and the hub's contribution to the per-unit fourth cumulant of unit i of layer l is

    loc_i += 4 S1 Y_i^3 Wt_i + 6 S2 Y_i^2 Wt_i^2 + 4 S3 Y_i Wt_i^3 + S4 Wt_i^4
           = Wt_i (4 S1 Y_i^3 + Wt_i (6 S2 Y_i^2 + Wt_i (4 S3 Y_i + S4 Wt_i)))

summed over the pairs born at most `age` layers earlier (age = 2 in the screens; ages <= 4 are never joined into the old tier).
The isotropic level the card was trained with is kept: with alpha_i = mu_i / sqrt(v_i), w_i = ((alpha_i^2 - 1) exp(-alpha_i^2/2))^2
(the size of the kurtosis term of the mean formula),

    beta = sum_i w_i loc_i v_i^2 / sum_i w_i v_i^4,      dk_i = loc_i - beta v_i^2,      k4_i <- k4_i + c dk_i     (c = 1)

applied to the k4 the pair map receives at layers 1-14 (argument and `ctx['k4']`) and to the final layer's k4.
Everything is elementwise on arrays the step already holds (the pushed young columns and six numbers per pair kept from birth).
Estimated bill: about 14 flops per element of the two youngest cohorts per layer = about 0.13% of the bill; no product.

## Evidence
Open loop (network 35, layers 5/9/11/13, truth 32M with an independent second draw, weights w_i): isotropic + loc explains
0.28 / 0.26 / 0.25 / 0.34 of the card's per-unit k4 residual energy with fitted loc coefficient 0.88 / 0.89 / 0.88 / 1.02.
Closed loop, current torch card, learned on, no refit, final-layer MSE against the organizers' truth (networks 35 / 100 / 37 / 39):

| arm | 35 | 100 | 37 | 39 | pooled |
|---|---|---|---|---|---|
| K4L, age <= 2, c = 1 | -2.70% | -3.65% | -2.94% | -1.96% | -2.83% |
| K4L, all ages, c = 1 | -3.20% | -4.09% | -3.50% | +0.05% | -2.74% |
| K4L, all ages, c = 0.5 | -3.25% | -3.62% | -3.31% | -1.76% | -3.01% |

With the physicist's F4 (theta 0.4, age 2, cheap anchor) installed first: see `e16b_rank*_s2.log` (additive on 35 and 100).
