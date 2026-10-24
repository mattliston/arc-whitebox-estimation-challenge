# Head input (c): row sizes of the two fold terms' left factors — exact definition and native price

TAG opus_engineer, 2026-10-02 (lead 18:42Z: "for (c) please write the exact torch definition next to the trainer's column list").
No session behind this file: it is read from `O/ncar6.py`, `O/k31l.py`, `O/trainer/otr_m.py` (columns) and the native fold
`O/engineer/a38_pairlaw/fragment_np_v9.py` (`_np_left`).

## What the two numbers are

The law's fold term in the bank read of layer l is, on a response row i,

    term_ij = - k31 (mu_i / (n v_i)) (B_ij - forms_ij),      B = Lw pp',      Lw_ir = w_r pp_ir + x_r pp_i,partner(r)

(`ncar6.build`: `B = (pp * w[None, :] + pp[:, partner] * x[None, :]) @ pp.T`). The native read never forms B: it holds the LEFT
factor Lw on the response rows (`r2` in `_np_left`, with lam_r r2_ir = kap Lw_ir, kap = strength / n) and multiplies it with the
pushed columns inside the read's own product. The young pairs' term has the same shape,

    P_ij = sum over young pairs h of  L+_ih p+_jh + L-_ih p-_jh,      term'_ij = - k31 (mu_i / (3 v_i)) P_ij.

## Definition (torch names; layer l = 1..14; float64 until the last step)

1. rows = the response rows of this layer's bank read; pp = the columns pushed into l (n x R, float64); v = diag(C) floored at 1e-12;
   mu = the pre-activation mean.
2. LAW (module n6x / ncar6, state S): w, x, partner = S['w'][:R], S['x'][:R], S['partner'][:R].
   Lw = pp * w[None, :] + pp[:, partner] * x[None, :];   a_i = sum_r Lw_ir^2;   x_i = (|mu_i| / v_i^2) sqrt(a_i)   for i in rows.
3. YOUNG PAIRS (module k31l, inside `plane_of`, for every cohort j that has a strength wa; tp, tm, gap, AY, AE as in the module, AFTER
   the chaos4 lines):
   Lp = wa * (AE - tm[None, :] * AY) / gap;   Lm = wa * (tp[None, :] * AY - AE) / gap        (so that AY @ Y.T + AE @ E.T = Lp @ pplus.T + Lm @ pminus.T)
   a'_i = sum over the cohorts and their pairs of (Lp_ih^2 + Lm_ih^2);   x'_i = (|mu_i| / v_i^2) sqrt(a'_i)   for i in rows.
4. Each of x, x': z_i = clip((x_i - mean) / sd, -8, 8), mean and sd over the rows (sd with n - 1); 0 if sd is 0 or not finite;
   every unit outside the rows gets 0; layers 0 and 15: 0. Cast to float32.

Per-layer constants (k31, 1 / n, 1 / 3, kap) drop out in step 4, so they are left out of steps 2-3.
Native: a_i = sum_r (lam_r r2_ir)^2 before the f-scaling, a'_i = sum over the young columns of (lam_r value_ir)^2
(`local[2]` in `_np_left`); both sides then compute the same number up to float32 rounding.

## Price, and why I do not recommend it

- It is NOT free (my 18:13Z line said 0.05% of bill each; the count below replaces it). a_i is one more pass over
  (response rows x all pushed columns): square, then a matrix-vector product with lam^2 = 3 flops per element. Elements per
  network = sum over layers of rows x columns = about 818 x 167,104 = 137 M (stored S41 session, network 651: 167,104 pushed columns
  over the 14 folds; rows per layer from the trainer's 18:47Z line). 410 Mflop = 0.16% of bill, 4 requests a layer. a'_i runs over
  the young columns only (about 1,400 of 11,900): 0.02% of bill.
- It is a weak stand-in for columns 1-2. Columns 1-2 measure the row of the SPECIFIC part R = B - forms. a_i measures the row of
  the whole matrix B, forms included, and without the Gram matrix of the columns (|B_i.|^2 = Lw_i. (pp' pp) Lw_i.'). The forms
  cannot be taken out of a_i, and the Gram matrix is a product of n x R^2 = 1.5e11 flops a layer. One line in torch tells how much
  is lost: the correlation over the rows between column 1 and x_i of step 2.

## Native price of the other columns of the second batch (same counting)

| column | what the native fold has | extra work | bill | requests a layer |
|---|---|---|---|---|
| 0 / 6, 5 (diagonal residual, fit share) | the diagonal on the response rows | with the table regression (dfit 2, the fired file) the fold fits three regressors without the constant; column 6's fit has four, so it is one more 4 x 4 solve on per-unit rows | 0 | about 8 |
| 7, 8 (carried part, cross part) | only their sum | one matrix-vector product of the squared columns with lam q over (rows x columns); the cross part is the difference | 0.11% | about 4 |
| 9 (young pairs' plane diagonal) | the left factor on the young columns | one product and one matrix-vector product over (rows x young columns) | 0.02% | about 4 |
| (c) law / young pairs | the left factors | see above | 0.16% / 0.02% | 4 / 4 |
| 10, 11 (W' rv, W' (d o Pv)) | the per-unit ingredients on all units; rv and Pv only at the birth hubs | one polynomial evaluation on all units and two products with the float32 weights | 0.02% | about 8 |

So of the second batch, 9, 10 and 11 are close to free, 7-8 cost a tenth of a percent, (c) costs a sixth of a percent for a weak proxy.
