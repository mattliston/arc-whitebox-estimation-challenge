# Cheap consistent third+fourth-order plane supply for the covariance output: none found (fable_physicist_v1/fourth, 2026-09-22 night)

Target (lead, fable_k4diag_v1/planes_covonly_oracle.log): truth K21+K22+K31 fed only to the pair map's covariance output, projected
orthogonal to the post mean, gives 0.762 (r35) / 0.746 (r100). K22+K31 alone gives 0.844 / 0.786; K21 alone 1.056 / 1.113.

## 1. Independent-source family: the carried columns also carry the fourth order

Physics: if the non-Gaussianity comes from independent sources along the carried columns, K3 = sum lam_j p_j^3 (the card's bank) and
the consistent fourth cumulant is K4 = sum lam4_j p_j^4 on the SAME columns. Transport is then free.

Oracle test (`col_lambda4_fit.py`, r35 L12): one FREE lam4 per column, fitted to the 8M truth errors of K31 and K22 (off-diagonal,
response rows x live columns). The fit is weighted by what the covariance output reads at r=0: |alpha_a| pdf_a g_c / v_a for K31, and
pdf_a pdf_c / (s_a s_c) for K22.

| columns | K31 alone | K22 alone | one shared lam4 (K31 / K22) |
|---|---|---|---|
| young (5,440) | 40% | 36% | 39% / 17% |
| old (14,912) | 43% | 36% | 42% / 11% |
| all (20,352) | 51% | 44% | 51% / 19% |

One-draw noise is about 3% of these error energies. So even with free weights the family carries only half of the K31 error. It is
also inconsistent: the weights that fit K31 misfit K22, which the independent-source picture forbids.

## 2. Fit to the NET covariance change of all three planes (`net_read_fit.py`, r35 L12)

The pair map's covariance output is linear in the planes. The r=0 read of the truth errors on the response x live block:

| read | size |
|---|---|
| net change | 0.042 |
| K21 part | 0.037 |
| K31 part | 0.021 |
| K22 part | 0.009 |

The K21 part and the K22+K31 part have cosine -0.09.

Free per-column families fitted to the net change:

| family | explained |
|---|---|
| lam4 only | 27% |
| K21 re-weighting dlam only | 37% |
| both (40,704 free weights) | 38% |

## 3. Cost

One (3,1)-type read (p o p o p) diag(lam4) p^T on response rows x live columns, summed over L2-14, costs 54 dense units on the young
columns and 108 on all columns. A K22 read costs the same again. Today's bill is about 188 units (util 0.184), and the lead's budget is
+3-6%, i.e. 6-11 units. The young-only fourth-order reads would add about 30-57% of the bill in dense terms, for a family that carries
under 40% of the net change even with free weights.

## 4. The other candidates

| candidate | result | verdict |
|---|---|---|
| (2,1,1) channel | The pair map's covariance output reads only K21, K22, K31 and the marginals (first-order Edgeworth). The (2,1,1) slice acts only indirectly, through the births' K4 -> K3 feed. The exact channel costs 0.5-1.0 x B (504aldo F58/F64); the card's radial is the memoryless law they found. | dead on cost |
| third-order x nonlinearity terms (P1, gain-on-skewed-base), covariance-only | K31 supplies act only through the covariance anyway, and orthogonal projection changes nothing for them (lead's perp test). They explain 4-17% of the K31 read. Closed loop -0.1% to +2.1% (P1 variants), +0.5% / +1.3% (gain-skew fits); the consistent gain-skew triple is +45% / +145% (lead). | dead |
| second-order K3 x K3 terms (the physics behind P1: a curvature vertex joining two third-order vertices) | Only the coincident-index piece (P1) is elementwise. The full term needs a column Gram P^T diag(d) P (N^2, about 4e8 entries at L12) per layer. | dead on cost |

## Answer

No cheap consistent supply exists within the physically natural families. The fourth-order content that carries the covariance lever
is broad per-pair structure. The carried columns cannot represent it even with free weights (at most 38-51%), and every structured
term I derived explains well under a third of it. A fourth-order read of the bank costs 30-57% of the bill. The remaining route is a
direct model of the net covariance correction (the analyst's predictability test), not a plane supply.
