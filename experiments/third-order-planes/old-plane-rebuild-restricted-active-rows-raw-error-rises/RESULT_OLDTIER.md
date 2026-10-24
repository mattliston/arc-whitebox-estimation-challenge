# Old third-order tier at one third of its cost: negative, with arithmetic (fable_physicist_v1/oldtier, 2026-09-22 evening)

Question (lead): carry the transported old third-order content (cohorts aged >= 5 in the shared rank-384 frame) at <= 1/3 of
today's per-layer cost with <= ~0.3% static plane and diagonal error.
Answer: no representation I found meets both limits. The best combination is about 0.9x today's tier cost (row-restricted rebuilds
are ruled out, section 2), and its approximate part costs about as much raw error as it saves bill. This agrees with 504aldo F88 and the lead's dead list.

## 1. Exact cost model of today's tier (`ot_cost_model.py`, r35 dump sizes, dense MACs, 1 unit = 2^30 MACs)

Per layer at L14 (N_old = 17,728 old columns, 544 response rows, 768 live columns): rebuild of all 1024 rows (U F) 6.5 units,
squared inner product 3.5, rotation of F at the join (overlap F) 2.3, join (range finder, QR, projections) 2.1, basis push 0.3,
plane product 0.15. Summed over L5-14 the tier is 107 dense units:

| part | share |
|---|---|
| full-row rebuild | 38% |
| squared inner product | 25% |
| join (range finder, QR, projections) | 26% |
| rotation of F | 11% |

The cost is N_old x r x (n + n_resp) plus r^2 N_old. N_old is pinned because CP compression is dead (importance is flat). r is pinned
by the youngest old cohorts (ages 5-8).

## 2. Exact identities

| identity | result |
|---|---|
| Rebuild only the response rows (plane needs 544 rows; diagonal and omitted-radial need all) | NOT free. Old diagonal on response rows only: +30.3 / +38.3 / +39.0 / +35.2% raw (r35 / r100 / r37 / r39). Old columns out of the omitted-row radial term: +5.2 / +2.3 / +6.5 / +2.8%. Both: +22-25%. |
| Rebuild only the alive rows (non-alive rows never enter the omitted-radial term) | NOT free. +4.3 / +7.0 / +2.9 / +5.8%. |
| No per-join rotation of F (keep each join group in its own frame, chain r x r overlaps) | -1.2% of the tier (the group products n r^2 per group cost almost what the rotation saves) |
| Static F between joins | none: every layer 5-14 has a join |

The non-response rows carry 33-44% of the old diagonal's energy at L14 (r35: 38%), and the learned closure reads every unit's k3.
(`ot_consumer_ablate.py`, `ot_alive_ablate.py`, card copy `../model_bank_oldtier.py`, torch, learned parts ON, no refit.)

## 3. Approximate representations, static error on the r35 dumps (relative to the total consumed plane / diagonal)

| representation | plane error | diagonal error | tier cost |
|---|---|---|---|
| whole tier at frame rank 320 | 1.0% (L9, L14) | 0.6% | ~0.85x |
| whole tier at rank 256 | 2.5-2.8% | 1.6% | ~0.7x |
| whole tier at rank 128 | 9.5-11% | 6.0-6.5% | ~0.4x |
| rank 256 + lost content's Sym(C x x) put into b (`trunc_gain_recover.py`) | 2.4-2.6% | 1.2-1.3% | ~0.7x |
| nested sub-frame, ages >= 9 at rank 256 (lead's subframe_info_loss) | 0.17-0.28% | 0.10-0.16% | 0.91x |
| nested sub-frame, ages >= 8 at rank 256 | 0.31-0.52% | 0.21-0.31% | ~0.88x |
| Tucker core in the frame | exact | exact | worse than today: at L14 the read alone is 14.3 units (n_resp r^3/2), plus 9 units for the all-row diagonal (n r^3/6), plus an r^4 core rotation at every join |

Gain structure: the old plane is 73% Sym(C x x) on the response x live block, with x along the mean (cos 0.84-0.97;
`old_sym_fit.py`). The content lost to frame truncation is NOT gain-type, though: its best Sym(C x x) part removes only 5-10% of it.
So "move the gain part into b, shrink the frame" is dead.

Best achievable combination: sub-frame (ages >= 9, rank 256) + rotation bookkeeping = about 0.9x the tier, about -3% of the bill. By
the card's measured sensitivity (0.3% static plane error -> about +4% raw), that is roughly a wash on adjusted score. Adding row
restrictions is not possible (section 2).

## 4. Why (one paragraph)

Every read of the old tier needs T(u_a, u_a, .) for each response row a: the core T = sum_j lam_j F_j (x) F_j (x) F_j contracted
twice with the row's frame coordinates u_a. The diagonal needs T(u_a, u_a, u_a) for all 1024 rows. With a CP core this costs 2 N r
per row; with a dense core, r^3/2 per row. The Hadamard square rules out any linear sketch. The plane's low matrix rank (98% rank 64)
does not help, because every row still needs its N squared projections. N is set by flat importance (CP-incompressible), and r is set
by the ages-5-8 cohorts. The consumers need all rows, and the learned closure reads k3 on every unit.
