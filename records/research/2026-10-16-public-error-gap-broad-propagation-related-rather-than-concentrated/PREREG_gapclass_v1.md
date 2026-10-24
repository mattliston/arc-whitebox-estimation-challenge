# [opus-gapclass] pre-registration — written 2026-10-16T13:45:41Z, before computing any per-network gap

Question (lead 10-16): on which public networks do marius #336419, J2W #336188, oliver #336473 beat our #336412 most,
and does the gap track anything we can compute at runtime?

Constraint known beforehand: the board's public networks cannot be regenerated (memory p2-1013-hard-public-networks-persistent),
so no weights. The only per-network runtime fingerprints are on the board: per-net FLOPs (ours encode our adaptive cuts:
dcv dead-unit cut, RC column cut, OTT rows, birth floors) and per-net final / all-layer MSE.

Data: analyst scan files, 'pub' = [index, final MSE, all-layer MSE] for 50 public nets; 'flops' (first 50 assumed public; checked).

Q1 concentration: share of the aggregate final-MSE gap sum_n (MSE_us - MSE_L) carried by the top 10 networks, per leader.
   Concentrated = top-10 share >= 50% for >= 2 of 3 leaders.
Q2 class link: Spearman rho (n = 50) between the per-net log gap log(MSE_us/MSE_L) and 5 proxies, fixed now:
   (i) our per-net FLOPs, (ii) the leader's per-net FLOPs, (iii) consensus difficulty (mean log final MSE of the 4 files),
   (iv) our late-birth proxy log(final/all-layer MSE), (v) the all-layer log gap (propagation vs head).
   15 tests; |rho| >= 0.36 is p < .01 two-sided at n 50 (~0.15 chance hits expected).
   PASS = a proxy with |rho| >= 0.40, same sign, in >= 2 of 3 leaders.
Step 3 only on PASS of (i): calibrate our native FLOPs -> network features (dead / kink fraction L8-15, MC) on the 24
   generated control nets (ctl_s2tr + ctl_s2tr_a80) and name the network class.
Kill: no proxy passes -> the telemetry cannot name a class; report concentration and stop. Report by ~14:50Z (07:50 PT).
