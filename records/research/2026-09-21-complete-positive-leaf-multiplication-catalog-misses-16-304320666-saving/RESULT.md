# Complete positive-leaf catalog search

This study removes the earlier minimum leaf dimension of eight and includes
every positive classical leaf in the same two finite multiplication catalogs.
It preserves the earlier source, results and correctly scoped negative.

| Fixed catalog | 35 optimistic saving | 100 optimistic saving | 35 with leaf limit | 100 with leaf limit |
| --- | ---: | ---: | ---: | ---: |
| executable_smirnov_and_bank_winograd | 13.840242483% | 14.705920377% | 12.711995822% | 13.555899153% |
| extended_optimistic_catalog | 15.312586729% | 16.187350012% | 14.023710695% | 14.859381284% |

The unchanged-raw uniform planning cut is **16.304320666%**
for 3e-9. The fixed gate requires BOTH current development cases to cross.
Both catalogs miss the gate. Changed per-call saving records:
286/1180; changed cost/memory/path entries: 514/2360.
The exact per-network additional savings and chosen leaf histograms are
recorded in summary.json; audit_result.json retains every changed call.
The two original cold bills remain 413448696909 / 416264792412 F.

All catalog partitions have axes at least two, so each nontrivial child's
largest axis strictly decreases and is at most ceil(parent maximum/2).
All 586 nonempty current shapes have maximum axis at most 16384. Fourteen
stages therefore cover the entire declared positive-leaf family; the search
terminates by shape, without an arbitrary remaining-stage cutoff.

The memoized frontier retains every Pareto entry with simultaneous leaf
storage at most 1168900096 bytes, plus the unrestricted minimum-cost entry.
An above-limit footprint cannot become feasible under an ancestor's rank
multiplier and fringe maximum. It can affect only the unrestricted cost,
whose cheapest child is retained. This is not a total-workspace bound.

Independent replay certifies all 2360 chosen schedules and 1180 call records,
including exact integer costs, memory, totals and catalog monotonicity.
The unpruned exhaustive reference, full producer Pareto frontier and compact
frontier agree in 2224 fixed fixture/arena/subordinate-limit comparisons.
There are 864 synthetic ancestor checks, 15 exact rational Winograd identities,
4664 inherited classical-fringe checks, two complete source inverses and
per-call nonworsening against the old minimum-eight search. Fixtures include
positive dimension one, nonpowers of two and nontrivial cost/memory tradeoffs.
The exact current native dispatch source identifies
46 current-census tile witnesses
with a leaf axis below eight; this is metadata replay, not a new native bill.

These are optimistic arithmetic envelopes. They omit replacement transforms'
full workspace, copying, dispatch, initialization, runtime and numerical
error. Savings favorably replace entire measured old calls. No candidate,
native runtime qualification, new score or official forecast is established.
Earlier Smirnov runtime failures and the Read6 development failure stand.
Uniform recursive children, the fixed catalogs and declared padding/fringes
define the scope; heterogeneous children, structural-zero simplifications,
other algorithms, streaming and combined accuracy/cost work remain outside it.

Three guarded groups retired; controller 46267 terminal. Search state counts:
31824 / 869604. Peak sampled RSS
942358528 B; self-report
948109312 B; zero added swap. About
1.960 MB artifacts before closeout. No estimator calls,
array decoding, reference reads, fits, new samples, downloads or GPU calls.

Official 331684 remains adjusted 3.5844144212297072e-9, raw
1.949715441540434e-8, 100/100 complete with zero failures. The 4e-9 crossing
is banked; 3/2/~1e-9 remain unmet. The unchanged-raw multiplier floor
1.949715441540434e-9 rules out cost alone reaching ~1e-9, but not 2e-9.
The original goal remains active and qualified uploads remain authorized.

This closes these all-positive-leaf uniform-child catalogs as a standalone
uniform-cost route to 3e-9 under the fixed two-case gate.
