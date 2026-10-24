# Complete finite catalog depth/padding search: no further saving

Completing the search from five to seven stages and admitting ceil-padded
children of width8 from parent dimensions below16 changes **none** of the
current-call cost optima. The previous five-stage search was narrower than
an unrestricted minimum-leaf-eight search. Its numerical optima survive
this new completion, with the original outputs preserved.

| Fixed catalog | 35 optimistic saving | 100 optimistic saving | 35 with leaf limit | 100 with leaf limit |
| --- | ---: | ---: | ---: | ---: |
| executable_smirnov_and_bank_winograd | 13.546294% | 14.413668% | 12.658075% | 13.501748% |
| extended_optimistic_catalog | 14.679674% | 15.553837% | 13.641481% | 14.481085% |

Both catalogs still miss the **16.304320666%**
unchanged-raw uniform planning cut for3e-9 on BOTH current development cases.
Additional saving from the new search is exactly zero for every one of the
1180 call records (590 calls in each catalog). No deeper or newly admitted
small-parent padded path improves an actual current-call optimum. This is
an optimistic arithmetic screen, not a measured estimator cost or score.
The original cold bills remain413448696909 /416264792412 F.
An additional exact comparison verifies that all 2360 chosen entries have
identical cost, leaf memory and schedule path, with each call matched by
catalog, network, index, shape, caller and actual measured call bill.

The structural depth argument is explicit: all catalog partition dimensions
are at least2 and each current outer product has a minimum dimension at
most1024. Even with ceiling, eight stages make the minimum dimension at
most4, below the fixed8 leaf minimum. Therefore seven stages cover every
admissible path in this specified family. The family still chooses one
uniform recursive child program at each node, with the declared ceil or
floor-core/classical-fringe convention. It does not include arbitrary
algorithms, heterogeneous child schedules, alternative leaves or streaming.

Independent replay checks all2360 selected schedules,
their exact integer costs and simultaneous leaf footprints, and all
1180 call records. An independent exhaustive
enumeration of all up-to-two-stage schedules agrees with the producer on
64 fixed fixture/leaf-limit comparisons. Eight rational Winograd identities
cover depths0..7. The old4664 classical-fringe checks also pass. Every old
per-call optimum is nonworsening and catalog-superset monotonicity holds.
The newly admitted parent15 case is not a vacuous code change: for shape
64x15x256 a padded Winograd option costs467712 versus475136 classical
operations. This tiny witness is not a current estimator improvement.

The1168900096-byte limit counts simultaneous leaf A/B/C storage only.
Transforms, full workspace, copies, dispatch, rounding and actual runtime
are not qualified; costs also favorably allow replacing measured old call
initialization. Earlier Smirnov streaming/global-batch runtime failures and
Read6's actual development gate failure remain intact. No implementation,
native qualification, candidate, package or upload follows automatically.

Preparation attempt1 stopped before numerical work because a strict pin
dictionary comparison rejected extra rank/call-count fields on two old
census records. Their path, SHA256 and byte counts matched exactly. The
original source, failure record and already written drafts are preserved.
Additive prepare_v2 normalizes only file-identity fields and verifies/adopts
those identical drafts. Producer formulas, inputs, protocol and gates did
not change. Two whole-source inverses independently verify the declared
producer and path-verifier changes against their preserved originals.

Three guarded groups retired; controller37572 terminal. Repricing visited
16962 /780696 cached states in the two catalogs,
using stdlib integer/rational arithmetic only. Peak sampled RSS
731611136 B, self-report
739737600 B, zero added swap. About
1.130 MB artifacts before closeout. No estimator calls,
target/network array decoding, fits, new samples, downloads or GPU.

This closes the specified completed finite family as a standalone3e-9
uniform-cost route, not smaller/composed improvements or all algorithms.
Official331684 stays adjusted3.5844144212297072e-9,100/100 complete,zero
failures.4e-9 is banked;3/2/~1e-9 remain unmet. The unchanged-raw floor
1.949715441540434e-9 prevents cost alone reaching~1e-9, but not2e-9.
The original goal remains active.
