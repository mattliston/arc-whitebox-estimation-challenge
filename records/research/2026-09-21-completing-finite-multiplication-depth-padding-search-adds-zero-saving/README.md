# Completing finite multiplication depth and padding search adds zero saving across 1180 calls

2026-09-21 · finding · rejected · measured

Admitting deeper recursion and padded small-parent paths left every earlier cost optimum unchanged.

Recorded measurement:

    children of width8 from parent dimensions below16 changes **none** of the
    | Fixed catalog | 35 optimistic saving | 100 optimistic saving | 35 with leaf limit | 100 with leaf limit |
    | executable_smirnov_and_bank_winograd | 13.546294% | 14.413668% | 12.658075% | 13.501748% |
    | extended_optimistic_catalog | 14.679674% | 15.553837% | 13.641481% | 14.481085% |
    Both catalogs still miss the **16.304320666%**
    unchanged-raw uniform planning cut for3e-9 on BOTH current development cases.
    1180 call records (590 calls in each catalog). No deeper or newly admitted
    The original cold bills remain413448696909 /416264792412 F.
    An additional exact comparison verifies that all 2360 chosen entries have
    are at least2 and each current outer product has a minimum dimension at
    most1024. Even with ceiling, eight stages make the minimum dimension at
    most4, below the fixed8 leaf minimum. Therefore seven stages cover every
    Independent replay checks all2360 selected schedules,
    1180 call records. An independent exhaustive
    64 fixed fixture/leaf-limit comparisons. Eight rational Winograd identities
    cover depths0..7. The old4664 classical-fringe checks also pass. Every old
    The newly admitted parent15 case is not a vacuous code change: for shape
    64x15x256 a padded Winograd option costs467712 versus475136 classical
    The1168900096-byte limit counts simultaneous leaf A/B/C storage only.
    Read6's actual development gate failure remain intact. No implementation,
    Preparation attempt1 stopped before numerical work because a strict pin
    Additive prepare_v2 normalizes only file-identity fields and verifies/adopts
    Three guarded groups retired; controller37572 terminal. Repricing visited
    16962 /780696 cached states in the two catalogs,
    731611136 B, self-report

## Programs

- [audit.py](audit.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [prepare.py](prepare.py)
- [prepare_v2.py](prepare_v2.py)
- [../2026-09-21-complete-positive-leaf-multiplication-catalog-misses-16-304320666-saving/reprice-97d673e8.py](../2026-09-21-complete-positive-leaf-multiplication-catalog-misses-16-304320666-saving/reprice-97d673e8.py)
- [run.py](run.py)
- [../2026-09-21-complete-positive-leaf-multiplication-catalog-misses-16-304320666-saving/verify_paths-2bca9122.py](../2026-09-21-complete-positive-leaf-multiplication-catalog-misses-16-304320666-saving/verify_paths-2bca9122.py)
- [reprice.py](reprice.py)
- [verify_paths.py](verify_paths.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [REEXAMINATION_INDEX_20260919.md#L681](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L681)
- [REEXAMINATION_INDEX_20260919.md#L684](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L684)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-completing-finite-multiplication-depth-padding-search-adds-zero-saving) · [Copy provenance](../../../PROVENANCE.json)

