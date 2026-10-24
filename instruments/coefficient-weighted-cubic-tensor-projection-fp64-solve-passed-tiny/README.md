# Coefficient-weighted cubic tensor projection with an FP64 solve passed tiny algebra and native API gates

2026-09-07T08:49:41Z · tool · informational · measured

A new compression of the CP (canonical polyadic) third-order factor bank was specified: project the complete all-distinct cubic tensor onto selected existing atoms with the kernel k_D = s1^3 - 3 s1 s2 + 2 s3 (s_r = (p^r).T (q^r)), solve the weights with one supported FP64 fnp.linalg.solve, and restore every repeated-index entry through the coordinate star update 3 R.T - 2 diag(diag R). The fixed n8/a6/M12/K6 synthetic certificate passed every predeclared gate with the native bill predicted exactly from source; an old-geometry census put the saving for compressing cohorts 0..6 after layer 7 to K=1024 atoms at 17.56-18.08% of the old complete bill, which would only narrowly cross 4e-9 from the retained panel if it transferred at unchanged raw.

Recorded measurement:

    Kernel vs explicit features 2.7749303e-7 / 4.6432160e-7; solved coefficient error 6.1418492e-7 / 9.3551684e-6; restored repeated entries 1.6792220e-7 / 1.3887537e-7; native FNP 0.12.0 bill exactly 9772 FLOPs per fixture (216+2376+1884+432+4864); modeled old-geometry remaining saving 88711402044 F (18.08056%, rank 35) and 85962131344 F (17.55928%, rank 100).

## Programs

- [code/qualify_global_cp_kernel_tiny_v2.py](code/qualify_global_cp_kernel_tiny_v2.py)
- [code/qualify_global_cp_kernel_tiny_v1.py](code/qualify_global_cp_kernel_tiny_v1.py)

## Evidence

- [GLOBAL_CP_KERNEL_TINY_V2_RESULT.md#L31](GLOBAL_CP_KERNEL_TINY_V2_RESULT.md#L31)
- [GLOBAL_CP_KERNEL_TINY_V2_RESULT.md#L38](GLOBAL_CP_KERNEL_TINY_V2_RESULT.md#L38)
- [GLOBAL_CP_KERNEL_PREFLIGHT_V1.md#L19](GLOBAL_CP_KERNEL_PREFLIGHT_V1.md#L19)

<details>
<summary>All result files and records (6)</summary>

- [GLOBAL_CP_KERNEL_PREFLIGHT_V1.md](GLOBAL_CP_KERNEL_PREFLIGHT_V1.md) · record
- [GLOBAL_CP_KERNEL_TINY_V2_RESULT.md](GLOBAL_CP_KERNEL_TINY_V2_RESULT.md) · result
- [GLOBAL_CP_KERNEL_TINY_V2_PREEXECUTION_AMENDMENT.md](GLOBAL_CP_KERNEL_TINY_V2_PREEXECUTION_AMENDMENT.md) · record
- [GLOBAL_CP_KERNEL_TINY_RESOURCE_RELAUNCH_V1.md](GLOBAL_CP_KERNEL_TINY_RESOURCE_RELAUNCH_V1.md) · record
- [global_cp_kernel_tiny_v2_nice10.json](global_cp_kernel_tiny_v2_nice10.json) · result
- [global_cp_kernel_tiny_v2.json](global_cp_kernel_tiny_v2.json) · log

</details>

## Related work

- enables: [Whole-pair global projection after layer 7: 512 pairs cut compute 19.03% but increase raw error 6.06x](../../experiments/third-order-planes/whole-pair-global-projection-after-layer-7-512-pairs/README.md)

[Research graph](../../graph/index.html#node=2026-09-07-coefficient-weighted-cubic-tensor-projection-fp64-solve-passed-tiny) · [Copy provenance](../../PROVENANCE.json)

