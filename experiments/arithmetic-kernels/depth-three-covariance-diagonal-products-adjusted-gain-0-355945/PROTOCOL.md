# One fixed D3 implementation follow-on

Root selected depth3 once from the complete fixed D4/D3/D2 source-price table
762578cb8d6868338f860d16e85e785d5ae278de0b3edad5f03fe145a3b811ce.
This is an explicitly outcome-informed implementation follow-on to D5, whose
TRAIN2 adjusted gain was1.1441358% but boundary residual.357850751 exceeded.35.
Its failed receipt remains unchanged; no old study or gate is overridden.

Append only two depth-map/size assignments and the matching group guard to
exact D5 source42e50cdc. Five active sides change5→3; four historical sides
stay6. The nine-key pool, groups, arena offsets/allocation, side constructor,
forward/matmul/unwind core, copies, resets, E1/YNT and parameters are unchanged.
The group wrapper calls the original captured arithmetic core directly.

Analytical helper707714016→738303136F and422→378 calls. Fourteen helpers add
428247680F (~0.160% current bill) and remove616 calls. Active setup saves
25354560F and82 calls. Full arena stays1566408704B; smaller views release no
allocation. No timing or accuracy improvement is claimed from source prices.

Root primitive: NATIVE -B probe.py BUILD_SHA EXCLUSIVE_RESULT_JSON,120s,
7.5GB, zero added swap, BLAS1. Reuse the D5 integer/Gaussian fixtures and
seeds, exact warm bills/calls, within-arm repeat bytes, input immutability,
Gaussian error<=2e-4, setup counts682/649+9, offsets/sharing and reset checks.
Depth changes rounding; cross-parent bytes are not required.

After primitive, use the same frozen prospective source-specific research gates
as D5: fresh TRAIN2 versusS33 (mean<=−.5%,>=1win,maxreg<=.5%,all16finite,L0exact,
YNT crop>=1/rows>=64/retired), then unchanged early boundary (.35 residual,
105s projection,4.1s setup,5pp CPU agreement). Only a boundary pass permits
TRAIN6 and pooled8 (mean<=−.1%,>=6wins,CIupper<0), with additional6 separately
<=−.5%. Final40 bank still requires>=1%,>=28wins,CIupper<0,maxreg<=10%, plus
full serving qualification. Root alone admits numerical jobs in that order.
No D4/D2 variant, changed step/threshold or same-source retry follows a failure.
The reused panels are not pristine holdouts. No numerical work is run by build.
