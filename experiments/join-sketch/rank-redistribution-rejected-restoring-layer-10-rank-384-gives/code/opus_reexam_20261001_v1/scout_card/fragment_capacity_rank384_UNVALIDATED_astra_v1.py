# Source-only capacity candidate; native execution and current bill not validated.
# Changes only the L10 join rank. L7/L13 ranks and ru remain the parent's values.
# Rank-specific cache paths may fall back; the candidate needs its own census,
# setup/prebind and current native bill measurement before any stacking decision.
if _R320_RULE != {7: 384, 10: 320, 13: 384}:
    raise RuntimeError('Astra capacity route requires the current 384/320/384 parent')
_R320_RULE = {7: 384, 10: 384, 13: 384}
