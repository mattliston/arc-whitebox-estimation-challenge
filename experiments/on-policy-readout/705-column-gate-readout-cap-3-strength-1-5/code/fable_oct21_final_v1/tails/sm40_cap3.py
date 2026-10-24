
# fable 2026-10-21 variant sm40_cap3
class _Sm40CapPlans(dict):
    def get(self, key, default=None):
        if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP:
            return None
        return dict.get(self, key, default)


_SM40_CAP = 3
_ASTRA_SM40_PLANS = _Sm40CapPlans(_ASTRA_SM40_PLANS)
