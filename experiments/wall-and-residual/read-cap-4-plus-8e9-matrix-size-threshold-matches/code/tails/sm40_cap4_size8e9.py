
# fable 2026-10-21 variant sm40_cap4_size8e9: drop the Smirnov plans with m*k*n >= 8.0e9, then cap reads per predict at 4
_ASTRA_SM40_PLANS = {k: v for k, v in _ASTRA_SM40_PLANS.items() if not (len(k) >= 3 and k[0] * k[1] * k[2] >= 8.0e9)}
class _Sm40CapPlans(dict):
    def get(self, key, default=None):
        if _ASTRA_SM40_STATS.get('calls', 0) >= _SM40_CAP:
            return None
        return dict.get(self, key, default)


_SM40_CAP = 4
_ASTRA_SM40_PLANS = _Sm40CapPlans(_ASTRA_SM40_PLANS)
