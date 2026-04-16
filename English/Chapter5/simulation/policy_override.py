"""既存の政策関数に上書き制約を加えるラッパークラス群。"""

import numpy as np

class PolicyOverride:
    """
    既存モデルの predict(mean_X, var_X, nu_t, t=None, actor=None, zones=None)
    を横取りし、(t, zone, actor) 単位で値を上書きする。
    """
    def __init__(self, base_model):
        self.base = base_model
        self._overrides = {}  # key=(t, zone, actor)

    def set_override(self, t, zone, actor, value):
        self._overrides[(int(t), int(zone), str(actor))] = float(value)

    def clear(self):
        self._overrides.clear()

    def predict(self, X, nu, t=None, actor=None, zones=None):
        y = self.base.predict(X, nu).astype(float)
        if t is None or actor is None:
            return y
        Z = np.arange(len(y)) if zones is None else np.asarray(zones)
        for idx, z in enumerate(Z):
            key = (int(t), int(z), str(actor))
            if key in self._overrides:
                y[idx] = self._overrides[key]
        return y
    
class PolicyLocationControl:
    """
    既存モデルの predict(mean_X, var_X, nu_t, t=None, actor=None, zones=None)
    を横取りし、zone 単位で立地規制 (上限を適用)を適用する。
    """
    def __init__(self, base_model):
        self.base = base_model
        self._location_caps = {}  # key=zone

    def set_location_cap(self, zone, cap_value):
        self._location_caps[int(zone)] = float(cap_value)

    def clear(self):
        self._location_caps.clear()

    def predict(self, X, nu, t=None, actor="dev", zones=None):
        y = self.base.predict(X, nu).astype(float)
        if actor is None or actor != "dev":
            return y
        Z = np.arange(len(y)) if zones is None else np.asarray(zones)
        for idx, z in enumerate(Z):
            if int(z) in self._location_caps:
                y[idx] = min(y[idx], self._location_caps[int(z)])
        return y
    
class PolicyTransportControl:
    """
    既存モデルの predict(mean_X, var_X, nu_t, t=None, actor=None, zones=None)
    を横取りし、zone 単位で立地規制 (上限を適用)を適用する。
    """
    def __init__(self, base_model):
        self.base = base_model
        self._tranport_caps = {}  # key=zone

    def set_transport_cap(self, zone, cap_value):
        self._tranport_caps[int(zone)] = float(cap_value)

    def clear(self):
        self._tranport_caps.clear()

    def predict(self, X, nu, t=None, actor="dev", zones=None):
        y = self.base.predict(X, nu).astype(float)
        if actor is None or actor != "dev":
            return y
        Z = np.arange(len(y)) if zones is None else np.asarray(zones)
        for idx, z in enumerate(Z):
            if int(z) in self._tranport_caps:
                y[idx] = min(y[idx], self._tranport_caps[int(z)])
        return y
