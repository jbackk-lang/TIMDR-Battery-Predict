import numpy as np
class TIMDRBattery:
    def __init__(self, mad_scale=1.4826, decay=0.98):
        self.mad_scale = mad_scale
        self.decay = decay
    def _mad_z(self, x):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * self.mad_scale
        if mad == 0:
            span = np.max(x) - np.min(x)
            if span == 0:
                return np.zeros_like(x)
            return (x - med) / (span / 4.0)
        return (x - med) / mad
    def fuse(self, t, voltage, current, temperature, resistance):
        t = np.asarray(t, float)
        X = np.column_stack([voltage, current, temperature, resistance])
        Z = np.column_stack([self._mad_z(X[:, i]) for i in range(X.shape[1])])
        E = np.sqrt(np.sum(Z**2, axis=1))
        return E, Z
    def twist(self, t, E):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        dE = np.gradient(E, t)
        ddE = np.gradient(dE, t)
        z = np.abs(self._mad_z(ddE))
        idx = np.where(z > 3.5)[0]
        return idx, z
    def trend(self, t, E, window=30):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        tc = t - t.mean()
        slopes = np.zeros_like(E)
        n = len(t)
        for i in range(n):
            j0 = max(0, i - window + 1)
            tt = tc[j0:i+1]
            ee = E[j0:i+1]
            A = np.column_stack([tt, np.ones_like(tt)])
            a, b = np.linalg.lstsq(A, ee, rcond=None)[0]
            slopes[i] = a
        z = self._mad_z(slopes)
        return slopes, z
    def anomalies(self, E):
        z = np.abs(self._mad_z(E))
        for i in range(1, len(z)):
            z[i] = max(z[i], z[i-1] * self.decay)
        idx = np.where(z > 3.0)[0]
        return idx, z
    def rhythm(self, E, max_lag=120, power_thresh=0.4):
        E = np.asarray(E, float)
        E = E - np.linspace(E[0], E[-1], len(E))
        E = E - np.mean(E)
        n = len(E)
        ac = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                ac[lag] = np.dot(E, E) / n
            else:
                overlap = n - lag
                if overlap <= 0:
                    break
                ac[lag] = np.dot(E[:-lag], E[lag:]) / overlap
        if ac[0] == 0:
            return [], 0.0
        ac /= ac[0]
        lags = np.arange(1, len(ac))
        power = ac[1:]
        dom = np.where(power >= power_thresh)[0]
        if dom.size == 0:
            return [], 0.0
        return lags[dom].tolist(), float(power[dom].max())
    def predict_failure(self, t, E, threshold=3.0):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        tc = t - t.mean()
        A = np.column_stack([tc, np.ones_like(tc)])
        a, b = np.linalg.lstsq(A, E, rcond=None)[0]
        if a <= 0:
            return np.inf
        ttf = (threshold - b) / a
        return float(max(ttf, 0.0))
    def health_score(self, E):
        z = np.abs(self._mad_z(E))
        for i in range(1, len(z)):
            z[i] = max(z[i], z[i-1] * self.decay)
        score = np.clip(np.max(z) / 5.0, 0.0, 1.0)
        return float(1.0 - score)
