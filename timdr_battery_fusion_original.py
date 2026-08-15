import numpy as np
class TIMDRBatteryFusion:
    def __init__(self, mad_scale=1.4826):
        self.mad_scale = mad_scale
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
    def _align(self, t, sensors):
        t = np.asarray(t, float)
        out = []
        for s in sensors:
            s = np.asarray(s, float)
            if len(s) != len(t):
                ti = np.linspace(t.min(), t.max(), len(s))
                si = np.interp(t, ti, s)
                out.append(si)
            else:
                out.append(s)
        return np.column_stack(out)
    def fuse(self, t, voltage, current, temperature, resistance):
        t = np.asarray(t, float)
        X = self._align(t, [voltage, current, temperature, resistance])
        Z = np.column_stack([self._mad_z(X[:, i]) for i in range(X.shape[1])])
        E = np.sqrt(np.sum(Z**2, axis=1))
        return E, Z
    def twist(self, t, E):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        if len(t) < 3:
            return np.array([], int), np.zeros_like(E)
        dE = np.gradient(E, t)
        ddE = np.gradient(dE, t)
        z = np.abs(self._mad_z(ddE))
        idx = np.where(z > 3.5)[0]
        return idx, z
    def trend(self, t, E, window=30):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        n = len(t)
        slopes = np.zeros_like(E)
        if n < 2:
            return slopes, np.zeros_like(slopes)
        for i in range(n):
            j0 = max(0, i - window + 1)
            tt = t[j0:i+1]
            ee = E[j0:i+1]
            A = np.column_stack([tt, np.ones_like(tt)])
            a, b = np.linalg.lstsq(A, ee, rcond=None)[0]
            slopes[i] = a
        z = self._mad_z(slopes)
        return slopes, z
    def anomalies(self, E):
        E = np.asarray(E, float)
        if E.size == 0:
            return np.array([], int), np.zeros_like(E)
        z = np.abs(self._mad_z(E))
        idx = np.where(z > 3.0)[0]
        return idx, z
    def rhythm(self, E, max_lag=120, power_thresh=0.4):
        E = np.asarray(E, float)
        n = len(E)
        if n < 3:
            return [], 0.0
        E = E - np.mean(E)
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
    def fusion_score(self, twist_z, trend_z, anomaly_z, rhythm_score):
        return float(
            0.4 * (np.max(twist_z) if twist_z.size else 0.0) +
            0.3 * (np.max(trend_z) if trend_z.size else 0.0) +
            0.2 * (np.max(anomaly_z) if anomaly_z.size else 0.0) +
            0.1 * rhythm_score
        )
