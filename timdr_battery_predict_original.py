import numpy as np
class TIMDRBatteryPredict:
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
    def degradation_model(self, t, E):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        A = np.column_stack([t, np.ones_like(t)])
        a, b = np.linalg.lstsq(A, E, rcond=None)[0]
        Epos = np.clip(E, 1e-6, None)
        logE = np.log(Epos)
        A2 = np.column_stack([t, np.ones_like(t)])
        ae, be = np.linalg.lstsq(A2, logE, rcond=None)[0]
        return (a, b), (ae, be)
    def predict_failure(self, t, E, threshold=3.0):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        (a, b), (ae, be) = self.degradation_model(t, E)
        if a > 0:
            ttf_linear = (threshold - b) / a
        else:
            ttf_linear = np.inf
        if ae > 0:
            ttf_exp = (np.log(threshold) - be) / ae
        else:
            ttf_exp = np.inf
        ttf = min(ttf_linear, ttf_exp)
        if E[-1] >= threshold:
            return 0.0, float(ttf_linear), float(ttf_exp)
        return float(ttf), float(ttf_linear), float(ttf_exp)
    def health_score(self, E):
        z = np.abs(self._mad_z(E))
        score = np.clip(np.max(z) / 5.0, 0.0, 1.0)
        return float(1.0 - score)
