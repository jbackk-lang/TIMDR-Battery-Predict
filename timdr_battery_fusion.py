"""
timdr_battery_fusion.py — TIMDR Battery Fusion
=================================================
Fuzja czujników ogniwa/pakietu (napięcie, prąd, temperatura, rezystancja
wewnętrzna) w jeden sygnał "energii stanu" E(t), plus standardowy zestaw
detektorów TIMDR (twist, trend, anomalie, rytm) na tym sygnale.

Ten plik to poprawiona wersja kodu ze zgłoszenia (`timdr_battery_fusion_original.py`).
Zweryfikowano, że oryginał reprodukował dokładnie ten sam błąd rytmu co
przed-fix wersja TIMDR-Industrial-Fusion — patrz README.md.
"""

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
            tt = t[j0:i + 1]
            ee = E[j0:i + 1]
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
        """
        POPRAWKA (bug krytyczny, identyczny jak w TIMDR-Industrial-Fusion):
        oryginalny kod ze zgłoszenia tylko odejmował średnią (`E - mean(E)`)
        i zgłaszał KAŻDY lag powyżej `power_thresh`, nie tylko lokalne
        maksima. Dla sygnału z TRENDEM (typowe dla degradacji ogniwa -
        rosnąca rezystancja, spadające napięcie pod obciążeniem) kolejne
        próbki są do siebie podobne z powodu gładkiego trendu, nie
        cykliczności.

        Zweryfikowano: czysty rosnący trend (bez śladu okresowości, z
        realistycznym szumem czujnika, n=200) dawał na oryginalnym kodzie
        `rhythm_score=0.989` i **48 "wykrytych okresów"**.

        Naprawiono dwutorowo (identycznie jak w Industrial-Fusion): (1)
        pełny detrend (nachylenie + wyraz wolny, nie tylko średnia) przed
        liczeniem autokorelacji, (2) zgłaszane są tylko LOKALNE MAKSIMA
        autokorelacji powyżej progu, nie każdy lag który go przekracza.
        Po poprawce ten sam test daje `[]`, `0.0`.
        """
        E = np.asarray(E, float)
        n = len(E)
        if n < 3:
            return [], 0.0

        t_idx = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t_idx, E, 1)
        E = E - (slope * t_idx + intercept)

        max_lag = min(max_lag, n - 1)
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

        peaks = [
            (i, float(ac[i])) for i in range(1, len(ac) - 1)
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] >= power_thresh
        ]
        if not peaks:
            return [], 0.0
        score = max(p for _, p in peaks)
        return [p for p, _ in peaks], score

    def fusion_score(self, twist_z, trend_z, anomaly_z, rhythm_score):
        def safe_max(x):
            x = np.asarray(x, float)
            return float(np.max(x)) if x.size else 0.0

        return float(
            0.4 * safe_max(twist_z) +
            0.3 * safe_max(trend_z) +
            0.2 * safe_max(anomaly_z) +
            0.1 * rhythm_score
        )
