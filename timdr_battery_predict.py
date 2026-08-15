"""
timdr_battery_predict.py — TIMDR Battery Predict
===================================================
Predykcyjne utrzymanie ogniwa/pakietu: model degradacji E(t), prognoza
czasu do awarii (TTF) i wynik zdrowia (health_score).

Poprawiona wersja kodu ze zgłoszenia (`timdr_battery_predict_original.py`).
Zweryfikowano, że oryginał reprodukował dokładnie te same 3 błędy co
przed-fix wersja TIMDR-Industrial-Predict — patrz README.md.
"""

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

    def degradation_model(self, t, E, window=60):
        """
        POPRAWKA 1 (niestabilność zależna od długości historii): oryginalny
        kod dopasowywał regresję do CAŁEJ historii E(t), nie do ostatniego
        okna. Dokładnie jak w Industrial-Predict, TTF zależy wtedy silnie
        od tego, ile "zdrowej" historii jest w danych - dodano `window`
        (domyślnie 60 próbek) ograniczające dopasowanie do niedawnych
        danych.

        POPRAWKA 2 (numeryczne niedokondycjonowanie): oryginalny kod liczył
        `lstsq` na surowym, nieucentrowanym `t`. Dla realnych znaczników
        czasu (epoch, rzędu 1.7e9) to daje katastrofalnie złe dopasowanie
        (patrz README - test dawał TTF ~2.5 miliarda jednostek zamiast
        ~93). Naprawiono centrowaniem `t0 = t[0]`, dopasowanie na
        `t_rel = t - t0`.

        Zwraca teraz 3-elementową krotkę `(a,b),(ae,be),t0` (zmiana API
        względem oryginału, który zwracał 2-elementową).
        """
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        n = len(t)
        j0 = max(0, n - window)
        t_win = t[j0:]
        E_win = E[j0:]

        t0 = t_win[0] if len(t_win) else 0.0
        t_rel = t_win - t0

        A = np.column_stack([t_rel, np.ones_like(t_rel)])
        a, b = np.linalg.lstsq(A, E_win, rcond=None)[0]

        Epos = np.clip(E_win, 1e-6, None)
        logE = np.log(Epos)
        ae, be = np.linalg.lstsq(A, logE, rcond=None)[0]

        return (a, b), (ae, be), t0

    def predict_failure(self, t, E, threshold=3.0, window=60):
        """
        POPRAWKA (bug krytyczny: absolutny zamiast relatywnego TTF):
        oryginalny kod rozwiązywał E(t)=threshold względem tej samej osi
        `t`, co dane wejściowe, i zwracał to jako "czas do awarii" - w
        rzeczywistości zwracał WSPÓŁRZĘDNĄ na osi t, nie czas OD TERAZ.

        Zweryfikowano: przesunięcie wszystkich `t` o +1000 (np. inny punkt
        odniesienia zegara) zmieniało zwracany TTF o dokładnie +1000,
        mimo że fizyczna sytuacja (ile czasu do awarii OD OSTATNIEGO
        pomiaru) się nie zmieniła. Z realnymi znacznikami epoki dawało to
        TTF rzędu miliardów jednostek zamiast rozsądnej liczby.

        Naprawiono: liczymy `t_ref = t[-1] - t0` (pozycja ostatniej
        próbki względem centrowania z `degradation_model`) i odejmujemy
        ją od momentu przecięcia progu, więc wynik to naprawdę czas OD
        OSTATNIEGO pomiaru, niezależny od punktu odniesienia zegara.
        Ujemny TTF (już po awarii) jest ucinany do 0.
        """
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        (a, b), (ae, be), t0 = self.degradation_model(t, E, window=window)
        t_ref = t[-1] - t0

        if a > 0:
            ttf_linear = (threshold - b) / a - t_ref
        else:
            ttf_linear = np.inf
        if ae > 0:
            ttf_exp = (np.log(threshold) - be) / ae - t_ref
        else:
            ttf_exp = np.inf

        ttf_linear = max(0.0, ttf_linear) if np.isfinite(ttf_linear) else ttf_linear
        ttf_exp = max(0.0, ttf_exp) if np.isfinite(ttf_exp) else ttf_exp
        ttf = min(ttf_linear, ttf_exp)

        if E[-1] >= threshold:
            return 0.0, float(ttf_linear), float(ttf_exp)
        return float(ttf), float(ttf_linear), float(ttf_exp)

    def health_score(self, E, threshold=3.0, window=20):
        """
        POPRAWKA (bug krytyczny: trwałe "zatrucie" jednym starym skokiem):
        oryginalny kod liczył `max(|MAD-z(E)|)` na CAŁEJ historii E - jeden
        stary skok na początku nagrania dawał `health_score=0.0` NA
        ZAWSZE, nawet gdy wszystkie kolejne setki próbek były normalne.

        Zweryfikowano: pojedynczy skok w próbce 50 (z 500) dawał
        health_score=0.000 mimo 450 zdrowych próbek po nim.

        Naprawiono (identycznie jak w Industrial-Predict): liczymy medianę
        z OSTATNIEGO okna (domyślnie 20 próbek) i porównujemy ją
        bezpośrednio do tego samego `threshold`, co `predict_failure` -
        wynik odzwierciedla AKTUALNY stan, nie najgorszy moment w całej
        historii, i jest spójny z tym samym progiem awarii.
        """
        E = np.asarray(E, float)
        if E.size == 0:
            return 1.0
        recent = E[-window:] if len(E) >= 1 else E
        level = float(np.median(recent))
        score = np.clip(level / threshold, 0.0, 1.0) if threshold > 0 else 0.0
        return float(1.0 - score)
