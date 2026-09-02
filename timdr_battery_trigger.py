# timdr_battery_trigger.py
# ============================================
# TIMDR Battery Trigger Module
# ============================================
#
# ROLA: czujnik sygnałowy — NIE model predykcyjny (do tego służy
# TIMDRBatteryPredict.predict_failure()/health_score()). Dispatcher nad
# już przetestowanym TIMDRBatteryFusion (twist/anomalies) i
# TIMDRBatteryPredict (predict_failure) — jedyna jego robota: powiedzieć,
# KTÓRY typ zdarzenia się odpalił i GDZIE. Ten sam wzorzec i ten sam
# podział priorytetów co w siostrzanym repo TIMDR-Industrial-Predict
# (timdr_industrial_trigger.py) — tam samo brakujące ogniwo: api.py::
# api_analyze() liczy twist_idx/anomaly_idx/ttf/health_score jako
# odrębne, równoległe pola, ale nic ich nie łączy w jedno zdarzenie.
#
# Priorytet: FAILURE_IMMINENT (przewidywany czas do awarii — TTF —
# poniżej `alert_ttf_seconds`) > STRUCTURE (twist — nagła zmiana energii
# stanu E(t)) > ANOMALY (pojedyncza statystyczna anomalia w E(t)) > NONE.

from enum import Enum

import numpy as np

from timdr_battery_fusion import TIMDRBatteryFusion
from timdr_battery_predict import TIMDRBatteryPredict


class BatteryTriggerType(Enum):
    FAILURE_IMMINENT = "failure_imminent"
    STRUCTURE = "structure_twist"
    ANOMALY = "anomaly"
    NONE = "none"


class BatteryTriggerResult:
    def __init__(self, triggered=False, trigger_type=BatteryTriggerType.NONE,
                 location=None, message=""):
        self.triggered = triggered
        self.trigger_type = trigger_type
        self.location = location
        self.message = message

    def as_dict(self):
        return {
            "triggered": self.triggered,
            "type": self.trigger_type.value,
            "location": self.location,
            "message": self.message,
        }


class BatteryTrigger:
    """
    Dispatcher nad TIMDRBatteryFusion.twist()/anomalies() i
    TIMDRBatteryPredict.predict_failure(). `fusion`/`predictor` można
    wstrzyknąć (np. w testach) - domyślnie tworzą prawdziwe instancje.

    UWAGA: `twist()`/`anomalies()` w TIMDRBatteryFusion mają WŁASNE,
    zakodowane na stałe progi (3.5 / 3.0) - identycznie jak w
    TIMDR-Industrial-Fusion - więc ten dispatcher świadomie NIE przyjmuje
    parametrów progowych, których i tak nie dałoby się przekazać dalej
    (martwy parametr konstruktora - błąd znaleziony wcześniej w
    TIMDR-Security-Module). Jedyny prawdziwy próg dostrajalny tutaj to
    `alert_ttf_seconds` (dla predict_failure()) i `threshold`/`window`
    przekazywane do predict_failure() przy każdym `analyze()`.
    """

    def __init__(self, alert_ttf_seconds=3600.0, fusion=None, predictor=None):
        self.fusion = fusion if fusion is not None else TIMDRBatteryFusion()
        self.predictor = predictor if predictor is not None else TIMDRBatteryPredict()
        self.alert_ttf_seconds = alert_ttf_seconds
        self.last_result = BatteryTriggerResult()

    def analyze(self, t, E, threshold=3.0, window=60):
        ttf, _ttf_lin, _ttf_exp = self.predictor.predict_failure(
            t, E, threshold=threshold, window=window,
        )
        if np.isfinite(ttf) and ttf <= self.alert_ttf_seconds:
            loc = int(len(t) - 1) if len(t) else None
            return self._set_result(
                True, BatteryTriggerType.FAILURE_IMMINENT, loc,
                f"Przewidywany czas do przekroczenia progu: {ttf:.0f}s."
            )

        twist_idx, _tw_z = self.fusion.twist(t, E)
        if len(twist_idx):
            loc = int(min(twist_idx))
            return self._set_result(
                True, BatteryTriggerType.STRUCTURE, loc,
                "Nagła zmiana energii stanu E(t) (twist)."
            )

        anomaly_idx, _an_z = self.fusion.anomalies(E)
        if len(anomaly_idx):
            loc = int(min(anomaly_idx))
            return self._set_result(
                True, BatteryTriggerType.ANOMALY, loc,
                "Pojedyncza statystyczna anomalia w E(t)."
            )

        return self._set_result(
            False, BatteryTriggerType.NONE, None,
            "Brak wykrytego zdarzenia sygnałowego."
        )

    def _set_result(self, triggered, trigger_type, location, message):
        self.last_result = BatteryTriggerResult(triggered, trigger_type, location, message)
        return self.last_result

    def get_last(self):
        return self.last_result
