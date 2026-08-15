import numpy as np
import pytest
from demo_scenarios import DEFAULT_THRESHOLDS, SCENARIOS, make_demo_data
from timdr_battery_fusion import TIMDRBatteryFusion

fusion = TIMDRBatteryFusion()
REQUIRED = ["voltage", "current", "temperature", "resistance"]


def test_wszystkie_scenariusze_maja_prog_i_opis():
    for name in SCENARIOS:
        assert name in DEFAULT_THRESHOLDS


def test_nieznany_scenariusz_rzuca_czytelny_blad():
    with pytest.raises(ValueError):
        make_demo_data("nie_istnieje")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenariusz_generuje_poprawne_dane_ze_wszystkimi_4_czujnikami(name):
    t, sensors = make_demo_data(name)
    assert len(t) > 0
    assert set(sensors.keys()) == set(REQUIRED)
    for arr in sensors.values():
        assert len(arr) == len(t)
        assert np.all(np.isfinite(arr))


def test_normal_cycling_wykrywa_rytm():
    """Regresja: pierwsza wersja (prad/napiecie w kwadratowym cyklu
    ladowanie/rozladowanie) dawala E(t) prawie stale (std=0.036) - L2
    fuzja dwoch kanalow o tej samej amplitudzie w obu fazach cyklu traci
    informacje o fazie. Po zmianie na sinusoidalny przebieg E(t)
    faktycznie oscyluje (choc zrektyfikowane - wykryty okres to POLOWA
    prawdziwego okresu cyklu, ten sam mechanizm co w
    TIMDR-Industrial-Fusion)."""
    t, s = make_demo_data("normal_cycling")
    E, _ = fusion.fuse(t, s["voltage"], s["current"], s["temperature"], s["resistance"])
    periods, score = fusion.rhythm(E, max_lag=60, power_thresh=0.4)
    assert 20 in periods  # period=40 w generatorze, wykryta polowa = 20
    assert score > 0.4


def test_capacity_fade_wykrywa_trend_i_daje_skonczone_ttf():
    from timdr_battery_predict import TIMDRBatteryPredict
    predict = TIMDRBatteryPredict()
    t, s = make_demo_data("capacity_fade")
    E, _ = fusion.fuse(t, s["voltage"], s["current"], s["temperature"], s["resistance"])
    slopes, _ = fusion.trend(t, E, window=20)
    assert slopes[-1] > 0
    ttf, _, _ = predict.predict_failure(t, E, threshold=DEFAULT_THRESHOLDS["capacity_fade"])
    assert 0 < ttf < float("inf")


def test_thermal_event_wykrywa_twist_i_anomalie():
    t, s = make_demo_data("thermal_event")
    E, _ = fusion.fuse(t, s["voltage"], s["current"], s["temperature"], s["resistance"])
    tw_idx, _ = fusion.twist(t, E)
    an_idx, _ = fusion.anomalies(E)
    assert len(tw_idx) > 0
    assert len(an_idx) > 0


def test_loose_connection_wykrywa_rytm_i_twist():
    t, s = make_demo_data("loose_connection")
    E, _ = fusion.fuse(t, s["voltage"], s["current"], s["temperature"], s["resistance"])
    periods, score = fusion.rhythm(E, max_lag=60, power_thresh=0.4)
    tw_idx, _ = fusion.twist(t, E)
    assert 20 in periods
    assert len(tw_idx) > 0


def test_internal_short_wykrywa_anomalie_i_daje_skonczone_ttf():
    from timdr_battery_predict import TIMDRBatteryPredict
    predict = TIMDRBatteryPredict()
    t, s = make_demo_data("internal_short")
    E, _ = fusion.fuse(t, s["voltage"], s["current"], s["temperature"], s["resistance"])
    an_idx, _ = fusion.anomalies(E)
    assert len(an_idx) > 0
    ttf, _, _ = predict.predict_failure(t, E, threshold=DEFAULT_THRESHOLDS["internal_short"])
    assert 0 <= ttf < float("inf")
