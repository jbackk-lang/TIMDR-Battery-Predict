import numpy as np
import pytest
from timdr_battery_fusion import TIMDRBatteryFusion


@pytest.fixture
def fusion():
    return TIMDRBatteryFusion()


def _four(n, seed=0):
    rng = np.random.default_rng(seed)
    voltage = rng.normal(3.7, 0.01, n)
    current = rng.normal(1.0, 0.05, n)
    temperature = rng.normal(25, 1, n)
    resistance = rng.normal(0.05, 0.002, n)
    return voltage, current, temperature, resistance


def test_fuse_zwraca_E_nieujemne(fusion):
    t = np.arange(50, dtype=float)
    v, c, te, r = _four(50)
    E, Z = fusion.fuse(t, v, c, te, r)
    assert len(E) == 50
    assert np.all(E >= 0)
    assert Z.shape == (50, 4)


def test_fuse_stala_kolumna_nie_wnosi_energii(fusion):
    """Kanal o zerowej wariancji (idealnie staly, np. czujnik niezwiazany
    ze scenariuszem) powinien wnosic dokladnie 0 do E - patrz uzasadnienie
    w demo_scenarios.py."""
    n = 60
    t = np.arange(n, dtype=float)
    v, c, _, r = _four(n)
    temperature_const = np.full(n, 25.0)
    E, Z = fusion.fuse(t, v, c, temperature_const, r)
    assert np.allclose(Z[:, 2], 0.0)


def test_twist_krotki_sygnal_nie_crashuje(fusion):
    for n in [0, 1, 2]:
        t = np.arange(n, dtype=float)
        E = np.ones(n)
        idx, z = fusion.twist(t, E)
        assert len(idx) == 0
        assert len(z) == n


def test_twist_wykrywa_nagla_zmiane(fusion):
    n = 200
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(1)
    E = rng.normal(1.0, 0.02, n)
    E[100:110] += np.linspace(0, 5, 10)  # nagly wzrost
    idx, z = fusion.twist(t, E)
    assert len(idx) > 0


def test_trend_krotki_sygnal_nie_crashuje(fusion):
    for n in [0, 1]:
        t = np.arange(n, dtype=float)
        E = np.ones(n)
        slopes, z = fusion.trend(t, E)
        assert len(slopes) == n


def test_trend_wykrywa_narastajacy_sygnal(fusion):
    n = 100
    t = np.arange(n, dtype=float)
    E = np.linspace(0, 10, n)
    slopes, z = fusion.trend(t, E, window=20)
    assert slopes[-1] > 0


def test_anomalies_pusty_sygnal(fusion):
    idx, z = fusion.anomalies(np.array([]))
    assert len(idx) == 0


def test_anomalies_wykrywa_wyrazny_skok(fusion):
    rng = np.random.default_rng(2)
    n = 200
    E = rng.normal(1.0, 0.05, n)
    E[100] += 3.0
    idx, z = fusion.anomalies(E)
    assert 100 in idx


def test_rhythm_brak_okresowosci_na_krotkim_sygnale(fusion):
    for n in [0, 1, 2]:
        periods, score = fusion.rhythm(np.ones(n))
        assert periods == []
        assert score == 0.0


def test_rhythm_regresja_czysty_trend_nie_daje_falszywego_rytmu(fusion):
    """Regresja kluczowego bledu ze zgloszenia: oryginalny kod ze
    zgloszenia (`timdr_battery_fusion_original.py`) dawal na tym samym
    sygnale score=0.989 i 48 'wykrytych okresow' - zweryfikowano przed
    poprawka. Po poprawce (pelny detrend + lokalne maksima) oczekiwane
    jest [], 0.0."""
    rng = np.random.default_rng(0)
    n = 200
    t = np.arange(n, dtype=float)
    E = np.linspace(0, 10, n) + rng.normal(0, 0.1, n)
    periods, score = fusion.rhythm(E, max_lag=60, power_thresh=0.4)
    assert periods == []
    assert score == 0.0


def test_rhythm_wykrywa_prawdziwa_okresowosc(fusion):
    n = 200
    idx = np.arange(n, dtype=float)
    E = np.sin(2 * np.pi * idx / 15)
    periods, score = fusion.rhythm(E, max_lag=60, power_thresh=0.4)
    assert 15 in periods
    assert score > 0.4


def test_fusion_score_puste_tablice_nie_crashuja(fusion):
    score = fusion.fusion_score(np.array([]), np.array([]), np.array([]), 0.0)
    assert score == 0.0
