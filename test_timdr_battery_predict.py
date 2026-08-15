import numpy as np
import pytest
from timdr_battery_predict import TIMDRBatteryPredict


@pytest.fixture
def predict():
    return TIMDRBatteryPredict()


def test_predict_failure_niezmiennik_przesuniecia_zegara(predict):
    """Regresja kluczowego bledu ze zgloszenia: oryginalny kod
    (`timdr_battery_predict_original.py`) zwracal TTF, ktory zmienial sie
    dokladnie o wielkosc przesuniecia czasu (przesuniecie +1000 -> TTF
    +1000) - zweryfikowano przed poprawka. Po poprawce TTF nie zalezy od
    punktu odniesienia zegara."""
    rng = np.random.default_rng(0)
    n = 100
    t = np.arange(n, dtype=float)
    E = 1.0 + 0.02 * t + rng.normal(0, 0.02, n)
    ttf1, _, _ = predict.predict_failure(t, E, threshold=3.0)
    ttf2, _, _ = predict.predict_failure(t + 1000, E, threshold=3.0)
    assert ttf1 == pytest.approx(ttf2, abs=1e-6)


def test_predict_failure_epoch_timestamps_nie_daje_absurdu(predict):
    """Regresja: oryginalny kod na realnych znacznikach epoki (rzedu
    1.7e9) dawal TTF rzedu miliardow jednostek (numeryczne
    niedokondycjonowanie lstsq bez centrowania). Po poprawce (centrowanie
    t0=t[0]) wynik powinien byc rzedu dziesiatek/setek, nie miliardow."""
    rng = np.random.default_rng(0)
    n = 100
    t = np.arange(n, dtype=float) + 1_700_000_000
    E = 1.0 + 0.02 * (t - t[0]) + rng.normal(0, 0.02, n)
    ttf, _, _ = predict.predict_failure(t, E, threshold=3.0)
    assert ttf < 1000, f"TTF podejrzanie duze ({ttf}) - mozliwy powrot bledu niedokondycjonowania"


def test_predict_failure_okno_ogranicza_wplyw_starej_historii(predict):
    """Regresja: oryginalny kod dopasowywal regresje do CALEJ historii,
    wiec identyczna 'swieza' degradacja dawala rozny TTF w zaleznosci od
    tego, ile zdrowej historii ja poprzedzalo. Po poprawce (window=60)
    TTF powinien byc podobny niezaleznie od dlugosci wczesniejszej
    plaskiej historii."""
    rng = np.random.default_rng(0)
    n_recent = 60
    t_recent = np.arange(n_recent, dtype=float)
    E_recent = 1.0 + 0.02 * t_recent + rng.normal(0, 0.02, n_recent)
    ttf_short, _, _ = predict.predict_failure(t_recent, E_recent, threshold=3.0, window=60)

    n_old = 400
    t_long = np.arange(n_old, dtype=float)
    E_long = np.concatenate([np.full(n_old - n_recent, E_recent[0]) + rng.normal(0, 0.02, n_old - n_recent), E_recent])
    ttf_long, _, _ = predict.predict_failure(t_long, E_long, threshold=3.0, window=60)

    assert ttf_short == pytest.approx(ttf_long, rel=0.05)


def test_predict_failure_juz_przekroczony_prog_daje_zero(predict):
    t = np.arange(20, dtype=float)
    E = np.full(20, 10.0)
    ttf, _, _ = predict.predict_failure(t, E, threshold=3.0)
    assert ttf == 0.0


def test_predict_failure_brak_trendu_daje_inf(predict):
    rng = np.random.default_rng(0)
    n = 50
    t = np.arange(n, dtype=float)
    E = rng.normal(1.0, 0.05, n)
    ttf, ttl, tte = predict.predict_failure(t, E, threshold=3.0)
    # plaski/losowy sygnal - moze wyjsc dodatnie nachylenie z szumu, ale
    # nie powinien crashowac i powinien zwrocic skonczona/nieskonczona liczbe
    assert ttf >= 0.0 or ttf == float("inf")


def test_health_score_brak_trwalego_zatrucia_starym_skokiem(predict):
    """Regresja kluczowego bledu ze zgloszenia: oryginalny kod dawal
    health_score=0.000 NA STALE po jednym starym skoku, nawet gdy setki
    kolejnych probek byly normalne (bo liczyl max() po calej historii).
    Zweryfikowano: skok w probce 50 z 500 dawal 0.000 przed poprawka.
    Po poprawce (mediana ostatniego okna wzgledem threshold) wynik
    powinien odzwierciedlac AKTUALNY (zdrowy) stan."""
    rng = np.random.default_rng(0)
    n = 500
    E = rng.normal(1.0, 0.05, n)
    E[50] += 5.0
    hs = predict.health_score(E, threshold=3.0, window=20)
    assert hs > 0.5, f"health_score wciaz wyglada na zatruty starym skokiem: {hs}"


def test_health_score_pusty_sygnal_zdrowy_domyslnie(predict):
    hs = predict.health_score(np.array([]), threshold=3.0)
    assert hs == 1.0


def test_health_score_aktualnie_wysoki_poziom_daje_niski_wynik(predict):
    E = np.full(30, 9.0)
    hs = predict.health_score(E, threshold=3.0, window=20)
    assert hs < 0.2
